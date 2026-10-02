# Copyright 2026 Fractalyze Inc. All rights reserved.
"""A Qwen3.8-27B full-attention layer for one decode token in one launch
(s2mk/csrc/qwen38_layer.cu), and the PyTorch model it is held to.

Every projection is asymmetric int4 in groups of 32 (s2mk/int4.py); the norms are bf16
and zero-centered, scaling by 1 + w. The KV cache is vLLM's paged cache, read
and written in place through a block table.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sglang.kernels.decode_mk import _ext, int4
from sglang.kernels.decode_mk.barrier import DEFAULT_TIMEOUT_NS, ErrorRecord, sync_words
from sglang.kernels.decode_mk.gemv import num_ctas
from sglang.kernels.decode_mk.thinker_attention import PagedCache

# The language model's config (Qwen3_5TextConfig): every 4th layer of 64 is
# full attention, and every layer has the dense MLP.
DIM, Q_HEADS, KV_HEADS, HEAD_DIM, FFN = 5120, 24, 4, 256, 17408
Q_DIM, KV_DIM = Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
# The kernel's fused rows: q, k, v, then the output gate.
QKV_ROWS = 2 * Q_DIM + 2 * KV_DIM
# partial_rotary_factor 0.25 of the head turns, at rope_theta 10⁷.
ROTARY_DIM = HEAD_DIM // 4
ROPE_THETA = 10_000_000.0
# Interleaved M-RoPE with sections 11/11/10 (vLLM's apply_interleaved_rope):
# frequency i turns at the height position when i mod 3 = 1 and i < 33, at
# the width position when i mod 3 = 2 and i < 30, else at the temporal one.
MROPE_SECTION = (11, 11, 10)

Int4Weight = int4.Int4Weight
Projection = int4.Projection


def cos_sin_table(positions: int, device: torch.device | str = "cuda") -> torch.Tensor:
    """vLLM's cos_sin_cache for the layer: bf16 [positions, ROTARY_DIM], each
    position's 32 cosines, then its 32 sines."""
    inv_freq = 1.0 / ROPE_THETA ** (torch.arange(0, ROTARY_DIM, 2, dtype=torch.float,
                                                 device=device) / ROTARY_DIM)
    freqs = torch.outer(torch.arange(positions, dtype=torch.float, device=device), inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).bfloat16()


def mrope_axes(device: torch.device | str = "cuda") -> torch.Tensor:
    """For each of the ROTARY_DIM / 2 frequencies, the index (0 temporal, 1
    height, 2 width) of the position it turns at."""
    i = torch.arange(ROTARY_DIM // 2, device=device)
    height = (i % 3 == 1) & (i < 3 * MROPE_SECTION[1])
    width = (i % 3 == 2) & (i < 3 * MROPE_SECTION[2])
    return torch.where(height, 1, torch.where(width, 2, 0))


def _normal(gen: torch.Generator, *shape: int) -> torch.Tensor:
    return torch.randn(*shape, generator=gen, device=gen.device)


def _random_proj(gen: torch.Generator, n: int, k: int) -> Int4Weight:
    return int4.quantize_asymmetric(_normal(gen, n, k) / k**0.5)


def _random_norm(gen: torch.Generator, n: int) -> torch.Tensor:
    return (0.1 * _normal(gen, n)).bfloat16()


@dataclass(frozen=True)
class MlpWeights:
    """The dense MLP every layer ends with. gate_proj's and up_proj's rows are
    kept interleaved as the kernel reads them, pair j in rows 2j and 2j + 1;
    each int4 row carries its own scales and zero points, so interleaving is
    exact."""
    post_norm: torch.Tensor  # bf16 [DIM]: post_attention_layernorm
    w13: Projection  # [2 × FFN, DIM]
    down_proj: Projection  # [DIM, FFN]
    eps: float = 1e-6

    @classmethod
    def of(cls, post_norm: torch.Tensor, gate_proj: Projection, up_proj: Projection,
           down_proj: Projection, eps: float = 1e-6) -> MlpWeights:
        """From the checkpoint's separate gate_proj and up_proj, both in one
        format."""
        def interleave(gate, up):
            return torch.stack([gate, up], dim=1).flatten(0, 1).contiguous()

        if isinstance(gate_proj, torch.Tensor):
            return cls(post_norm, interleave(gate_proj, up_proj), down_proj, eps)
        w13 = tuple(interleave(*parts) for parts in zip(gate_proj, up_proj))
        return cls(post_norm, w13, down_proj, eps)

    @classmethod
    def random(cls, gen: torch.Generator) -> MlpWeights:
        return cls(post_norm=_random_norm(gen, DIM), w13=_random_proj(gen, 2 * FFN, DIM),
                   down_proj=_random_proj(gen, DIM, FFN))


@dataclass(frozen=True)
class LayerWeights:
    """One full-attention layer's tensors as the checkpoint names and lays
    them out, with its MLP. q_proj's rows are each head's HEAD_DIM q rows, then
    its HEAD_DIM output-gate rows."""
    input_norm: torch.Tensor  # bf16 [DIM]
    q_proj: Projection  # [2 × Q_DIM, DIM]
    k_proj: Projection  # [KV_DIM, DIM]
    v_proj: Projection  # [KV_DIM, DIM]
    q_norm: torch.Tensor  # bf16 [HEAD_DIM]
    k_norm: torch.Tensor  # bf16 [HEAD_DIM]
    o_proj: Projection  # [DIM, Q_DIM]
    mlp: MlpWeights
    eps: float = 1e-6

    @classmethod
    def random(cls, seed: int) -> LayerWeights:
        gen = torch.Generator(device="cuda").manual_seed(seed)
        return cls(input_norm=_random_norm(gen, DIM), q_proj=_random_proj(gen, 2 * Q_DIM, DIM),
                   k_proj=_random_proj(gen, KV_DIM, DIM), v_proj=_random_proj(gen, KV_DIM, DIM),
                   q_norm=_random_norm(gen, HEAD_DIM), k_norm=_random_norm(gen, HEAD_DIM),
                   o_proj=_random_proj(gen, DIM, Q_DIM), mlp=MlpWeights.random(gen))


def fuse_qkv(w: LayerWeights) -> Projection:
    """q, k, v and the output gate's rows in the kernel's order
    (csrc/qwen38_layer.h), in the weights' format. Each int4 row carries its
    own scales and zero points, so regrouping rows is exact."""
    def fuse(q_proj, k, v):
        per_head = q_proj.view(Q_HEADS, 2, HEAD_DIM, -1)
        q, gate = per_head[:, 0].flatten(0, 1), per_head[:, 1].flatten(0, 1)
        return torch.cat([q, k, v, gate]).contiguous()

    if isinstance(w.q_proj, torch.Tensor):
        return fuse(w.q_proj, w.k_proj, w.v_proj)
    return tuple(fuse(*parts) for parts in zip(w.q_proj, w.k_proj, w.v_proj))


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, dtype: torch.dtype) -> torch.Tensor:
    """Qwen3_5RMSNorm: normalized in fp32, scaled by 1 + w, then cast."""
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * (1.0 + weight.float())).to(dtype)


def _linear(x: torch.Tensor, w: Projection) -> torch.Tensor:
    return F.linear(x, int4.dense(w).to(x.dtype))


@dataclass
class LayerOutput:
    hidden: torch.Tensor  # fp32 [DIM]: residual_in + the attention block's output
    residual: torch.Tensor  # fp32 [DIM]: hidden + the MLP's output
    key: torch.Tensor  # [KV_HEADS, HEAD_DIM]: this step's key, after QK-norm and RoPE
    value: torch.Tensor  # [KV_HEADS, HEAD_DIM]


def reference(w: LayerWeights, residual: torch.Tensor, cache: PagedCache, pos: int,
              positions: torch.Tensor, cos_sin: torch.Tensor,
              dtype: torch.dtype) -> LayerOutput:
    """HF's Qwen3_5DecoderLayer (full attention) in `dtype` (fp32 for ground
    truth, bf16 for the budget) on the dequantized weights, attending to the
    cache's positions [0, pos) and this step's key and value. The residual
    stream stays fp32."""
    x = residual.float()
    h = rms_norm(x, w.input_norm, w.eps, dtype)
    q, gate = _linear(h, w.q_proj).view(Q_HEADS, 2 * HEAD_DIM).chunk(2, dim=-1)
    q = rms_norm(q, w.q_norm, w.eps, dtype)
    k = rms_norm(_linear(h, w.k_proj).view(KV_HEADS, HEAD_DIM), w.k_norm, w.eps, dtype)
    v = _linear(h, w.v_proj).view(KV_HEADS, HEAD_DIM)
    turn_at = positions.long().to(cos_sin.device)[mrope_axes(cos_sin.device)]
    freq = torch.arange(ROTARY_DIM // 2, device=cos_sin.device)
    cos = cos_sin[turn_at, freq].to(dtype)
    sin = cos_sin[turn_at, ROTARY_DIM // 2 + freq].to(dtype)

    def rotate(x):
        x0, x1 = x[..., :ROTARY_DIM // 2], x[..., ROTARY_DIM // 2:ROTARY_DIM]
        return torch.cat([x0 * cos - x1 * sin, x1 * cos + x0 * sin, x[..., ROTARY_DIM:]],
                         dim=-1)

    q, k = rotate(q), rotate(k)
    past_k, past_v = cache.gather(pos)
    keys = torch.cat([past_k.to(dtype), k[None]])  # [pos + 1, KV_HEADS, HEAD_DIM]
    values = torch.cat([past_v.to(dtype), v[None]])
    group = torch.arange(Q_HEADS, device=q.device) // (Q_HEADS // KV_HEADS)
    scores = torch.einsum("hd,thd->ht", q.float(), keys[:, group].float()) / HEAD_DIM**0.5
    attn = torch.einsum("ht,thd->hd", torch.softmax(scores, -1), values[:, group].float())
    attn = attn.to(dtype) * torch.sigmoid(gate)
    hidden = x + _linear(attn.flatten(), w.o_proj).float()
    return LayerOutput(hidden, mlp_reference(w.mlp, hidden, dtype), k, v)


def mlp_reference(w: MlpWeights, hidden: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """hidden + HF's Qwen3_5MLP over the post-attention norm, in `dtype` on
    the dequantized weights, as fp32. `hidden` is [..., DIM]."""
    n = rms_norm(hidden, w.post_norm, w.eps, dtype)
    gate, up = _linear(n, w.w13).unflatten(-1, (FFN, 2)).unbind(-1)
    return hidden + _linear(F.silu(gate) * up, w.down_proj).float()


def mlp_params(w: MlpWeights, act: torch.Tensor) -> torch.Tensor:
    """The MLP block's params as CPU bytes, with `act` (bf16 [FFN]) as its
    workspace."""
    return _ext.load().qwen38_mlp_params(
        post_norm=w.post_norm, w13=int4.parts(w.w13), w2=int4.parts(w.down_proj), eps=w.eps,
        act=act)


class Qwen38Layer:
    """Runs the layer on the kernel, one token a launch."""

    def __init__(self, weights: LayerWeights, cos_sin: torch.Tensor,
                 timeout_ns: int = DEFAULT_TIMEOUT_NS, ctas: int | None = None,
                 splits: int | None = None) -> None:
        """`splits`: attention chunks per query head, by default as many as
        the CTAs allow."""
        self.weights = weights
        self.wqkv = fuse_qkv(weights)
        self.int4 = not isinstance(self.wqkv, torch.Tensor)
        self.cos_sin = cos_sin
        device = weights.input_norm.device
        self.ctas = ctas or num_ctas(device)
        self.splits = splits or self.ctas // Q_HEADS
        self._timeout_ns = timeout_ns
        self._qkv = torch.zeros(QKV_ROWS, dtype=torch.float32, device=device)
        items = Q_HEADS * self.splits
        self._partial_ml = torch.zeros(items, 2, dtype=torch.float32, device=device)
        self._partial_o = torch.zeros(items, HEAD_DIM, dtype=torch.float32, device=device)
        self._act = torch.zeros(FFN, dtype=torch.bfloat16, device=device)
        self._sync = sync_words(device)
        self._error = ErrorRecord()

    def run(self, residual: torch.Tensor, cache: PagedCache, pos: int,
            positions: torch.Tensor) -> LayerOutput:
        """The layer's outputs for a token at cache position `pos` with M-RoPE
        `positions` (int32 [3]: temporal, height, width). Writes the token's
        key and value into `cache` at `pos`; the output's key and value are
        that slot's."""
        device = residual.device
        hidden = torch.empty(DIM, dtype=torch.float32, device=device)
        out = torch.empty(DIM, dtype=torch.float32, device=device)
        params = self.params(residual.float().contiguous(), hidden, out, cache, pos,
                             positions.int().contiguous())
        self.launch(params)
        self._error.synchronize()
        key, value = cache.gather(pos + 1)
        return LayerOutput(hidden, out, key[pos], value[pos])

    def launch(self, params: torch.Tensor) -> None:
        if not self.int4:
            raise TypeError("the layer kernel reads its projections as int4")
        _ext.load().run_qwen38_layer(params=params, num_ctas=self.ctas)

    def params(self, residual_in: torch.Tensor, hidden: torch.Tensor, residual: torch.Tensor,
               cache: PagedCache, pos: int, positions: torch.Tensor) -> torch.Tensor:
        """The layer's launch params as CPU bytes, reading `residual_in` and
        writing `hidden`, `residual` and the cache at `pos`. With bf16
        weights they are the MTP head's layer (s2mk/qwen38_mtp.py)."""
        w, mlp = self.weights, self.weights.mlp
        return _ext.load().qwen38_layer_params(
            residual_in=residual_in, input_norm=w.input_norm, wqkv=int4.parts(self.wqkv),
            q_norm=w.q_norm, k_norm=w.k_norm, wo=int4.parts(w.o_proj), cos_sin=self.cos_sin,
            positions=positions,
            key_cache=cache.key, value_cache=cache.value, block_table=cache.block_table,
            pos=pos, splits=self.splits, post_norm=mlp.post_norm, w13=int4.parts(mlp.w13),
            w2=int4.parts(mlp.down_proj), eps=w.eps, timeout_ns=self._timeout_ns, qkv=self._qkv,
            partial_ml=self._partial_ml, partial_o=self._partial_o, act=self._act,
            hidden=hidden, residual=residual, sync=self._sync, error=self._error.tensor,
            num_ctas=self.ctas)
