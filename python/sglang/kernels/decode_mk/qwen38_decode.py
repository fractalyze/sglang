# Copyright 2026 Fractalyze Inc. All rights reserved.
"""One decode step of Qwen3.8-27B's language model in one launch
(s2mk/csrc/qwen38_decode.cu), and the PyTorch model it is held to.

Layer i is full attention (s2mk/qwen38_layer.py) when i mod FULL_INTERVAL =
FULL_INTERVAL - 1, else linear attention (s2mk/gdn.py); every layer ends with
the dense MLP. The embedding is bf16; the LM head is bf16, or asymmetric int4
(Qwen38Weights.with_int4_lm_head).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from sglang.kernels.decode_mk import _ext, gdn, int4
from sglang.kernels.decode_mk.barrier import DEFAULT_TIMEOUT_NS, ErrorRecord, sync_words
from sglang.kernels.decode_mk.gdn import Gdn, GdnState, GdnWeights
from sglang.kernels.decode_mk.gemv import num_ctas
from sglang.kernels.decode_mk.qwen38_layer import (DIM, FFN, HEAD_DIM, KV_HEADS, Q_HEADS, LayerWeights, MlpWeights,
                               Qwen38Layer, mlp_params, mlp_reference, rms_norm)
from sglang.kernels.decode_mk.qwen38_layer import reference as full_reference
from sglang.kernels.decode_mk.thinker_attention import PagedCache

LAYERS, VOCAB, FULL_INTERVAL = 64, 248320, 4
# Random models draw their layers from this many distinct weight sets of each
# kind, so 64 layers fit in GPU memory beside other work. Every set outgrows
# the L2 several times, so each layer still streams its weights from DRAM.
LINEAR_POOL, FULL_POOL = 5, 3


def is_full(layer: int) -> bool:
    return layer % FULL_INTERVAL == FULL_INTERVAL - 1


@dataclass(frozen=True)
class LinearLayerWeights:
    attention: GdnWeights
    mlp: MlpWeights


@dataclass(frozen=True)
class Qwen38Weights:
    embed: torch.Tensor  # bf16 [vocab, DIM]
    layers: list[LinearLayerWeights | LayerWeights]
    final_norm: torch.Tensor  # bf16 [DIM], applied as 1 + w
    lm_head: int4.Projection  # [vocab, DIM]
    eps: float = 1e-6

    @classmethod
    def random(cls, num_layers: int = LAYERS, vocab: int = VOCAB, seed: int = 0) -> Qwen38Weights:
        """Linear layer j takes pool set j mod LINEAR_POOL and full layer k set
        k mod FULL_POOL, so a layer that took a neighbour's params would read
        other weights."""
        def layer_seed(i):
            return 1000 * seed + i

        linear = []
        for i in range(LINEAR_POOL):
            gen = torch.Generator(device="cuda").manual_seed(layer_seed(i))
            linear.append(LinearLayerWeights(GdnWeights.random(layer_seed(i)),
                                             MlpWeights.random(gen)))
        full = [LayerWeights.random(layer_seed(LINEAR_POOL + i)) for i in range(FULL_POOL)]
        layers = []
        for i in range(num_layers):
            if is_full(i):
                layers.append(full[(i // FULL_INTERVAL) % FULL_POOL])
            else:
                layers.append(linear[(i - i // FULL_INTERVAL) % LINEAR_POOL])
        gen = torch.Generator(device="cuda").manual_seed(layer_seed(999))

        def normal(*shape):
            return torch.randn(*shape, generator=gen, device="cuda")

        # Drawn in bf16, so fp32 never holds a whole table.
        embed = torch.randn(vocab, DIM, generator=gen, device="cuda", dtype=torch.bfloat16)
        lm_head = torch.randn(vocab, DIM, generator=gen, device="cuda", dtype=torch.bfloat16)
        return cls(embed=embed, layers=layers, final_norm=(0.1 * normal(DIM)).bfloat16(),
                   lm_head=lm_head.mul_(DIM**-0.5))

    @property
    def vocab(self) -> int:
        return int4.rows(self.lm_head)

    def with_int4_lm_head(self) -> Qwen38Weights:
        """These weights with the LM head quantized to asymmetric int4 in
        groups of 32 (int4.quantize_asymmetric), as the kernels read it by
        its format."""
        if not isinstance(self.lm_head, torch.Tensor):
            return self
        return replace(self, lm_head=int4.quantize_asymmetric(self.lm_head))


@dataclass
class Qwen38State:
    """What a sequence carries across tokens: each linear layer's states and
    each full layer's paged KV cache, in layer order."""
    linear: list[GdnState]
    caches: list[PagedCache]

    @classmethod
    def random(cls, num_layers: int, positions: int, seed: int,
               block_size: int = 16) -> Qwen38State:
        """States as a prefill might leave them, and caches of `positions`
        random keys and values each, on shuffled block tables."""
        full = num_layers // FULL_INTERVAL
        linear = [GdnState.random(1000 * seed + j) for j in range(num_layers - full)]
        gen = torch.Generator(device="cuda").manual_seed(1000 * seed + 999)
        blocks = -(-positions // block_size)
        caches = []
        for _ in range(full):
            kv = torch.randn(blocks, block_size, KV_HEADS, 2 * HEAD_DIM, generator=gen,
                             device="cuda").bfloat16()
            key, value = kv.split(HEAD_DIM, dim=-1)
            table = torch.randperm(blocks, generator=gen, device="cuda").int()
            caches.append(PagedCache(key, value, table))
        return cls(linear, caches)

    @classmethod
    def zeros(cls, num_layers: int, positions: int, block_size: int = 16) -> Qwen38State:
        """A new sequence's: zero states, and caches of `positions` slots."""
        full = num_layers // FULL_INTERVAL
        linear = [GdnState(torch.zeros(gdn.CONV_DIM, gdn.CONV_WIDTH - 1, device="cuda"),
                           torch.zeros(gdn.V_HEADS, gdn.HEAD_DIM, gdn.HEAD_DIM, device="cuda"))
                  for _ in range(num_layers - full)]
        blocks = -(-positions // block_size)
        caches = []
        for _ in range(full):
            kv = torch.zeros(blocks, block_size, KV_HEADS, 2 * HEAD_DIM, device="cuda",
                             dtype=torch.bfloat16)
            key, value = kv.split(HEAD_DIM, dim=-1)
            table = torch.arange(blocks, dtype=torch.int32, device="cuda")
            caches.append(PagedCache(key, value, table))
        return cls(linear, caches)

    def clone(self) -> Qwen38State:
        return Qwen38State(
            [s.clone() for s in self.linear],
            [PagedCache(c.key.clone(), c.value.clone(), c.block_table) for c in self.caches])


def require_int4(weights: Qwen38Weights) -> None:
    """Raises TypeError unless every projection the decode and verify steps
    read as int4 is int4: the full layers' attention and every layer's MLP.
    (The linear layers' input projection has no bf16 form.)"""
    for i, layer in enumerate(weights.layers):
        mlp = layer.mlp
        projections = [mlp.w13, mlp.down_proj]
        if is_full(i):
            projections += [layer.q_proj, layer.k_proj, layer.v_proj, layer.o_proj]
        if any(isinstance(w, torch.Tensor) for w in projections):
            raise TypeError(f"layer {i}: the decode step reads its projections as int4")


class Qwen38Decoder:
    """Runs decode steps on the kernel, one launch a step, advancing `state`
    in place, on `ctas` CTAs (default one per SM) with `splits` attention
    chunks per query head (default as many as the CTAs allow)."""

    def __init__(self, weights: Qwen38Weights, state: Qwen38State, cos_sin: torch.Tensor,
                 timeout_ns: int = DEFAULT_TIMEOUT_NS, ctas: int | None = None,
                 splits: int | None = None) -> None:
        require_int4(weights)
        device = weights.final_norm.device
        self.weights = weights
        self.state = state
        self.num_ctas = ctas or num_ctas(device)
        self.splits = splits or self.num_ctas // Q_HEADS
        self._timeout_ns = timeout_ns
        self.residual = torch.zeros(DIM, dtype=torch.float32, device=device)
        self.logits = torch.zeros(weights.vocab, dtype=torch.float32, device=device)
        # step()'s own copy of what launch() reads per step.
        self._token = torch.zeros(1, dtype=torch.int32, device=device)
        self._pos = torch.zeros(1, dtype=torch.int32, device=device)
        self._positions = torch.zeros(3, dtype=torch.int32, device=device)
        # The kernel points every block at `residual` and the step's position,
        # so the blocks' own are placeholders. The blocks own the workspace
        # their params name; the MLPs of the linear layers share `_act`.
        self._act = torch.zeros(FFN, dtype=torch.bfloat16, device=device)
        self._blocks = []
        linear, linear_mlp, full = [], [], []
        for i, layer in enumerate(weights.layers):
            if is_full(i):
                block = Qwen38Layer(layer, cos_sin, timeout_ns, self.num_ctas, self.splits)
                full.append(block.params(self.residual, self.residual, self.residual,
                                         state.caches[i // FULL_INTERVAL], 0, self._positions))
            else:
                block = Gdn(layer.attention, timeout_ns, self.num_ctas)
                linear.append(block.params(self.residual, self.residual,
                                           state.linear[i - i // FULL_INTERVAL]))
                linear_mlp.append(mlp_params(layer.mlp, self._act))
            self._blocks.append(block)
        self._linear = torch.stack(linear).to(device)
        self._linear_mlp = torch.stack(linear_mlp).to(device)
        self._full = torch.stack(full).to(device)
        self._sync = sync_words(device)
        self._error = ErrorRecord()

    def launch(self, token: torch.Tensor, pos: torch.Tensor, positions: torch.Tensor,
               hidden: torch.Tensor | None = None) -> None:
        """Queues one step without waiting for it, so a CUDA graph can capture
        it: `token` (int32 [1]) at cache position `pos` (int32 [1]) with M-RoPE
        `positions` (int32 [3]: temporal, height, width); the token must lie
        in the vocabulary, which the kernel does not check. Writes the logits;
        with `hidden` (fp32 [layers + 1, DIM]), every layer's input and the
        last layer's output."""
        w = self.weights
        _ext.load().run_qwen38_decode(
            linear=self._linear, linear_mlp=self._linear_mlp, full=self._full, token=token,
            pos=pos, positions=positions, embed=w.embed, final_norm=w.final_norm,
            lm_head=int4.parts(w.lm_head), eps=w.eps, timeout_ns=self._timeout_ns,
            residual=self.residual,
            logits=self.logits, hidden=hidden, sync=self._sync, error=self._error.tensor,
            num_ctas=self.num_ctas)

    def step(self, token: int, pos: int, positions: tuple[int, int, int] | None = None,
             hidden: torch.Tensor | None = None) -> torch.Tensor:
        """Logits (fp32 [vocab]) for `token` at cache position `pos`, by
        default a text token, whose M-RoPE positions are all `pos`. Waits for
        it."""
        if not 0 <= token < self.weights.vocab:
            raise ValueError(f"token {token} is outside the vocabulary of {self.weights.vocab}")
        self._token.fill_(token)
        self._pos.fill_(pos)
        self._positions.copy_(torch.tensor(positions or (pos, pos, pos), dtype=torch.int32))
        self.launch(self._token, self._pos, self._positions, hidden)
        self._error.synchronize()
        return self.logits


def reference_step(weights: Qwen38Weights, state: Qwen38State, token: int, pos: int,
                   positions: torch.Tensor, cos_sin: torch.Tensor,
                   dtype: torch.dtype) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """HF's Qwen3.5 text model decoding `token` at cache position `pos` in
    `dtype` on the dequantized weights, the residual stream in fp32:
    (logits, fp32 [vocab]; the residual entering each layer and the last
    layer's output). Advances `state` in place, writing each full layer's key
    and value, bf16 as a cache holds them, at `pos`."""
    x = weights.embed[token].float()
    hidden = [x]
    for i, layer in enumerate(weights.layers):
        if is_full(i):
            cache = state.caches[i // FULL_INTERVAL]
            out = full_reference(layer, x, cache, pos, positions, cos_sin, dtype)
            block_size = cache.key.shape[1]
            block = int(cache.block_table[pos // block_size])
            cache.key[block, pos % block_size] = out.key.bfloat16()
            cache.value[block, pos % block_size] = out.value.bfloat16()
            x = out.residual
        else:
            j = i - i // FULL_INTERVAL
            h, state.linear[j] = gdn.reference(layer.attention, x, state.linear[j], dtype)
            x = mlp_reference(layer.mlp, h, dtype)
        hidden.append(x)
    n = rms_norm(x, weights.final_norm, weights.eps, dtype)
    logits = torch.cat([F.linear(n, rows.to(dtype)) for rows in int4.dense_slices(weights.lm_head)])
    return logits.float(), hidden
