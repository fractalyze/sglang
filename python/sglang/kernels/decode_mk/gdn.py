# Copyright 2026 Fractalyze Inc. All rights reserved.
"""A Qwen3.8 linear-attention (gated delta rule) layer for one token in one
launch (s2mk/csrc/gdn.cu), and the PyTorch model it is held to.

The projections are asymmetric W4A16 (s2mk/int4.py) except where Qwen3.8's
int4 checkpoint keeps them in bf16: the gates b and a always, and out_proj in
layer 0. The layer carries two states across tokens, both fp32 and both read
and written in place: the conv's last CONV_WIDTH - 1 inputs per channel, and
each value head's delta-rule state S, stored by column as [V_HEADS, dv, dk].
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sglang.kernels.decode_mk import _ext, int4
from sglang.kernels.decode_mk.barrier import DEFAULT_TIMEOUT_NS, ErrorRecord, sync_words
from sglang.kernels.decode_mk.gemv import num_ctas

DIM, K_HEADS, V_HEADS, HEAD_DIM, CONV_WIDTH = 5120, 16, 48, 128, 4
KEY_DIM, VALUE_DIM = K_HEADS * HEAD_DIM, V_HEADS * HEAD_DIM
# The conv's channels: q, then k, then v.
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
# The int4 in_proj's rows: q, k, v, then z; the bf16 gates' rows: b, then a,
# one per value head.
IN_ROWS = CONV_DIM + VALUE_DIM
GATE_ROWS = 2 * V_HEADS
# Fewest CTAs a launch may use (kGdnMinCtas in s2mk/csrc/gdn.h).
MIN_CTAS = 96


@dataclass(frozen=True)
class GdnWeights:
    norm: torch.Tensor  # bf16 [DIM]: input_layernorm, applied as 1 + norm
    in_proj: int4.Projection  # [IN_ROWS, DIM]
    gates: torch.Tensor  # bf16 [GATE_ROWS, DIM]
    conv: torch.Tensor  # bf16 [CONV_DIM, CONV_WIDTH], oldest tap first
    a_log: torch.Tensor  # fp32 [V_HEADS]
    dt_bias: torch.Tensor  # fp32 [V_HEADS]
    out_norm: torch.Tensor  # bf16 [HEAD_DIM]: the gated RMSNorm's weight
    out_proj: int4.Projection  # [DIM, VALUE_DIM]
    eps: float = 1e-6

    @classmethod
    def random(cls, seed: int) -> GdnWeights:
        """Weights at Qwen3-Next's initial ranges: A = exp(A_log) in [1, 16]
        and dt = softplus(dt_bias) log-uniform in [1e-3, 1e-1], so a head's
        decay per token spans near-total forgetting to near 1."""
        gen = torch.Generator(device="cuda").manual_seed(seed)

        def normal(*shape):
            return torch.randn(*shape, generator=gen, device="cuda")

        def uniform(*shape):
            return torch.rand(*shape, generator=gen, device="cuda")

        in_proj = int4.quantize_asymmetric(normal(IN_ROWS, DIM) / DIM**0.5)
        gates = (normal(GATE_ROWS, DIM) / DIM**0.5).bfloat16()
        out_proj = int4.quantize_asymmetric(normal(DIM, VALUE_DIM) / VALUE_DIM**0.5)
        dt = torch.exp(math.log(1e-3) + uniform(V_HEADS) * math.log(1e2))
        return cls(norm=(0.1 * normal(DIM)).bfloat16(), in_proj=in_proj, gates=gates,
                   conv=(normal(CONV_DIM, CONV_WIDTH) / 2).bfloat16(),
                   a_log=torch.log(1 + 15 * uniform(V_HEADS)),
                   dt_bias=dt + torch.log(-torch.expm1(-dt)),
                   out_norm=(1 + 0.1 * normal(HEAD_DIM)).bfloat16(), out_proj=out_proj)


@dataclass
class GdnState:
    conv: torch.Tensor  # fp32 [CONV_DIM, CONV_WIDTH - 1]: past inputs, oldest first
    recurrent: torch.Tensor  # fp32 [V_HEADS, dv, dk]: each head's S, by column

    @classmethod
    def random(cls, seed: int) -> GdnState:
        """A state as a prefill might leave it: each S column of norm about 1."""
        gen = torch.Generator(device="cuda").manual_seed(seed)
        conv = torch.randn(CONV_DIM, CONV_WIDTH - 1, generator=gen, device="cuda")
        recurrent = torch.randn(V_HEADS, HEAD_DIM, HEAD_DIM, generator=gen,
                                device="cuda") / HEAD_DIM**0.5
        return cls(conv, recurrent)

    def clone(self) -> GdnState:
        return GdnState(self.conv.clone(), self.recurrent.clone())


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def reference(w: GdnWeights, residual: torch.Tensor, state: GdnState,
              dtype: torch.dtype) -> tuple[torch.Tensor, GdnState]:
    """(residual_in + the layer's output in fp32, the next state) for one
    token, as Hugging Face's Qwen3-Next gated delta net decodes it in `dtype`
    (fp32 for ground truth, bf16 for the budget) on the dequantized weights:
    activations and the conv in `dtype`, the delta rule in fp32 as its
    recurrent kernels keep it. `state` is not modified."""
    x = residual.float()
    h = (x * torch.rsqrt(x.pow(2).mean() + w.eps) * (1 + w.norm.float())).to(dtype)
    qkv, z = F.linear(h, int4.dense(w.in_proj).to(dtype)).split([CONV_DIM, VALUE_DIM])
    b, a = F.linear(h, w.gates.to(dtype)).split(V_HEADS)

    taps = torch.cat([state.conv.to(dtype), qkv[:, None]], dim=1)
    mixed = F.silu((taps * w.conv.to(dtype)).sum(-1))
    q, k, v = mixed.float().split([KEY_DIM, KEY_DIM, VALUE_DIM])
    group = torch.arange(V_HEADS, device=x.device) // (V_HEADS // K_HEADS)
    q = _l2norm(q.view(K_HEADS, HEAD_DIM))[group] / HEAD_DIM**0.5
    k = _l2norm(k.view(K_HEADS, HEAD_DIM))[group]
    v = v.view(V_HEADS, HEAD_DIM)
    beta = torch.sigmoid(b.float())
    g = -w.a_log.exp() * F.softplus(a.float() + w.dt_bias)

    # S[h] as dk × dv: the stored [dv, dk] transposed.
    s = state.recurrent.transpose(1, 2) * g.exp()[:, None, None]
    kv_mem = (s * k[:, :, None]).sum(1)
    delta = (v - kv_mem) * beta[:, None]
    s = s + k[:, :, None] * delta[:, None, :]
    core = (s * q[:, :, None]).sum(1).to(dtype).float()

    normed = core * torch.rsqrt(core.pow(2).mean(-1, keepdim=True) + w.eps)
    gated = (w.out_norm.float() * normed.to(dtype).float()
             * F.silu(z.float().view(V_HEADS, HEAD_DIM))).to(dtype)
    out = F.linear(gated.flatten(), int4.dense(w.out_proj).to(dtype))
    next_state = GdnState(taps[:, 1:].float(), s.transpose(1, 2).contiguous())
    return x + out.float(), next_state


class Gdn:
    """Runs the layer on the kernel, one token a launch."""

    def __init__(self, weights: GdnWeights, timeout_ns: int = DEFAULT_TIMEOUT_NS,
                 ctas: int | None = None) -> None:
        self.weights = weights
        device = weights.norm.device
        self._ctas = ctas or num_ctas(device)
        self._timeout_ns = timeout_ns

        def workspace(n):
            return torch.zeros(n, dtype=torch.float32, device=device)

        self._mixed = workspace(CONV_DIM)
        self._z = workspace(VALUE_DIM)
        self._beta = workspace(V_HEADS)
        self._decay = workspace(V_HEADS)
        self._core = workspace(VALUE_DIM)
        self._sync = sync_words(device)
        self._error = ErrorRecord()

    def run(self, residual: torch.Tensor, state: GdnState) -> torch.Tensor:
        """residual + the layer's output, fp32 [DIM]; advances `state` in place."""
        out = torch.empty(DIM, dtype=torch.float32, device=residual.device)
        self.launch(residual.float().contiguous(), out, state)
        self._error.synchronize()
        return out

    def launch(self, residual_in: torch.Tensor, residual: torch.Tensor,
               state: GdnState) -> None:
        """Enqueues one token, reading fp32 `residual_in` and writing fp32
        `residual`, without waiting: a CUDA graph may capture it."""
        _ext.load().run_gdn(params=self.params(residual_in, residual, state),
                            num_ctas=self._ctas)

    def params(self, residual_in: torch.Tensor, residual: torch.Tensor,
               state: GdnState) -> torch.Tensor:
        """The layer's launch params as CPU bytes, reading `residual_in`,
        writing `residual` and advancing `state` in place."""
        w = self.weights
        return _ext.load().gdn_params(
            residual_in=residual_in, norm=w.norm, in_proj=int4.parts(w.in_proj), gates=w.gates,
            conv=w.conv, a_log=w.a_log, dt_bias=w.dt_bias, out_norm=w.out_norm,
            out_proj=int4.parts(w.out_proj), eps=w.eps, timeout_ns=self._timeout_ns,
            conv_state=state.conv, state=state.recurrent, mixed=self._mixed, z=self._z,
            beta=self._beta, decay=self._decay, core=self._core, residual=residual,
            sync=self._sync, error=self._error.tensor)
