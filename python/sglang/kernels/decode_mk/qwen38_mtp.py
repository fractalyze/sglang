# Copyright 2026 Fractalyze Inc. All rights reserved.
"""Qwen3.8-27B's multi-token-prediction head for one step in one launch
(s2mk/csrc/qwen38_mtp.cu), and the PyTorch model it is held to.

The head drafts as SGLang's Qwen3_5ForCausalLMMTP does: at position pos it
takes the hidden state of the token at pos and the token at pos + 1, and
predicts the token at pos + 2. The hidden state is the model's final-normed
residual for the first draft and the head's own output after that. The head
is one full-attention layer with the dense MLP, all bf16 as the checkpoint
keeps it, with its own KV cache; it shares the model's embedding and LM head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sglang.kernels.decode_mk import _ext, int4
from sglang.kernels.decode_mk.barrier import DEFAULT_TIMEOUT_NS, ErrorRecord, sync_words
from sglang.kernels.decode_mk.gemv import num_ctas
from sglang.kernels.decode_mk.qwen38_decode import Qwen38Weights
from sglang.kernels.decode_mk.qwen38_layer import (DIM, FFN, HEAD_DIM, KV_DIM, Q_DIM, LayerWeights, MlpWeights,
                               Qwen38Layer, rms_norm)
from sglang.kernels.decode_mk.qwen38_layer import reference as layer_reference
from sglang.kernels.decode_mk.thinker_attention import PagedCache


@dataclass(frozen=True)
class MtpWeights:
    embed_norm: torch.Tensor  # bf16 [DIM]: pre_fc_norm_embedding, applied as 1 + w
    hidden_norm: torch.Tensor  # bf16 [DIM]: pre_fc_norm_hidden
    fc: torch.Tensor  # bf16 [DIM, 2 × DIM]: over [embedding, hidden]
    layer: LayerWeights  # bf16 projections
    final_norm: torch.Tensor  # bf16 [DIM]: mtp.norm
    eps: float = 1e-6

    @classmethod
    def random(cls, seed: int) -> MtpWeights:
        gen = torch.Generator(device="cuda").manual_seed(seed)

        def normal(*shape):
            return torch.randn(*shape, generator=gen, device="cuda")

        def proj(n, k):
            return (normal(n, k) / k**0.5).bfloat16()

        def norm(n=DIM):
            return (0.1 * normal(n)).bfloat16()

        mlp = MlpWeights.of(norm(), proj(FFN, DIM), proj(FFN, DIM), proj(DIM, FFN))
        layer = LayerWeights(input_norm=norm(), q_proj=proj(2 * Q_DIM, DIM),
                             k_proj=proj(KV_DIM, DIM), v_proj=proj(KV_DIM, DIM),
                             q_norm=norm(HEAD_DIM), k_norm=norm(HEAD_DIM),
                             o_proj=proj(DIM, Q_DIM), mlp=mlp)
        return cls(embed_norm=norm(), hidden_norm=norm(), fc=proj(DIM, 2 * DIM), layer=layer,
                   final_norm=norm())


def reference(w: MtpWeights, model: Qwen38Weights, token: int, hidden: torch.Tensor,
              cache: PagedCache, pos: int, cos_sin: torch.Tensor,
              dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """SGLang's Qwen3_5ForCausalLMMTP drafting from `hidden` (bf16 [DIM]) and
    `token` at position `pos`, in `dtype` (fp32 for ground truth, bf16 for
    the budget), the residual stream in fp32: (the head's normalized output,
    fp32 [DIM]; logits, fp32 [vocab]). Writes the layer's key and value, bf16
    as a cache holds them, at `pos`."""
    e = rms_norm(model.embed[token], w.embed_norm, w.eps, dtype)
    h = rms_norm(hidden, w.hidden_norm, w.eps, dtype)
    x = F.linear(torch.cat([e, h]), w.fc.to(dtype)).float()
    positions = torch.tensor([pos] * 3, device=x.device)
    out = layer_reference(w.layer, x, cache, pos, positions, cos_sin, dtype)
    block_size = cache.key.shape[1]
    block = int(cache.block_table[pos // block_size])
    cache.key[block, pos % block_size] = out.key.bfloat16()
    cache.value[block, pos % block_size] = out.value.bfloat16()
    n = rms_norm(out.residual, w.final_norm, w.eps, dtype)
    logits = torch.cat([F.linear(n, rows.to(dtype)) for rows in int4.dense_slices(model.lm_head)])
    return n.float(), logits.float()


class Qwen38Mtp:
    """Runs the head on the kernel, one launch a step, on `ctas` CTAs
    (default one per SM) with `splits` attention chunks per query head
    (default as many as the CTAs allow).

    A step reads `token`, `pos` and `hidden_in` and writes `hidden_out` and,
    unless asked not to, `logits`, all device tensors the head owns, so a
    CUDA graph can capture its launches."""

    def __init__(self, weights: MtpWeights, model: Qwen38Weights, cache: PagedCache,
                 cos_sin: torch.Tensor, timeout_ns: int = DEFAULT_TIMEOUT_NS,
                 ctas: int | None = None, splits: int | None = None) -> None:
        device = weights.fc.device
        self.weights = weights
        self.model = model
        self.cache = cache
        self.num_ctas = ctas or num_ctas(device)
        self._layer = Qwen38Layer(weights.layer, cos_sin, timeout_ns, self.num_ctas, splits)
        self.token = torch.zeros(1, dtype=torch.int32, device=device)
        self.pos = torch.zeros(1, dtype=torch.int32, device=device)
        self.hidden_in = torch.zeros(DIM, dtype=torch.bfloat16, device=device)
        self.hidden_out = torch.zeros(DIM, dtype=torch.bfloat16, device=device)
        self.logits = torch.zeros(model.vocab, dtype=torch.float32, device=device)
        self._residual = torch.zeros(DIM, dtype=torch.float32, device=device)
        self._sync = sync_words(device)
        self._error = ErrorRecord()
        # The layer's residual, hidden, pos and positions fields are unused.
        placeholder = torch.zeros(DIM, dtype=torch.float32, device=device)
        layer = self._layer.params(placeholder, placeholder, placeholder, cache, 0,
                                   torch.zeros(3, dtype=torch.int32, device=device))
        self._params = {with_logits: self._params_of(layer, timeout_ns, with_logits)
                        for with_logits in (True, False)}

    def _params_of(self, layer: torch.Tensor, timeout_ns: int,
                   with_logits: bool) -> torch.Tensor:
        w, m = self.weights, self.model
        return _ext.load().qwen38_mtp_params(
            token=self.token, pos=self.pos, hidden_in=self.hidden_in, embed=m.embed,
            embed_norm=w.embed_norm, hidden_norm=w.hidden_norm, fc=w.fc, layer=layer,
            final_norm=w.final_norm, lm_head=int4.parts(m.lm_head), eps=w.eps,
            timeout_ns=timeout_ns,
            residual=self._residual, hidden_out=self.hidden_out,
            logits=self.logits if with_logits else None, sync=self._sync,
            error=self._error.tensor)

    def launch(self, with_logits: bool = True) -> None:
        """Queues one step without waiting for it. Without logits the step
        stops after the final norm: it only writes the cache and
        `hidden_out`."""
        _ext.load().run_qwen38_mtp(params=self._params[with_logits], num_ctas=self.num_ctas)

    def step(self, token: int, hidden: torch.Tensor, pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(hidden_out, logits) drafting from `hidden` (bf16 [DIM]) and
        `token` at position `pos`. Waits for it."""
        if not 0 <= token < self.model.vocab:
            raise ValueError(f"token {token} is outside the vocabulary of {self.model.vocab}")
        self.token.fill_(token)
        self.pos.fill_(pos)
        self.hidden_in.copy_(hidden)
        self.launch()
        self._error.synchronize()
        return self.hidden_out, self.logits
