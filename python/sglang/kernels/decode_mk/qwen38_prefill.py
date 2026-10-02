# Copyright 2026 Fractalyze Inc. All rights reserved.
"""The prompt pass of Qwen3.8-27B's language model on the kernel, a chunk of
up to MAX_TOKENS tokens a launch (s2mk/csrc/qwen38_prefill.cu).

The prefill runs on a Qwen38Decoder's per-layer params, so it advances the
decoder's own linear-attention states and fills its KV caches, and the decode
steps that follow continue from them.
"""

from __future__ import annotations

import torch

from sglang.kernels.decode_mk import _ext, gdn, int4
from sglang.kernels.decode_mk.barrier import ErrorRecord, sync_words
from sglang.kernels.decode_mk.gemv import num_ctas
from sglang.kernels.decode_mk.qwen38_decode import Qwen38Decoder, is_full
from sglang.kernels.decode_mk.qwen38_layer import DIM, FFN, HEAD_DIM, Q_DIM, Q_HEADS

MAX_TOKENS = 64
# The fewest CTAs a launch may use (kQwen38PrefillMinCtas).
MIN_CTAS = 2 * gdn.V_HEADS


# Each layer's phases, a grid barrier after each (s2mk/csrc/qwen38_prefill.cu).
_MLP_PHASES = ("mlp norm", "gate-up", "down")
LINEAR_PHASES = ("norm", "in_proj", "conv", "delta rule", "gated norm", "out_proj",
                 *_MLP_PHASES)
FULL_PHASES = ("norm", "qkv", "kv", "attention", "attention merge", "o_proj",
               *_MLP_PHASES)
# Cache positions an attention partial covers (kQwen38PrefillAttnSpan).
ATTN_SPAN = 512


def phases(num_layers: int) -> list[tuple[int, str]]:
    """(layer, phase) of each grid barrier of a launch over `num_layers`
    layers, in order."""
    return [(layer, phase) for layer in range(num_layers)
            for phase in (FULL_PHASES if is_full(layer) else LINEAR_PHASES)]


class Qwen38Prefiller:
    """Runs prompt chunks on the kernel, on `decoder`'s weights, states and
    caches, on `ctas` CTAs (default one per SM)."""

    def __init__(self, decoder: Qwen38Decoder, ctas: int | None = None) -> None:
        self.decoder = decoder
        device = decoder.residual.device
        self.num_ctas = ctas or num_ctas(device)

        def buf(*shape, dtype=torch.float32):
            return torch.zeros(*shape, dtype=dtype, device=device)

        self.residual = buf(MAX_TOKENS, DIM)
        self._h = buf(MAX_TOKENS, DIM, dtype=torch.bfloat16)
        self._proj = buf(MAX_TOKENS, gdn.IN_ROWS)
        self._beta = buf(MAX_TOKENS, gdn.V_HEADS)
        self._gate = buf(MAX_TOKENS, gdn.V_HEADS)
        self._query = buf(MAX_TOKENS, gdn.KEY_DIM)
        self._key = buf(MAX_TOKENS, gdn.KEY_DIM)
        self._value = buf(MAX_TOKENS, gdn.VALUE_DIM)
        self._core = buf(MAX_TOKENS, gdn.VALUE_DIM)
        self._attn = buf(MAX_TOKENS, Q_DIM, dtype=torch.bfloat16)
        self._act = buf(MAX_TOKENS, FFN, dtype=torch.bfloat16)
        # Positions the caches hold, and the attention spans that cover them.
        caches = decoder.state.caches
        self.max_positions = (caches[0].block_table.numel() * caches[0].key.shape[1]
                              if caches else MAX_TOKENS)
        spans = -(-self.max_positions // ATTN_SPAN)
        self._partial_ml = buf(MAX_TOKENS, Q_HEADS, spans, 2)
        self._partial_o = buf(MAX_TOKENS, Q_HEADS, spans, HEAD_DIM)
        self.logits = buf(decoder.weights.vocab)
        self._pos0 = torch.zeros(1, dtype=torch.int32, device=device)
        self._sync = sync_words(device)
        self._error = ErrorRecord()

    def launch(self, ids: torch.Tensor, pos0: torch.Tensor, positions: torch.Tensor,
               logits: bool, hidden: torch.Tensor | None = None,
               profile: torch.Tensor | None = None) -> None:
        """Queues one chunk without waiting: tokens `ids` (int64 [tokens], in
        the vocabulary, which the kernel does not check) at cache positions
        pos0 (int32 [1]) on, below max_positions, which it does not check
        either, turning at M-RoPE `positions` (int32 [3,
        tokens]: temporal, height, width). With `logits`, writes the last
        token's; with `hidden` (fp32 [layers + 1, tokens, DIM]), every
        layer's input and the last layer's output. `profile` (int64
        [num_ctas, barriers(layers), 2]) takes each CTA's globaltimer on
        arriving at and leaving every grid barrier."""
        tokens = ids.shape[0]
        if not 0 < tokens <= MAX_TOKENS:
            raise ValueError(f"a chunk takes 1 to {MAX_TOKENS} tokens, got {tokens}")
        d = self.decoder
        w = d.weights
        self.residual[:tokens].copy_(w.embed[ids])
        _ext.load().run_qwen38_prefill(
            linear=d._linear, linear_mlp=d._linear_mlp, full=d._full, tokens=tokens,
            pos0=pos0, positions=positions, final_norm=w.final_norm,
            lm_head=int4.parts(w.lm_head), eps=w.eps, timeout_ns=d._timeout_ns,
            residual=self.residual, h=self._h,
            proj=self._proj, beta=self._beta, gate=self._gate, query=self._query,
            key=self._key, value=self._value, core=self._core, attn=self._attn,
            act=self._act, partial_ml=self._partial_ml, partial_o=self._partial_o,
            logits=self.logits if logits else None, hidden=hidden,
            profile=profile, sync=self._sync, error=self._error.tensor, num_ctas=self.num_ctas)

    def run(self, ids: list[int], pos0: int = 0,
            hidden: torch.Tensor | None = None) -> torch.Tensor:
        """The last token's logits (fp32 [vocab]) after prompt `ids`, text
        tokens whose M-RoPE positions are all their cache positions, pos0 on,
        in chunks of MAX_TOKENS. With `hidden` (fp32 [layers + 1, len(ids),
        DIM]), every layer's input and the last layer's output per token.
        Waits for it."""
        vocab = self.decoder.weights.vocab
        if not ids:
            raise ValueError("the prompt is empty")
        if any(not 0 <= token < vocab for token in ids):
            raise ValueError(f"a token is outside the vocabulary of {vocab}")
        if pos0 + len(ids) > self.max_positions:
            raise ValueError(f"the prompt reaches position {pos0 + len(ids)}, past the "
                             f"caches' {self.max_positions}")
        device = self.residual.device
        for start in range(0, len(ids), MAX_TOKENS):
            chunk = ids[start:start + MAX_TOKENS]
            tokens = len(chunk)
            last = start + tokens == len(ids)
            pos = torch.arange(pos0 + start, pos0 + start + tokens, dtype=torch.int32,
                               device=device)
            self._pos0.fill_(pos0 + start)
            out = None
            if hidden is not None:
                out = torch.empty(hidden.shape[0], tokens, DIM, device=device)
            self.launch(torch.tensor(chunk, device=device), self._pos0,
                        pos.expand(3, tokens).contiguous(), last, out)
            if hidden is not None:
                hidden[:, start:start + tokens] = out
        self._error.synchronize()
        return self.logits
