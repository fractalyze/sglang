# Copyright 2026 Fractalyze Inc. All rights reserved.
"""Up to MAX_TOKENS consecutive tokens of Qwen3.8-27B through the decode
megakernel's layers in one launch (s2mk/csrc/qwen38_verify.cu), as
speculative decoding verifies a token and its drafts.

Each token's logits and states are bitwise those of decoding the tokens one
step at a time (s2mk/qwen38_decode.py) at the same CTA count and attention
chunks. Each linear layer keeps its states in slots: a step reads one slot and
leaves token t's states in the next slot but t, so accepting a prefix of the
tokens only moves the slot it continues from.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.kernels.decode_mk import _ext, gdn, int4
from sglang.kernels.decode_mk.barrier import DEFAULT_TIMEOUT_NS, ErrorRecord, sync_words
from sglang.kernels.decode_mk.gdn import Gdn, GdnState
from sglang.kernels.decode_mk.gemv import num_ctas
from sglang.kernels.decode_mk.qwen38_decode import (FULL_INTERVAL, Qwen38State, Qwen38Weights, is_full,
                                require_int4)
from sglang.kernels.decode_mk.qwen38_layer import (DIM, FFN, HEAD_DIM, Q_HEADS, QKV_ROWS, Qwen38Layer, mlp_params)
from sglang.kernels.decode_mk.thinker_attention import PagedCache

MAX_TOKENS = 4  # kQwen38MaxVerifyTokens


@dataclass
class SlottedState:
    """A sequence's states as the verify step keeps them: each linear layer's
    conv and delta-rule states in `slots` copies, and each full layer's paged
    KV cache."""
    conv: list[torch.Tensor]  # fp32 [slots, CONV_DIM, CONV_WIDTH - 1] per linear layer
    recurrent: list[torch.Tensor]  # fp32 [slots, V_HEADS, dv, dk] per linear layer
    caches: list[PagedCache]

    @classmethod
    def of(cls, state: Qwen38State, slots: int) -> SlottedState:
        """`state`'s copy in slot 0, the other slots zero; the caches are
        `state`'s own."""
        def slotted(x):
            out = torch.zeros(slots, *x.shape, dtype=x.dtype, device=x.device)
            out[0] = x
            return out

        return cls([slotted(s.conv) for s in state.linear],
                   [slotted(s.recurrent) for s in state.linear], state.caches)

    @property
    def slots(self) -> int:
        return self.conv[0].shape[0]

    def linear(self, j: int, slot: int) -> GdnState:
        """Linear layer j's states in `slot`, as views."""
        return GdnState(self.conv[j][slot], self.recurrent[j][slot])


class Qwen38Verifier:
    """Runs verify steps on the kernel, one launch a step, on `ctas` CTAs
    (default one per SM) with `splits` attention chunks per query head
    (default as many as the CTAs allow), as Qwen38Decoder does."""

    def __init__(self, weights: Qwen38Weights, state: SlottedState, cos_sin: torch.Tensor,
                 timeout_ns: int = DEFAULT_TIMEOUT_NS, ctas: int | None = None,
                 splits: int | None = None) -> None:
        require_int4(weights)
        if state.slots <= MAX_TOKENS:
            raise ValueError(f"a verify step needs more than {MAX_TOKENS} slots, "
                             f"got {state.slots}")
        device = weights.final_norm.device
        self.weights = weights
        self.state = state
        self.num_ctas = ctas or num_ctas(device)
        self.splits = splits or self.num_ctas // Q_HEADS
        self._timeout_ns = timeout_ns
        n, items = MAX_TOKENS, Q_HEADS * self.splits

        def workspace(*shape, dtype=torch.float32):
            return torch.zeros(*shape, dtype=dtype, device=device)

        self.residual = workspace(n, DIM)
        self.logits = workspace(n, weights.vocab)
        self.final_hidden = workspace(n, DIM, dtype=torch.bfloat16)
        self._workspace = dict(
            mixed=workspace(n, gdn.CONV_DIM), z=workspace(n, gdn.VALUE_DIM),
            beta=workspace(n, gdn.V_HEADS), decay=workspace(n, gdn.V_HEADS),
            core=workspace(n, gdn.VALUE_DIM), qkv=workspace(n, QKV_ROWS),
            partial_ml=workspace(n, items, 2), partial_o=workspace(n, items, HEAD_DIM),
            act=workspace(n, FFN, dtype=torch.bfloat16))
        # step()'s own copy of what launch() reads.
        self._tokens = torch.zeros(n, dtype=torch.int32, device=device)
        self._pos = torch.zeros(1, dtype=torch.int32, device=device)
        self._slot = torch.zeros(1, dtype=torch.int32, device=device)

        # The kernel reads neither the blocks' residuals, positions nor
        # workspace, so those params point at placeholders. The blocks own
        # tensors their params name, the full layers' fused QKV weights among
        # them, so they live as long as the verifier.
        placeholder = workspace(DIM)
        positions = torch.zeros(3, dtype=torch.int32, device=device)
        self._act = workspace(FFN, dtype=torch.bfloat16)
        self._blocks = []
        linear, linear_mlp, full = [], [], []
        for i, layer in enumerate(weights.layers):
            if is_full(i):
                block = Qwen38Layer(layer, cos_sin, timeout_ns, self.num_ctas, self.splits)
                full.append(block.params(placeholder, placeholder, placeholder,
                                         state.caches[i // FULL_INTERVAL], 0, positions))
            else:
                j = i - i // FULL_INTERVAL
                block = Gdn(layer.attention, timeout_ns, self.num_ctas)
                linear.append(block.params(placeholder, placeholder, state.linear(j, 0)))
                linear_mlp.append(mlp_params(layer.mlp, self._act))
            self._blocks.append(block)
        self._linear = torch.stack(linear).to(device)
        self._linear_mlp = torch.stack(linear_mlp).to(device)
        self._full = torch.stack(full).to(device)
        self._sync = sync_words(device)
        self._error = ErrorRecord()

    def launch(self, tokens: torch.Tensor, pos: torch.Tensor, slot: torch.Tensor) -> None:
        """Queues one step without waiting for it, so a CUDA graph can capture
        it: `tokens` (int32 [n], n in [1, MAX_TOKENS]) at cache positions
        *pos, *pos + 1, …, reading the linear layers' states from slot *slot
        (int32 [1] each). Writes the first n rows of `logits`, `residual` and
        `final_hidden`. The tokens must lie in the vocabulary, which the
        kernel does not check."""
        n = tokens.numel()
        w = self.weights
        _ext.load().run_qwen38_verify(
            linear=self._linear, linear_mlp=self._linear_mlp, full=self._full, tokens=tokens,
            pos=pos, slot=slot, num_slots=self.state.slots, embed=w.embed,
            final_norm=w.final_norm, lm_head=int4.parts(w.lm_head), eps=w.eps, splits=self.splits,
            timeout_ns=self._timeout_ns, residual=self.residual[:n], logits=self.logits[:n],
            final_hidden=self.final_hidden[:n], sync=self._sync, error=self._error.tensor,
            num_ctas=self.num_ctas, **{k: v[:n] for k, v in self._workspace.items()})

    def step(self, tokens: list[int], pos: int, slot: int) -> torch.Tensor:
        """Logits (fp32 [len(tokens), vocab]) for `tokens` from cache position
        `pos`, continuing from the linear layers' slot `slot`. Waits for it."""
        if not 0 < len(tokens) <= MAX_TOKENS:
            raise ValueError(f"a verify step takes 1 to {MAX_TOKENS} tokens, got {len(tokens)}")
        if not all(0 <= t < self.weights.vocab for t in tokens):
            raise ValueError(f"tokens {tokens} leave the vocabulary of {self.weights.vocab}")
        n = len(tokens)
        self._tokens[:n].copy_(torch.tensor(tokens, dtype=torch.int32))
        self._pos.fill_(pos)
        self._slot.fill_(slot)
        self.launch(self._tokens[:n], self._pos, self._slot)
        self._error.synchronize()
        return self.logits[:n]
