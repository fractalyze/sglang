# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""Qwen3.8-27B on decode-mk's megakernels (sglang/kernels/decode_mk), behind
the SGLANG_DECODE_MK_* switches.

With SGLANG_DECODE_MK_DECODE=1, get_model_architecture builds this model for
Qwen3_5ForConditionalGeneration instead of the stock one. The kernels own its
weights, read from the checkpoint by decode-mk's loader, and one request's
states: each full-attention layer's KV cache and each linear-attention
layer's conv and delta-rule states. SGLang keeps the rest: the tokenizer, the
scheduler, sampling from the kernels' logits, streaming and the API.
arg_groups/decode_mk_hook.py refuses a server with more than one running
request, a radix cache, or anything else these states cannot serve.

A request's prompt runs from position 0 into zeroed states: with
SGLANG_DECODE_MK_PREFILL=1 on the prefill kernel, layer by layer over the
whole prompt in the launches decode-mk's own prefill makes of it, else one
decode step a token. Each decode step is one launch of the decode kernel. Under
SGLANG_DECODE_MK_MTP=1, speculative/decode_mk_mtp_worker.py drives the
verify step and the MTP head through the same runner.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.kernels.decode_mk import qwen38_checkpoint
from sglang.kernels.decode_mk.qwen38_decode import (
    Qwen38Decoder,
    Qwen38State,
    Qwen38Weights,
)
from sglang.kernels.decode_mk.qwen38_layer import DIM, cos_sin_table, rms_norm
from sglang.kernels.decode_mk.qwen38_mtp import MtpWeights
from sglang.kernels.decode_mk.qwen38_prefill import Qwen38Prefiller
from sglang.kernels.decode_mk.qwen38_spec import Qwen38Generator
from sglang.kernels.decode_mk.qwen38_verify import MAX_TOKENS as VERIFY_TOKENS
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.runtime_context import get_model, get_spec

logger = logging.getLogger(__name__)

STOCK_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"


def decode_mk_architectures(architectures: list[str]) -> list[str]:
    """The architectures to build under SGLANG_DECODE_MK_DECODE=1: this
    model's in place of Qwen3.8's. Raises for any other model, which the
    kernels do not run."""
    if STOCK_ARCHITECTURE not in architectures:
        raise ValueError(
            f"SGLANG_DECODE_MK_DECODE=1 serves Qwen3.8-27B ({STOCK_ARCHITECTURE}), "
            f"not {architectures}"
        )
    return [Qwen3_5DecodeMkForConditionalGeneration.__name__]


def checkpoint_dir(model_path: str, revision: Optional[str]) -> str:
    """The local directory of the checkpoint `--model-path` names, which
    decode-mk's loader reads: SGLang's own loader resolves it as for the stock
    model, though the weight iterator it hands load_weights is never read. A
    hub id's safetensors, their index and config.json are downloaded at
    `revision` into --download-dir, and --model-checksum is verified. The
    ignore patterns are LoadConfig's default, as in the model runner's."""
    loader = DefaultModelLoader(
        LoadConfig(
            load_format=LoadFormat.SAFETENSORS, download_dir=get_model().download_dir
        )
    )
    folder, _, _ = loader._prepare_weights(
        model_path,
        revision,
        fall_back_to_pt=False,
        allow_patterns_overrides=["*.safetensors", "*.json"],
    )
    return folder


class DecodeMkRunner:
    """One request at a time on the megakernels, from its prompt's first
    token, in states sized for `max_positions` positions.

    Given the MTP head's weights, greedy decoding can also run speculatively,
    `drafts` drafts a cycle: decode-mk's Qwen38Generator then holds the
    states, in the verify step's slots, and the CUDA graphs of its cycle. The decode and prefill kernels run on slot 0's
    states, where every request's prompt starts, and the MTP head runs over
    the prompt as it goes, ready to draft from its last token."""

    def __init__(
        self,
        weights: Qwen38Weights,
        max_positions: int,
        prefill: bool,
        mtp: Optional[MtpWeights] = None,
        drafts: int = 1,
    ) -> None:
        self.weights = weights
        if mtp is None:
            self.generator = None
            state = Qwen38State.zeros(len(weights.layers), max_positions)
        else:
            self.generator = Qwen38Generator(weights, max_positions, mtp, drafts)
            slotted = self.generator.state
            state = Qwen38State(
                [slotted.linear(j, 0) for j in range(len(slotted.conv))],
                slotted.caches,
            )
        self.state = state
        self.decoder = Qwen38Decoder(self.weights, state, cos_sin_table(max_positions))
        self.prefiller = Qwen38Prefiller(self.decoder) if prefill else None
        self.max_positions = max_positions
        device = self.decoder.logits.device
        self._pos = torch.zeros(1, dtype=torch.int32, device=device)
        self._positions = torch.zeros(3, dtype=torch.int32, device=device)
        # The head's input at the last prompt position extended, which waits
        # for the token after it; None once speculation has begun.
        self._pending: Optional[torch.Tensor] = None

    def _reset(self) -> None:
        for s in self.state.linear:
            s.conv.zero_()
            s.recurrent.zero_()
        self._pending = None

    def extend(self, ids: torch.Tensor, pos0: int) -> torch.Tensor:
        """The logits (fp32 [vocab]) after prompt tokens `ids` (int64 [n]) at
        positions pos0 on, which continue the request's last call; a request
        starts at pos0 0."""
        n = ids.numel()
        if pos0 + n > self.max_positions:
            raise ValueError(
                f"the prompt reaches position {pos0 + n}, past the kernels' "
                f"{self.max_positions}"
            )
        if pos0 == 0:
            self._reset()
        # Under MTP, the last layer's output per token, which the head reads.
        residual = None
        if self.generator is not None:
            residual = torch.empty(n, DIM, device=ids.device)
        if self.prefiller is not None:
            logits = self._prefill(ids, pos0, residual)
        else:
            logits = self._step_through(ids, pos0, residual)
        if residual is not None:
            self._head_over_prompt(ids, pos0, residual)
        return logits

    def _prefill(
        self, ids: torch.Tensor, pos0: int, residual: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """The prompt on the prefill kernel, layer by layer over as many
        tokens a launch as decode-mk's own Qwen38Prefiller.run takes."""
        for start, tokens in self.prefiller.launches(ids.numel()):
            end = start + tokens
            self._pos.fill_(pos0 + start)
            positions = torch.arange(
                pos0 + start, pos0 + end, dtype=torch.int32, device=ids.device
            ).expand(3, tokens)
            self.prefiller.launch(
                ids[start:end],
                self._pos,
                positions.contiguous(),
                logits=end == ids.numel(),
                last_hidden=None if residual is None else residual[start:end],
            )
        return self.prefiller.logits

    def _step_through(
        self, ids: torch.Tensor, pos0: int, residual: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """The prompt one decode step a token."""
        hidden = None
        if residual is not None:
            hidden = torch.empty(len(self.weights.layers) + 1, DIM, device=ids.device)
        for i in range(ids.numel()):
            self._pos.fill_(pos0 + i)
            self._positions.fill_(pos0 + i)
            self.decoder.launch(
                ids[i : i + 1].int(), self._pos, self._positions, hidden
            )
            if residual is not None:
                residual[i].copy_(hidden[-1])
        return self.decoder.logits

    def _head_over_prompt(
        self, ids: torch.Tensor, pos0: int, residual: torch.Tensor
    ) -> None:
        """The head's cache at prompt positions pos0 - 1 … pos0 + n - 2, each
        from the final-normed hidden state there and the token after it, from
        the last layer's output `residual` (fp32 [n, DIM]) at pos0 on; the last
        position waits for the next token.

        The kernels write the last layer's output, not the final-normed hidden
        state the head reads, so it is normed here; a draft from a state that
        differs in its last bits is only a weaker guess."""
        w = self.weights
        hidden = rms_norm(residual, w.final_norm, w.eps, torch.bfloat16)
        if self._pending is not None:
            rows = torch.cat([self._pending.unsqueeze(0), hidden[:-1]])
            following, first = ids, pos0 - 1
        else:
            rows, following, first = hidden[:-1], ids[1:], pos0
        if following.numel():
            self.generator.mtp.prompt(following.int(), rows, first)
        self._pending = hidden[-1].clone()

    def decode(self, token: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """The logits (fp32 [vocab]) after `token` (int [1]) at position
        `pos` (int [1]), both on the device."""
        self._pos.copy_(pos)
        self._positions.copy_(pos.expand(3))
        self.decoder.launch(token.int(), self._pos, self._positions)
        return self.decoder.logits

    def speculate(self, token: int, pos: int) -> list[int]:
        """The greedy tokens after `token` at position `pos` from one cycle of
        speculative decoding: the accepted drafts and the model's own choice
        after them. A request's first cycle starts decode-mk's loop from the
        prompt's last position; every later one continues where the cycle
        before left it."""
        if self._pending is not None:
            self.generator.start(self._pending, token, pos)
            self._pending = None
        return self.generator.cycle()


class Qwen3_5DecodeMkForConditionalGeneration(nn.Module):
    """Qwen3.8-27B's language model on the megakernels, at batch 1."""

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.runner: Optional[DecodeMkRunner] = None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        # The kernels read the checkpoint with decode-mk's own loader, in the
        # layout they run on; SGLang's weight iterator is left unread.
        model = get_model()
        path = checkpoint_dir(model.model_path, model.revision)
        positions = model.context_length
        speculative = envs.SGLANG_DECODE_MK_MTP.get()
        mtp, drafts = None, 1
        if speculative:
            drafts = get_spec().speculative_num_steps
            # A cycle near the end of the context writes its drafts past it.
            positions += VERIFY_TOKENS
        # SGLang loads under the model's dtype; decode-mk allocates its fp32
        # states and workspaces under torch's own default.
        with set_default_torch_dtype(torch.float32):
            kernel_weights = qwen38_checkpoint.load(path)
            if speculative:
                mtp = qwen38_checkpoint.load_mtp(path)
                # Speculative decoding as decode-mk runs it: the decode,
                # verify and MTP steps share an int4 LM head.
                kernel_weights = kernel_weights.with_int4_lm_head()
            self.runner = DecodeMkRunner(
                kernel_weights,
                positions,
                envs.SGLANG_DECODE_MK_PREFILL.get(),
                mtp,
                drafts,
            )
        logger.info(
            "decode-mk: Qwen3.8-27B on the megakernels, %d positions, prefill "
            "kernel %s, MTP %s",
            positions,
            "on" if self.runner.prefiller is not None else "off",
            (
                f"{drafts} drafts a cycle on an int4 LM head"
                if mtp is not None
                else "off"
            ),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> LogitsProcessorOutput:
        if forward_batch.batch_size != 1:
            raise RuntimeError(
                f"decode-mk runs one request at a time, got a batch of "
                f"{forward_batch.batch_size}"
            )
        if forward_batch.contains_mm_inputs():
            raise RuntimeError("decode-mk runs Qwen3.8's text model only")
        mode = forward_batch.forward_mode
        if mode.is_decode():
            logits = self.runner.decode(input_ids, forward_batch.positions)
        elif mode.is_extend():
            pos0 = forward_batch.extend_prefix_lens_cpu[0]
            logits = self.runner.extend(input_ids, pos0)
        else:
            raise RuntimeError(f"decode-mk does not run {mode} batches")
        return LogitsProcessorOutput(next_token_logits=logits.unsqueeze(0))


EntryClass = [Qwen3_5DecodeMkForConditionalGeneration]
