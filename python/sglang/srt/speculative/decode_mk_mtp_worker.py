# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""DECODE_MK_MTP's worker: greedy speculative decoding of one Qwen3.8-27B
request on decode-mk's megakernels (models/qwen3_5_decode_mk.py).

A prompt runs on the target model, whose runner also runs the MTP head over
it. Each decode step of a greedy request is then one speculative cycle: the
head drafts K tokens, one verify launch checks them with the token before
them, and the step emits the accepted drafts and the model's own next token,
1 to K + 1 tokens, all of them plain greedy decoding's. A request that
samples, constrains its grammar or asks for logprobs decodes one token a step
on the decode kernel instead.

The kernels hold the request's KV and states, so the scheduler's KV slots
for the step are allocated and released by its own bookkeeping but never
written, as the target model never reads them.
"""

from __future__ import annotations

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_device, get_spec
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.ngram_info import NgramVerifyInput


class DecodeMkMtpWorker(BaseSpecWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__()
        self._target_worker = target_worker
        self._draft_worker = None
        self.model_runner = target_worker.model_runner
        self.speculative_num_draft_tokens = get_spec().speculative_num_draft_tokens
        self.device = get_device().device

    @property
    def runner(self):
        return self.model_runner.model.runner

    def _speculates(self, batch: ScheduleBatch) -> bool:
        return (
            batch.sampling_info.is_all_greedy
            and not batch.has_grammar
            and not batch.return_logprob
        )

    def forward_batch_generation(
        self, batch: ScheduleBatch, on_publish=None, pp_proxy_tensors=None
    ) -> GenerationBatchResult:
        width = self.speculative_num_draft_tokens
        accept_tokens = torch.zeros(width, dtype=torch.int32, device=self.device)
        logits_output = None
        if batch.forward_mode.is_decode():
            # The step decodes the request's last token, at the first position
            # the scheduler has not committed.
            req = batch.reqs[0]
            token = torch.tensor([req.output_ids[-1]], device=self.device)
            pos = int(batch.seq_lens_cpu[0])
            if self._speculates(batch):
                emitted = self.runner.speculate(token, pos)
            else:
                logits = self.runner.decode(
                    token, torch.tensor([pos], device=self.device)
                )
                logits_output = LogitsProcessorOutput(
                    next_token_logits=logits.unsqueeze(0)
                )
                forward_batch = ForwardBatch.init_new(
                    batch, self.model_runner, return_hidden_states_before_norm=False
                )
                emitted = self.model_runner.sample(
                    logits_output, forward_batch
                ).tolist()
            accept_tokens[: len(emitted)] = torch.tensor(emitted, dtype=torch.int32)
            accept_lens = torch.tensor(
                [len(emitted)], dtype=torch.int32, device=self.device
            )
            next_token_ids = accept_tokens
            new_seq_lens = batch.seq_lens + accept_lens
        else:
            result = self.target_worker.forward_batch_generation(batch)
            logits_output = result.logits_output
            next_token_ids = result.next_token_ids
            accept_tokens[0] = next_token_ids[0]
            accept_lens = torch.ones(1, dtype=torch.int32, device=self.device)
            new_seq_lens = batch.seq_lens.clone()
        if on_publish is not None:
            on_publish(new_seq_lens)
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            accept_lens=accept_lens,
            new_seq_lens=new_seq_lens,
            next_draft_input=NgramVerifyInput(
                draft_token_num=width,
                new_seq_lens=new_seq_lens,
                accept_tokens=accept_tokens,
                accept_lens=accept_lens,
            ),
            speculative_num_draft_tokens=width,
        )
