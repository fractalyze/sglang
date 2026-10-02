# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""The DECODE_MK_MTP speculative algorithm, registered when
SGLANG_DECODE_MK_MTP=1: Qwen3.8-27B's MTP head drafts on decode-mk's
megakernel, one draft a step as EAGLE runs it at topk 1, and the verify
megakernel checks the drafts in one launch (decode_mk_mtp_worker.py).
"""

from __future__ import annotations

from typing import Any

from sglang.srt.arg_groups.decode_mk_hook import MTP_ALGORITHM
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import CustomSpecAlgo

# Drafts a cycle the verify step checks with the token before them
# (kQwen38MaxVerifyTokens - 1).
MAX_DRAFTS = 3


class DecodeMkMtpAlgo(CustomSpecAlgo):
    # The scheduler asks every algorithm these, and CustomSpecAlgo does not
    # answer them; the answers are SpeculativeAlgorithm's for an algorithm
    # without a draft model.
    def carries_draft_hidden_states(self) -> bool:
        return False

    def need_topk(self) -> bool:
        return False

    def create_future_map(
        self,
        device,
        req_to_token_pool,
        needs_cpu_seq_lens=True,
        needs_confidence_relay=False,
    ):
        from sglang.srt.managers.overlap_utils import FutureMap

        return FutureMap(
            device, self, req_to_token_pool, needs_cpu_seq_lens, needs_confidence_relay
        )

    def handle_server_args(self, server_args: Any) -> None:
        """K = --speculative-num-steps drafts a cycle (default 1, the fastest
        at batch 1), a chain: topk 1, K + 1 tokens verified."""
        steps = server_args.speculative_num_steps or 1
        if not 1 <= steps <= MAX_DRAFTS:
            raise ValueError(
                f"{MTP_ALGORITHM} drafts 1 to {MAX_DRAFTS} tokens a cycle, got "
                f"--speculative-num-steps {steps}"
            )
        server_args.speculative_num_steps = steps
        server_args.speculative_eagle_topk = 1
        server_args.speculative_num_draft_tokens = steps + 1


@SpeculativeAlgorithm.register(
    MTP_ALGORITHM, supports_overlap=False, spec_class=DecodeMkMtpAlgo
)
def _worker(server_args: Any) -> type:
    from sglang.srt.speculative.decode_mk_mtp_worker import DecodeMkMtpWorker

    return DecodeMkMtpWorker
