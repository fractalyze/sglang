# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""The server a SGLANG_DECODE_MK_* switch requires.

The megakernels own the model's weights and one request's states, so the
server they run in serves one request at a time, from the first token of its
prompt, with nothing of SGLang's own forward to fall back to: the stock model
is not built, and a second copy of its weights would not fit beside the
kernels' on the one RTX 5090 they target. A server asked for anything else
refuses to start rather than serve it on another path.
"""

from __future__ import annotations

from typing import Any

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.environ import envs

MTP_ALGORITHM = "DECODE_MK_MTP"


def decode_mk_errors(cfg: Any) -> list[str]:
    """What `cfg` (a resolved view of the server args) sets that the switches
    cannot serve, one line each; empty when it serves them, or when every
    switch is off."""
    decode = envs.SGLANG_DECODE_MK_DECODE.get()
    prefill = envs.SGLANG_DECODE_MK_PREFILL.get()
    mtp = envs.SGLANG_DECODE_MK_MTP.get()
    errors = []
    if not decode:
        if prefill or mtp:
            errors.append(
                "SGLANG_DECODE_MK_PREFILL and SGLANG_DECODE_MK_MTP run on "
                "SGLANG_DECODE_MK_DECODE=1's kernels; set it too"
            )
        if cfg.speculative_algorithm == MTP_ALGORITHM:
            errors.append(
                f"--speculative-algorithm {MTP_ALGORITHM} needs SGLANG_DECODE_MK_MTP=1"
            )
        return errors
    if cfg.max_running_requests != 1:
        errors.append(
            "--max-running-requests 1: the kernels hold one request's states, "
            f"got {cfg.max_running_requests}"
        )
    if not cfg.disable_radix_cache:
        errors.append(
            "--disable-radix-cache: a cached prefix would skip the prompt "
            "tokens the kernels' states are built from"
        )
    if not cfg.disable_cuda_graph:
        errors.append("--disable-cuda-graph: the kernels are one launch a step already")
    if cfg.context_length is None:
        errors.append(
            "--context-length: the kernels' KV caches are allocated for it at "
            "load, and the model's own 262,144 positions do not fit"
        )
    if cfg.tp_size != 1 or cfg.pp_size != 1 or cfg.dp_size != 1:
        errors.append(
            "--tp-size, --pp-size and --dp-size 1: the kernels run on one GPU"
        )
    if cfg.disaggregation_mode != "null":
        errors.append(
            "no --disaggregation-mode: one process holds the request's states"
        )
    if mtp and cfg.speculative_algorithm != MTP_ALGORITHM:
        errors.append(
            f"--speculative-algorithm {MTP_ALGORITHM} with SGLANG_DECODE_MK_MTP=1, "
            f"got {cfg.speculative_algorithm}"
        )
    if mtp and not cfg.disable_overlap_schedule:
        errors.append(
            "--disable-overlap-schedule with SGLANG_DECODE_MK_MTP=1: a cycle "
            "reads its drafts' verdicts on the host before the next"
        )
    if not mtp and cfg.speculative_algorithm is not None:
        errors.append(
            f"no --speculative-algorithm without SGLANG_DECODE_MK_MTP=1, got "
            f"{cfg.speculative_algorithm}"
        )
    return errors


def check_decode_mk_server_args(server_args: Any) -> None:
    errors = decode_mk_errors(resolving_view(server_args))
    if errors:
        raise ValueError(
            "The SGLANG_DECODE_MK_* switches serve one Qwen3.8-27B request at a "
            "time on one GPU, and need:\n  " + "\n  ".join(errors)
        )
