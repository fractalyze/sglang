import sys
from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.moe.token_dispatcher import DispatchOutputFormat
from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
    FlashinferCombineInput,
    FlashinferDispatcher,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import FlashinferA2ADispatchType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_MODULE = "sglang.srt.layers.moe.token_dispatcher.flashinfer"


def test_empty_mxfp8_dispatch_uses_same_payload_dtype_as_nonempty_rank():
    class FakeMoeAlltoAll:
        def dispatch(self, _topk_ids, payloads, *_args, **_kwargs):
            self.payload_dtypes = [payload.dtype for payload in payloads]
            return payloads

    dispatcher = object.__new__(FlashinferDispatcher)
    dispatcher.dispatch_type = FlashinferA2ADispatchType.MXFP8
    dispatcher.hidden_size = 128
    dispatcher.max_num_tokens = 0
    dispatcher.ep_size = 1
    dispatcher.invalid_token_expert_id = 8
    dispatcher.payload_in_workspace = False
    dispatcher.quant_config = {"use_mxfp8": True}
    dispatcher.moe_a2a = FakeMoeAlltoAll()

    hidden_states = torch.empty((0, 128), dtype=torch.bfloat16)
    topk_output = StandardTopKOutput(
        topk_weights=torch.empty((0, 1), dtype=torch.float32),
        topk_ids=torch.empty((0, 1), dtype=torch.int32),
        router_logits=None,
    )

    with (
        patch(
            "sglang.srt.layers.moe.token_dispatcher.flashinfer.get_dp_global_num_tokens",
            return_value=None,
        ),
        patch(
            "sglang.srt.layers.moe.token_dispatcher.flashinfer.is_dp_attention_enabled",
            return_value=False,
        ),
    ):
        output = dispatcher.dispatch(hidden_states, topk_output)

    assert dispatcher.moe_a2a.payload_dtypes == [
        torch.float8_e4m3fn,
        torch.uint8,
        torch.int32,
        torch.float32,
    ]
    assert output.hidden_states.dtype == torch.float8_e4m3fn
    assert output.hidden_states_scale.dtype == torch.uint8


class _RecordingMoeAlltoAll:
    def __init__(self):
        self.calls = []

    def dispatch(self, _topk_ids, payloads, runtime_max_tokens_per_rank, **_kwargs):
        self.calls.append(("dispatch", runtime_max_tokens_per_rank))
        return payloads

    def combine(self, payload, runtime_max_tokens_per_rank, **_kwargs):
        self.calls.append(("combine", runtime_max_tokens_per_rank))
        return payload.view(-1, payload.shape[-1])


def _make_dispatcher(*, prefill_max_num_tokens):
    dispatcher = object.__new__(FlashinferDispatcher)
    dispatcher.dispatch_type = FlashinferA2ADispatchType.BF16
    dispatcher.hidden_size = 16
    dispatcher.ep_size = 2
    dispatcher.invalid_token_expert_id = 8
    dispatcher.payload_in_workspace = False
    dispatcher.quant_config = {}
    dispatcher.moe_a2a = _RecordingMoeAlltoAll()
    dispatcher.max_num_tokens = 64
    dispatcher.prefill_moe_a2a = (
        _RecordingMoeAlltoAll() if prefill_max_num_tokens else None
    )
    dispatcher.prefill_max_num_tokens = prefill_max_num_tokens
    dispatcher._dispatch_prefill_allgather = lambda *_args: "allgather"
    return dispatcher


def _dispatch(dispatcher, *, num_tokens, dp_global, extend, capture=False):
    hidden_states = torch.zeros((num_tokens, 16), dtype=torch.bfloat16)
    topk_output = StandardTopKOutput(
        topk_weights=torch.zeros((num_tokens, 1), dtype=torch.float32),
        topk_ids=torch.zeros((num_tokens, 1), dtype=torch.int32),
        router_logits=None,
    )
    with ExitStack() as stack:
        for name, value in (
            ("get_is_extend_in_batch", extend),
            ("get_is_capture_mode", capture),
            ("get_dp_global_num_tokens", dp_global),
            ("is_dp_attention_enabled", True),
        ):
            stack.enter_context(patch(f"{_MODULE}.{name}", return_value=value))
        stack.enter_context(
            patch(
                f"{_MODULE}.get_parallel",
                return_value=type("P", (), {"attn_tp_size": 1}),
            )
        )
        return dispatcher.dispatch(hidden_states, topk_output)


def test_prefill_a2a_routes_dispatch_and_combine_through_its_own_workspace():
    """Extend must never touch the decode A2A, whose signal state its graphs own."""
    dispatcher = _make_dispatcher(prefill_max_num_tokens=32)

    output = _dispatch(dispatcher, num_tokens=20, dp_global=[20, 7], extend=True)
    dispatcher.combine(
        FlashinferCombineInput(
            hidden_states=torch.zeros((2 * 20, 16), dtype=torch.bfloat16)
        )
    )

    assert output.format == DispatchOutputFormat.FLASHINFER
    assert dispatcher.prefill_moe_a2a.calls == [("dispatch", 20), ("combine", 20)]
    assert dispatcher.moe_a2a.calls == []


@pytest.mark.parametrize(
    "prefill_max_num_tokens, dp_global, capture",
    [
        (0, [20, 7], False),  # prefill A2A not enabled
        (16, [20, 7], False),  # the largest DP rank overflows the prefill workspace
        (32, [20, 7], True),  # captured geometry would be replayed for other sizes
    ],
)
def test_extend_falls_back_to_allgather(prefill_max_num_tokens, dp_global, capture):
    dispatcher = _make_dispatcher(prefill_max_num_tokens=prefill_max_num_tokens)

    output = _dispatch(
        dispatcher, num_tokens=7, dp_global=dp_global, extend=True, capture=capture
    )

    assert output == "allgather"
    assert dispatcher.moe_a2a.calls == []


def test_decode_keeps_the_decode_workspace_with_prefill_a2a_enabled():
    dispatcher = _make_dispatcher(prefill_max_num_tokens=32)

    _dispatch(dispatcher, num_tokens=4, dp_global=[4, 3], extend=False)

    assert dispatcher.moe_a2a.calls == [("dispatch", 4)]
    assert dispatcher.prefill_moe_a2a.calls == []


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
