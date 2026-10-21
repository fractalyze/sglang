"""The ("deepep", "marlin") MoE paths against Marlin applied expert by expert, on
DeepEP layouts built by hand; random 4-bit weights, since only row movement is tested.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

import sglang.srt.layers.moe.moe_runner.marlin as marlin_runner
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
from sglang.srt.layers.moe.moe_runner.marlin import (
    MarlinMoeQuantInfo,
    fused_experts_deepep_to_marlin,
    marlin_moe_deepep_ll,
    marlin_moe_deepep_normal,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPLLDispatchOutput,
    DeepEPNormalCombineInput,
    DeepEPNormalDispatchOutput,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

E, K, N, GROUP = 4, 256, 256, 128
CONFIG = SimpleNamespace(
    is_gated=True,
    activation="silu",
    gemm1_alpha=None,
    gemm1_clamp_limit=None,
    swiglu_limit=None,
)


def _quant_info(seed: int) -> MarlinMoeQuantInfo:
    g = torch.Generator(device="cuda").manual_seed(seed)

    def qweight(*shape):
        return torch.randint(
            -(2**31), 2**31 - 1, shape, dtype=torch.int32, device="cuda", generator=g
        )

    def scales(*shape):
        return (torch.rand(shape, device="cuda", generator=g) * 0.02 + 0.005).to(
            torch.bfloat16
        )

    return MarlinMoeQuantInfo(
        w13_qweight=qweight(E, K // 16, 2 * N * 2),
        w2_qweight=qweight(E, N // 16, K * 2),
        w13_scales=scales(E, K // GROUP, 2 * N),
        w2_scales=scales(E, N // GROUP, K),
        w13_g_idx_sort_indices=None,
        w2_g_idx_sort_indices=None,
        weight_bits=4,
    )


def _one_expert(x: torch.Tensor, e: int, qi: MarlinMoeQuantInfo) -> torch.Tensor:
    ids = torch.full((x.shape[0], 1), e, dtype=torch.int32, device="cuda")
    w = torch.ones((x.shape[0], 1), dtype=torch.float32, device="cuda")
    return fused_marlin_moe(
        hidden_states=x.contiguous(),
        w1=qi.w13_qweight,
        w2=qi.w2_qweight,
        w1_scale=qi.w13_scales,
        w2_scale=qi.w2_scales,
        gating_output=w,
        topk_weights=w,
        topk_ids=ids,
        num_bits=4,
    )


def _assert_close(actual: torch.Tensor, expected: torch.Tensor):
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


def _assert_normal_matches(out, x, ids, weights, qi):
    ref = torch.zeros(x.shape[0], K, dtype=torch.float32, device="cuda")
    for e in range(E):
        rows, slots = torch.nonzero(ids == e, as_tuple=True)
        if rows.numel():
            y = _one_expert(x[rows], e, qi).float()
            ref.index_add_(0, rows, weights[rows, slots, None] * y)
    _assert_close(out, ref)


def _assert_ll_matches(out, x, counts, qi):
    for e, n in enumerate(counts):
        if n:
            _assert_close(out[e, :n], _one_expert(x[e, :n], e, qi))


def _random_normal_layout(num_tokens: int, top_k: int):
    x = torch.randn(num_tokens, K, dtype=torch.bfloat16, device="cuda") * 0.1
    # -1: the expert lives on another rank.
    ids = torch.randint(-1, E, (num_tokens, top_k), dtype=torch.int64, device="cuda")
    weights = torch.rand(num_tokens, top_k, dtype=torch.float32, device="cuda")
    return x, ids, weights


def test_normal_layout_runs_local_pairs_in_chunks(monkeypatch):
    torch.manual_seed(0)
    qi = _quant_info(1)
    x, ids, weights = _random_normal_layout(num_tokens=23, top_k=3)
    monkeypatch.setattr(marlin_runner, "_NORMAL_MAX_PAIRS", 5)
    out = marlin_moe_deepep_normal(
        hidden_states=x,
        topk_ids=ids,
        topk_weights=weights,
        quant_info=qi,
        runner_config=CONFIG,
    )
    _assert_normal_matches(out, x, ids, weights, qi)


def test_low_latency_layout_replays_in_cuda_graph():
    torch.manual_seed(0)
    qi = _quant_info(2)
    max_m = 64
    x = torch.randn(E, max_m, K, dtype=torch.bfloat16, device="cuda") * 0.1
    masked_m = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda")

    def run():
        return marlin_moe_deepep_ll(
            hidden_states=x,
            masked_m=masked_m,
            max_rows=E * max_m,
            quant_info=qi,
            runner_config=CONFIG,
        )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):  # compile Triton/Marlin outside the capture
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()

    for counts in ([7, 0, 33, 64], [64, 64, 64, 64], [0, 0, 0, 1]):
        masked_m.copy_(torch.tensor(counts, dtype=torch.int32))
        x.copy_(torch.randn_like(x) * 0.1)
        graph.replay()
        torch.cuda.synchronize()
        _assert_ll_matches(out, x, counts, qi)


def test_registered_entry_handles_both_layouts():
    torch.manual_seed(0)
    qi = _quant_info(3)
    x, ids, weights = _random_normal_layout(num_tokens=9, top_k=2)
    out = fused_experts_deepep_to_marlin(
        DeepEPNormalDispatchOutput(x, None, ids, weights, [0] * E), qi, CONFIG
    )
    assert isinstance(out, DeepEPNormalCombineInput)
    _assert_normal_matches(out.hidden_states, x, ids, weights, qi)

    xl = torch.randn(E, 16, K, dtype=torch.bfloat16, device="cuda") * 0.1
    counts = [3, 0, 16, 1]
    masked_m = torch.tensor(counts, dtype=torch.int32, device="cuda")
    out = fused_experts_deepep_to_marlin(  # eager: sized by the exact row count
        DeepEPLLDispatchOutput(xl, None, ids, weights, masked_m, 4), qi, CONFIG
    )
    assert isinstance(out, DeepEPLLCombineInput)
    _assert_ll_matches(out.hidden_states, xl, counts, qi)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
