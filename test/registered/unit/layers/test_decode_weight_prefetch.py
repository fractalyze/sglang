"""DecodeWeightPrefetch: the prefetch is planned outside capture, joins its side
stream, and attaches only to MoE layers that have a next local layer."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.decode_weight_prefetch import (
    DecodeWeightPrefetch,
    attach_decode_weight_prefetch,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    reason="l2_prefetch requires SM90 or newer",
)


def _prefetch(weights):
    return DecodeWeightPrefetch(
        weights=lambda: weights, budget_bytes=1 << 30, stream=torch.cuda.Stream()
    )


def test_first_call_inside_capture_skips_and_leaves_no_plan():
    weight = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(16, 1024, device="cuda", dtype=torch.bfloat16)
    expected = x @ weight
    prefetch = _prefetch([weight])

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with prefetch.overlapping():
            out = x @ weight
    graph.replay()
    torch.cuda.synchronize()

    assert prefetch._ranges is None
    assert torch.equal(out, expected)


def test_planned_prefetch_replays_in_graph_with_unchanged_result():
    weight = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16)
    prefetch = _prefetch([weight])
    with prefetch.overlapping():
        eager = x @ weight
    assert prefetch._ranges is not None

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with prefetch.overlapping():
            out = x @ weight
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager)


def test_attaches_to_moe_layers_with_a_next_local_layer():
    def layer(sparse):
        attn = SimpleNamespace(decode_weights_before_core=lambda: [])
        return SimpleNamespace(
            is_layer_sparse=sparse, decode_weight_prefetch=None, self_attn=attn
        )

    layers = {i: layer(sparse=i >= 3) for i in range(2, 6)}
    attach_decode_weight_prefetch(layers, [2, 3, 4, 5], budget_bytes=1 << 20)

    attached = [i for i, l in layers.items() if l.decode_weight_prefetch is not None]
    assert attached == [3, 4]
    assert (
        layers[3].decode_weight_prefetch._weights
        is layers[4].self_attn.decode_weights_before_core
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
