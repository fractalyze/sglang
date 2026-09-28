# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL text-encoder (diffusion runtime) fast paths against their eager references."""

import sys

import pytest
import torch

from sglang.multimodal_gen.runtime.models.encoders.qwen3vl import _make_text_rms_norm
from sglang.multimodal_gen.runtime.utils.layer_graphs import ShapeKeyedCudaGraphs
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)

EPS = 1e-6


def text_norm(hidden, fused):
    with envs.SGLANG_ENABLE_QWEN3VL_TEXT_FUSED_RMSNORM.override(fused):
        norm = _make_text_rms_norm(hidden, EPS).cuda().bfloat16()
    torch.manual_seed(1)
    norm.weight.data = torch.rand_like(norm.weight) + 0.5
    return norm


@pytest.mark.parametrize("hidden,tokens", [(128, 34 * 32), (4096, 34), (4096, 300)])
def test_fused_text_rmsnorm_matches_native_to_rounding(hidden, tokens):
    torch.manual_seed(0)
    x = torch.randn(1, tokens, hidden, device="cuda", dtype=torch.bfloat16) * 4
    native, fused = text_norm(hidden, False), text_norm(hidden, True)
    assert native._forward_method.__func__ is type(native).forward_native
    assert fused._resolve_forward_method().__func__ is not type(fused).forward_native
    reference = native(x)
    torch.testing.assert_close(native.forward_native(x), reference, atol=0, rtol=0)
    actual = fused(x)
    # One bf16 ulp at most: only the variance reduction order differs.
    torch.testing.assert_close(actual, reference, atol=3.2e-2, rtol=8e-3)
    assert (actual != reference).float().mean() < 0.05


class ToyLayers(torch.nn.Module):
    """A stack of pure-GPU layers standing in for the text encoder's decoder layers."""

    def __init__(self, width=256, depth=3):
        super().__init__()
        self.layers = torch.nn.ModuleList(torch.nn.Linear(width, width) for _ in range(depth))

    def outputs(self, hidden, scale):
        out = []
        for layer in self.layers:
            hidden = torch.nn.functional.silu(layer(hidden)) * scale
            out.append(hidden)
        return tuple(out)


def test_layer_graphs_replay_matches_eager_per_shape():
    torch.manual_seed(0)
    model = ToyLayers().cuda().bfloat16()
    graphs = ShapeKeyedCudaGraphs(model)
    scale = torch.tensor(0.5, device="cuda", dtype=torch.bfloat16)
    for tokens in (34, 51, 34):
        x = torch.randn(1, tokens, 256, device="cuda", dtype=torch.bfloat16)
        replayed = graphs.run(model.outputs, x, scale)
        eager = model.outputs(x, scale)
        assert len(replayed) == len(eager)
        for r, e in zip(replayed, eager):
            torch.testing.assert_close(r, e, atol=0, rtol=0)


def test_layer_graphs_recapture_when_weights_move():
    torch.manual_seed(0)
    model = ToyLayers().cuda().bfloat16()
    graphs = ShapeKeyedCudaGraphs(model)
    scale = torch.tensor(1.0, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(1, 34, 256, device="cuda", dtype=torch.bfloat16)
    graphs.run(model.outputs, x, scale)
    # new storage with new values, as after an offload round trip plus an update
    for p in model.parameters():
        p.data = (p.data * 2).clone()
    torch.testing.assert_close(graphs.run(model.outputs, x, scale)[-1],
                               model.outputs(x, scale)[-1], atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
