# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL text-encoder (diffusion runtime) fast paths against their eager references."""

import sys

import pytest
import torch

from sglang.multimodal_gen.runtime.models.encoders.qwen3vl import _make_text_rms_norm
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
