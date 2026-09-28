# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image 2.1 VAE decoder fast paths against their eager references."""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion.norm.channel_rmsnorm_nhwc import (
    can_use_channel_rmsnorm_nhwc,
    channel_rmsnorm_nhwc,
)
from sglang.kernels.ops.diffusion.norm.channel_rmsnorm_preserve_reduction import (
    channel_rmsnorm_preserve_reduction,
)
from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage21 import (
    QwenImage21CausalConv3d,
    QwenImage21Decoder3d,
    QwenImage21Resample,
    QwenImage21RMS_norm,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)


@pytest.mark.parametrize("channels,size,kernel", [(96, 256, 3), (192, 128, 3), (384, 64, 1)])
def test_causal_conv_implicit_padding_matches_padded_copy(channels, size, kernel):
    torch.manual_seed(0)
    conv = QwenImage21CausalConv3d(channels, channels, kernel, padding=kernel // 2)
    conv = conv.cuda().bfloat16()
    x = torch.randn(1, channels, 1, size, size, device="cuda", dtype=torch.bfloat16)
    padded = F.pad(x.squeeze(2), list(conv._padding))
    reference = F.conv2d(padded, conv.weight, conv.bias, conv.stride).unsqueeze(2)
    # Same math; cuDNN may pick another algorithm for the unpadded shape.
    torch.testing.assert_close(conv(x), reference, atol=2e-2, rtol=2e-2)


# The decoder's norm inputs at 1024x1024 (channels x resolution).
DECODER_NORM_SHAPES = [(1, 384, 1, 128, 128), (1, 192, 1, 512, 512), (1, 96, 1, 1024, 1024)]


def eager_norm(x, gamma, scale):
    """The module's eager path: F.normalize in fp32, rounded like the original."""
    normalized = F.normalize(x.float(), dim=1).to(x.dtype)
    return normalized * scale * gamma


@pytest.mark.parametrize("shape", DECODER_NORM_SHAPES)
@pytest.mark.parametrize("silu", [False, True])
def test_channel_rmsnorm_silu_is_bitwise(shape, silu):
    torch.manual_seed(0)
    x = (torch.randn(shape, device="cuda") * 3).bfloat16()
    gamma = (torch.rand(shape[1], 1, 1, 1, device="cuda") + 0.5).bfloat16()
    scale = shape[1] ** 0.5
    reference = eager_norm(x, gamma, scale)
    if silu:
        reference = F.silu(reference)
    actual = channel_rmsnorm_preserve_reduction(x, gamma, scale, silu)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)


def test_rms_norm_module_fused_silu_matches_eager():
    torch.manual_seed(0)
    norm = QwenImage21RMS_norm(192, images=False).cuda().bfloat16()
    norm.gamma.data = torch.rand_like(norm.gamma) + 0.5
    x = torch.randn(1, 192, 1, 64, 64, device="cuda", dtype=torch.bfloat16)
    expected = F.silu(eager_norm(x, norm.gamma, norm.scale))
    for _ in range(2):  # first call verifies the fused path, the second trusts it
        torch.testing.assert_close(norm(x, silu=True), expected, atol=0, rtol=0)


@pytest.mark.parametrize("shape", DECODER_NORM_SHAPES + [(1, 1152, 1, 64, 64)])
@pytest.mark.parametrize("silu", [False, True])
def test_channel_rmsnorm_nhwc_matches_nchw_to_rounding(shape, silu):
    torch.manual_seed(0)
    x = (torch.randn(shape, device="cuda") * 3).bfloat16()
    gamma = (torch.rand(shape[1], 1, 1, 1, device="cuda") + 0.5).bfloat16()
    scale = shape[1] ** 0.5
    x_nhwc = x.contiguous(memory_format=torch.channels_last_3d)
    assert can_use_channel_rmsnorm_nhwc(x_nhwc, gamma)
    assert not can_use_channel_rmsnorm_nhwc(x, gamma)
    actual = channel_rmsnorm_nhwc(x_nhwc, gamma, scale, silu)
    reference = eager_norm(x, gamma, scale)
    if silu:
        reference = F.silu(reference)
    assert actual.stride() == x_nhwc.stride()
    # Another reduction order: the last bit differs on a few elements.
    torch.testing.assert_close(actual, reference, atol=1.6e-2, rtol=1.6e-2)
    rel = (actual.float() - reference.float()).norm() / reference.float().norm()
    assert rel < 1e-4


def test_resample_single_frame_is_unchanged_and_keeps_layout():
    torch.manual_seed(0)
    resample = QwenImage21Resample(64, mode="upsample2d", upsample_out_dim=64).cuda().bfloat16()
    x = torch.randn(1, 64, 1, 32, 32, device="cuda", dtype=torch.bfloat16)
    reference = resample.resample(x[:, :, 0]).unsqueeze(2)
    torch.testing.assert_close(resample(x), reference, atol=0, rtol=0)
    for conv in resample.modules():
        if isinstance(conv, torch.nn.Conv2d):
            conv.weight.data = conv.weight.data.contiguous(memory_format=torch.channels_last)
    out = resample(x.contiguous(memory_format=torch.channels_last_3d))
    assert out.squeeze(2).is_contiguous(memory_format=torch.channels_last)


def test_channels_last_decoder_matches_nchw():
    torch.manual_seed(0)
    decoder = QwenImage21Decoder3d(
        dim=32, z_dim=4, dim_mult=[1, 2, 2], num_res_blocks=1,
        temperal_upsample=[False, False], is_residual=True,
    ).cuda().bfloat16().eval()
    z = torch.randn(1, 4, 1, 32, 32, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        reference = decoder(z, first_chunk=True)
        for conv in decoder.modules():
            if isinstance(conv, torch.nn.Conv2d):
                conv.weight.data = conv.weight.data.contiguous(memory_format=torch.channels_last)
        actual = decoder(z.contiguous(memory_format=torch.channels_last_3d), first_chunk=True)
    rel = (actual.float() - reference.float()).norm() / reference.float().norm()
    assert rel < 1e-2, rel


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
