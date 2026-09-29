"""FP8 cuDNN convs of the Qwen-Image 2.1 VAE decoder (SGLANG_ENABLE_QWEN_IMAGE21_VAE_FP8_CONV)."""

import pytest
import torch

from sglang.kernels.ops.diffusion.conv import fp8_conv_cudnn
from sglang.kernels.ops.diffusion.norm.channel_rmsnorm_nhwc import channel_rmsnorm_nhwc
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and fp8_conv_cudnn.is_available()),
    reason="requires CUDA and the cuDNN frontend Python package",
)

E4M3 = torch.float8_e4m3fn


def _conv(cin, cout):
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(cin, cout, 3, padding=1).cuda()
    return conv


def _reference(conv, x8, descale, residual=None):
    """fp32 conv of the dequantized operands, then the single bf16 rounding."""
    fp8 = fp8_conv_cudnn.Fp8Conv3x3(conv)
    w = fp8.weight.float() * fp8.w_scale.view(-1, 1, 1, 1)
    y = torch.nn.functional.conv2d(
        x8.float() * descale, w, padding=1
    ) + conv.bias.float().view(1, -1, 1, 1)
    if residual is not None:
        y = y + residual.squeeze(2).float()
    return y.to(torch.bfloat16)


@pytest.mark.parametrize("cin, cout, size", [(144, 144, 64), (288, 144, 32)])
@pytest.mark.parametrize("with_residual", [False, True])
def test_static_scale_conv_matches_the_dequantized_conv(cin, cout, size, with_residual):
    conv = _conv(cin, cout)
    x = torch.randn(1, cin, size, size, device="cuda").contiguous(
        memory_format=torch.channels_last
    )
    act_scale = 0.01
    x8 = (x / act_scale).to(E4M3)
    residual = None
    if with_residual:
        residual = torch.randn(
            1, cout, 1, size, size, device="cuda", dtype=torch.bfloat16
        ).contiguous(memory_format=torch.channels_last_3d)
    out = fp8_conv_cudnn.Fp8Conv3x3(conv, act_scale)(x8, residual=residual)
    assert out.shape == (1, cout, 1, size, size) and out.dtype == torch.bfloat16
    assert out.is_contiguous(memory_format=torch.channels_last_3d)
    expected = _reference(conv, x8, act_scale, residual)
    torch.testing.assert_close(
        out.squeeze(2).float(), expected.float(), rtol=2e-2, atol=2e-2
    )


def test_dynamic_scale_conv_reads_the_device_scale_each_call():
    conv = _conv(144, 144)
    fp8 = fp8_conv_cudnn.Fp8Conv3x3(conv)
    x = torch.randn(1, 144, 32, 32, device="cuda").contiguous(
        memory_format=torch.channels_last
    )
    for value in (0.02, 0.005):
        scale = torch.tensor([value], device="cuda")
        x8 = (x / value).to(E4M3)
        out = fp8(x8, act_scale=scale)
        torch.testing.assert_close(
            out.squeeze(2).float(),
            _reference(conv, x8, value).float(),
            rtol=2e-2,
            atol=2e-2,
        )


def test_norm_silu_fp8_is_the_bf16_norm_then_e4m3():
    # The fused kernel's channel reduction compiles differently from
    # channel_rmsnorm_nhwc's, so a bf16 value can land one ulp apart and cross an
    # e4m3 rounding boundary: rare, and never more than one e4m3 step.
    torch.manual_seed(0)
    c = 288
    x = torch.randn(1, c, 1, 48, 40, device="cuda", dtype=torch.bfloat16).contiguous(
        memory_format=torch.channels_last_3d
    )
    gamma = (torch.rand(c, 1, 1, device="cuda") + 0.5).to(torch.bfloat16)
    scale = fp8_conv_cudnn.norm_act_scale(gamma)
    out = fp8_conv_cudnn.norm_silu_fp8(x, gamma, c**0.5, scale)
    reference = channel_rmsnorm_nhwc(x, gamma, c**0.5, True).squeeze(2)
    expected = (
        (reference.float() * (1.0 / scale))
        .to(E4M3)
        .contiguous(memory_format=torch.channels_last)
    )
    assert out.is_contiguous(memory_format=torch.channels_last)
    differ = out.view(torch.uint8) != expected.view(torch.uint8)
    assert differ.float().mean().item() < 1e-4
    a, b = out.float()[differ], expected.float()[differ]
    assert bool(((a - b).abs() <= b.abs() * 0.125 + 2.0**-9).all())


def test_upsample2x_fp8_is_bytewise_the_nearest_upsample_then_e4m3():
    torch.manual_seed(0)
    x = torch.randn(1, 96, 20, 24, device="cuda", dtype=torch.bfloat16).contiguous(
        memory_format=torch.channels_last
    )
    out, scale = fp8_conv_cudnn.upsample2x_fp8(x)
    torch.testing.assert_close(
        scale, (x.abs().amax().float() / fp8_conv_cudnn.E4M3_MAX).reshape(1)
    )
    up = torch.nn.functional.interpolate(x.float(), scale_factor=2.0, mode="nearest")
    expected = (
        (up * (1.0 / scale)).to(E4M3).contiguous(memory_format=torch.channels_last)
    )
    assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))


def _rel_l2(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("in_dim, out_dim", [(128, 128), (128, 64)])
@pytest.mark.parametrize("channels_last_input", [True, False])
def test_residual_block_fp8_path_is_close_to_bf16(in_dim, out_dim, channels_last_input):
    from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage21 import (
        QwenImage21ResidualBlock,
    )

    torch.manual_seed(0)
    block = QwenImage21ResidualBlock(in_dim, out_dim).cuda().to(torch.bfloat16).eval()
    for module in block.modules():
        if isinstance(module, torch.nn.Conv2d):
            module.weight.data = module.weight.data.contiguous(
                memory_format=torch.channels_last
            )
    x = torch.randn(1, in_dim, 1, 64, 64, device="cuda", dtype=torch.bfloat16)
    if channels_last_input:
        x = x.contiguous(memory_format=torch.channels_last_3d)
    with torch.no_grad():
        expected = block(x)
        block.prepare_fp8_convs()
        out = block(x)
    assert out.is_contiguous(memory_format=torch.channels_last_3d)
    assert _rel_l2(out, expected) < 0.05


def test_upsample_fp8_path_is_close_to_bf16():
    from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage21 import (
        QwenImage21Resample,
    )

    torch.manual_seed(0)
    resample = QwenImage21Resample(128, "upsample2d").cuda().to(torch.bfloat16).eval()
    x = torch.randn(1, 128, 1, 32, 32, device="cuda", dtype=torch.bfloat16).contiguous(
        memory_format=torch.channels_last_3d
    )
    with torch.no_grad():
        expected = resample(x)
        resample.fp8_conv = fp8_conv_cudnn.Fp8Conv3x3(resample.resample[1])
        out = resample(x)
    assert out.shape == expected.shape == (1, 64, 1, 64, 64)
    assert _rel_l2(out, expected) < 0.05
