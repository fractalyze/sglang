# SPDX-License-Identifier: Apache-2.0
"""Channel RMSNorm (+ optional SiLU) for channels_last activations, in one pass.

With channels innermost, one program loads a tile of pixels x all channels,
reduces the fp32 sum of squares over the channel axis in registers and writes
the result, so the input is read once. The per-element rounding follows the
NCHW path (channel_rmsnorm_preserve_reduction); the reduction order differs,
so results match it only to rounding, not bitwise.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from sglang.srt.utils.custom_op import register_custom_op

# Channel tile is next_power_of_2(C) in registers; wider inputs take the NCHW path.
MAX_CHANNELS = 2048


@triton.jit
def _channel_rmsnorm_nhwc_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    PIXELS,
    CHANNELS: tl.constexpr,
    SCALE: tl.constexpr,
    SILU: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pix = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    ch = tl.arange(0, BLOCK_C)
    mask = (pix[:, None] < PIXELS) & (ch[None, :] < CHANNELS)
    offs = pix[:, None] * CHANNELS + ch[None, :]
    value = tl.load(x_ptr + offs, mask, 0).to(tl.float32)
    norm = tl.maximum(tl.sqrt_rn(tl.sum(value * value, axis=1)), 1.0e-12)
    weight = tl.load(weight_ptr + ch, ch < CHANNELS, 0).to(tl.float32)
    value = tl.div_rn(value, norm[:, None]).to(x_ptr.dtype.element_ty).to(tl.float32)
    value = (value * SCALE).to(x_ptr.dtype.element_ty).to(tl.float32)
    value = (value * weight[None, :]).to(x_ptr.dtype.element_ty).to(tl.float32)
    if SILU:
        value = tl.div_rn(value, 1.0 + libdevice.exp(-value))
    tl.store(out_ptr + offs, value + 0.0, mask)


def can_use_channel_rmsnorm_nhwc(x, weight):
    """True for a channels_last (4D) or channels_last_3d (5D) half-precision CUDA tensor."""
    if not (x.is_cuda and torch.version.hip is None and x.ndim in (4, 5)):
        return False
    memory_format = torch.channels_last_3d if x.ndim == 5 else torch.channels_last
    return (
        x.dtype in (torch.bfloat16, torch.float16)
        and x.numel() > 0
        and not x.is_contiguous()
        and x.is_contiguous(memory_format=memory_format)
        and x.shape[1] <= MAX_CHANNELS
        and weight.device == x.device
        and weight.dtype == x.dtype
        and weight.numel() == x.shape[1]
        and weight.is_contiguous()
    )


def _fake_channel_rmsnorm_nhwc(x, weight, scale, silu=False):
    return torch.empty_like(x)


@register_custom_op(
    op_name="channel_rmsnorm_nhwc",
    mutates_args=[],
    fake_impl=_fake_channel_rmsnorm_nhwc,
)
def channel_rmsnorm_nhwc(
    x: torch.Tensor, weight: torch.Tensor, scale: float, silu: bool = False
) -> torch.Tensor:
    """F.normalize-style channel RMSNorm of a channels_last tensor, then SiLU when ``silu``."""
    assert can_use_channel_rmsnorm_nhwc(x, weight)
    channels = x.shape[1]
    pixels = x.numel() // channels
    out = torch.empty_like(x)  # keeps the channels_last strides
    block_c = triton.next_power_of_2(channels)
    block_p = max(1, 4096 // block_c)  # 4096 fp32 values per program
    with torch.cuda.device(x.device):
        _channel_rmsnorm_nhwc_kernel[(triton.cdiv(pixels, block_p),)](
            x, weight, out, pixels, channels, scale, silu, block_p, block_c,
            num_warps=4, enable_fp_fusion=False,
        )
    return out
