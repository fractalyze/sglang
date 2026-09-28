# SPDX-License-Identifier: Apache-2.0
"""Fuse channel-first RMSNorm pointwise work (and an optional SiLU), preserving native FP32 L2 norm."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _channel_rmsnorm_finish_kernel(
    x_ptr,
    norm_ptr,
    weight_ptr,
    out_ptr,
    N: tl.constexpr,
    CHANNELS: tl.constexpr,
    SPATIAL: tl.constexpr,
    SCALE: tl.constexpr,
    SILU: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < N
    channel = index // SPATIAL % CHANNELS
    norm_index = index // (CHANNELS * SPATIAL) * SPATIAL + index % SPATIAL
    value = tl.load(x_ptr + index, mask, 0).to(tl.float32)
    norm = tl.maximum(tl.load(norm_ptr + norm_index, mask, 1), 1.0e-12)
    weight = tl.load(weight_ptr + channel, mask, 0).to(tl.float32)
    # F.normalize divides in FP32, then the original module rounds both the
    # normalized activation and its scale multiply before applying gamma.
    value = tl.div_rn(value, norm).to(x_ptr.dtype.element_ty).to(tl.float32)
    value = (value * SCALE).to(x_ptr.dtype.element_ty).to(tl.float32)
    value = (value * weight).to(x_ptr.dtype.element_ty).to(tl.float32)
    if SILU:
        # aten's SiLU on the rounded norm output: fp32 opmath, IEEE exp and divide
        value = tl.div_rn(value, 1.0 + libdevice.exp(-value))
    tl.store(out_ptr + index, value + 0.0, mask)


def can_use_channel_rmsnorm(x, weight):
    return (
        x.is_cuda
        and torch.version.hip is None
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.ndim in (4, 5)
        and x.numel() > 0
        and x.is_contiguous()
        and weight.device == x.device
        and weight.dtype == x.dtype
        and weight.shape == (x.shape[1],) + (1,) * (x.ndim - 2)
        and weight.is_contiguous()
    )


def _fake_channel_rmsnorm(x, weight, scale, silu=False):
    return torch.empty_like(x)


@register_custom_op(
    op_name="channel_rmsnorm_preserve_reduction",
    mutates_args=[],
    fake_impl=_fake_channel_rmsnorm,
)
def channel_rmsnorm_preserve_reduction(
    x: torch.Tensor, weight: torch.Tensor, scale: float, silu: bool = False
) -> torch.Tensor:
    """F.normalize-based channel RMSNorm, followed by SiLU when ``silu``."""
    assert can_use_channel_rmsnorm(x, weight)
    # Native L2-norm reduction with fp32 accumulation, read straight from the
    # half-precision input; F.normalize runs it on an fp32 copy, and the
    # caller's BitExactFusionGate checks the two agree before trusting it.
    norm = torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True, dtype=torch.float32)
    out = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _channel_rmsnorm_finish_kernel[(triton.cdiv(x.numel(), 512),)](
            x,
            norm,
            weight,
            out,
            x.numel(),
            x.shape[1],
            x.numel() // (x.shape[0] * x.shape[1]),
            scale,
            silu,
            512,
            enable_fp_fusion=False,
        )
    return out
