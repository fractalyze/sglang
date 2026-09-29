from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.common import get_device_core_count
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Below this many tokens the StreamK win was not measured; keep sgl-kernel's CUTLASS kernel.
MIN_STREAMK_M = 1024
# The persistent 128x128 tile of sgl-kernel's SM120 kernel, used to size its last wave.
_DEFAULT_TILE_MN = 128


class Sm120ChannelwiseTile(NamedTuple):
    m: int
    n: int
    k: int


def select_sm120_streamk_tile(
    m: int, n: int, k: int, num_sms: int
) -> Optional[Sm120ChannelwiseTile]:
    """StreamK tile for a per-token x per-channel FP8 GEMM on SM120, or None to keep
    sgl-kernel's persistent 128x128x128 kernel.

    StreamK only pays when the persistent kernel's last wave is mostly empty. The
    tile choice was measured on an RTX 5090 (170 SMs) at M=4096 for (K, N) in
    {(4096, 4096), (4096, 12288), (12288, 4096)}; re-tune when the kernel changes.
    """
    if m < MIN_STREAMK_M:
        return None
    tiles = -(-m // _DEFAULT_TILE_MN) * -(-n // _DEFAULT_TILE_MN)
    last_wave = tiles % num_sms or num_sms
    if last_wave > num_sms // 2:
        return None
    if k >= 2 * n:
        return Sm120ChannelwiseTile(128, 256, 64)
    if n >= 2 * k:
        return Sm120ChannelwiseTile(256, 128, 64)
    return Sm120ChannelwiseTile(128, 128, 128)


def _cuda_flags() -> list[str]:
    return [
        "-DNDEBUG",
        "-DCUTE_USE_PACKED_TUPLE=1",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "-DCUTLASS_VERSIONS_GENERATED",
        "-DCUTLASS_TEST_LEVEL=0",
        "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]


@cache_once
def _jit_module(tile: Sm120ChannelwiseTile) -> Module:
    return load_jit(
        "fp8_channelwise_scaled_mm_sm120",
        *tile,
        cuda_files=["gemm/fp8_channelwise/fp8_channelwise_scaled_mm_sm120.cuh"],
        cuda_wrappers=[
            (
                "fp8_channelwise_scaled_mm",
                f"fp8_channelwise_scaled_mm_sm120<{tile.m}, {tile.n}, {tile.k}, true>",
            )
        ],
        extra_dependencies=["cutlass"],
        extra_cuda_cflags=_cuda_flags(),
    )


@cache_once
def _num_sms(device_index: int) -> int:
    return get_device_core_count(device_index)


@register_custom_op(
    op_name="fp8_channelwise_streamk_scaled_mm_sm120",
    mutates_args=["out"],
)
def _fp8_channelwise_streamk_custom_op(
    out: torch.Tensor,
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    tile_m: int,
    tile_n: int,
    tile_k: int,
) -> None:
    module = _jit_module(Sm120ChannelwiseTile(tile_m, tile_n, tile_k))
    module.fp8_channelwise_scaled_mm(out, mat_a, mat_b, scales_a, scales_b)


@debug_kernel_api
def fp8_channelwise_streamk_scaled_mm_sm120(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    out_dtype: torch.dtype,
    tile: Sm120ChannelwiseTile,
) -> torch.Tensor:
    """FP8 e4m3 GEMM with per-token A and per-channel B scales, StreamK-scheduled, SM120.

    mat_a is [M, K] row major, mat_b is [K, N] column major, scales_a holds M and
    scales_b N fp32 values; the result is [M, N] in out_dtype (bf16 or fp16).
    """
    out = torch.empty(
        (mat_a.shape[0], mat_b.shape[1]), dtype=out_dtype, device=mat_a.device
    )
    _fp8_channelwise_streamk_custom_op(
        out, mat_a, mat_b, scales_a, scales_b, tile.m, tile.n, tile.k
    )
    return out


def maybe_fp8_channelwise_streamk_scaled_mm_sm120(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """The StreamK result when select_sm120_streamk_tile picks a tile, else None."""
    m, k = mat_a.shape
    n = mat_b.shape[1]
    tile = select_sm120_streamk_tile(
        m=m, n=n, k=k, num_sms=_num_sms(mat_a.device.index or 0)
    )
    if tile is None:
        return None
    return fp8_channelwise_streamk_scaled_mm_sm120(
        mat_a, mat_b, scales_a, scales_b, out_dtype=out_dtype, tile=tile
    )
