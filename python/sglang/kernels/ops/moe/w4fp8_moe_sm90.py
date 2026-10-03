"""Hopper grouped W4A8 MoE GEMM for large token blocks (see csrc/gemm/w4fp8_moe_sm90/).

Reads the ``w4a16_moe_sm90`` weight layout unchanged, so one copy of the expert
weights serves both kernels: 4-bit codes are dequantised to exact e4m3 ``q - z``
in registers and multiplied on FP8 wgmma against per-token, per-group e4m3
activations; the weight and activation group scales are applied to each group's
FP32 partial.

The reduction axis is permuted within every 16 columns (``K_PERMUTE16``) so the
w4a16 bf16 fragment order lines up with the e4m3 fragment the FP8 wgmma reads;
``quantize_activations`` writes activations in that same order.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.kernels.ops.moe.w4a16_moe_sm90 import TILE_K, TILE_N, W4A16MoeWeights

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# e4m3 column f of every 16-column block holds weight column K_PERMUTE16[f]:
# the bf16 m16k16 A columns {2t, 2t + 1, 2t + 8, 2t + 9} of thread t, listed in
# the e4m3 m16k32 order {4t, ..., 4t + 3}. Must match csrc/gemm/w4fp8_moe_sm90/fragment.cuh.
K_PERMUTE16 = tuple(2 * (f // 4) + f % 2 + 8 * (f % 4 // 2) for f in range(16))

_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

# Must match csrc/gemm/w4fp8_moe_sm90/kernel.cuh, which allows 16 to 128.
TOKEN_BLOCK = 128


@triton.jit
def _quantize_activations_kernel(
    a_ptr,
    q_ptr,
    scales_ptr,
    k,
    k_groups,
    E4M3_MAX: tl.constexpr,
    GROUP: tl.constexpr,
    GROUPS_PER_PROGRAM: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    first_group = tl.program_id(1) * GROUPS_PER_PROGRAM
    groups = first_group + tl.arange(0, GROUPS_PER_PROGRAM)
    offsets = groups[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
    x = tl.load(a_ptr + row * k + offsets).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-10) / E4M3_MAX
    q = tl.clamp(x / scale[:, None], -E4M3_MAX, E4M3_MAX)
    # Column 8h + 2t + l of each 16 moves to 4t + 2h + l: K_PERMUTE16 as a (h, t) transpose.
    q = tl.reshape(q, (GROUPS_PER_PROGRAM, GROUP // 16, 2, 4, 2))
    q = tl.reshape(tl.permute(q, (0, 1, 3, 2, 4)), (GROUPS_PER_PROGRAM, GROUP))
    tl.store(q_ptr + row * k + offsets, q.to(q_ptr.dtype.element_ty))
    tl.store(scales_ptr + row * k_groups + groups, scale)


@debug_kernel_api
def quantize_activations(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [R, K] -> (e4m3 [R, K] in K_PERMUTE16 order, fp32 scales [R, K / 128]).

    One dynamic scale per row and 128-column group, the AWQ group the weights use.
    """
    rows, k = a.shape
    assert a.is_contiguous() and k % TILE_K == 0, a.shape
    k_groups = k // TILE_K
    q = torch.empty((rows, k), dtype=torch.float8_e4m3fn, device=a.device)
    scales = torch.empty((rows, k_groups), dtype=torch.float32, device=a.device)
    # Up to eight groups (2 KB of bf16) per program; must divide the row's groups.
    groups_per_program = math.gcd(k_groups, 8)
    if rows > 0:
        _quantize_activations_kernel[(rows, k_groups // groups_per_program)](
            a,
            q,
            scales,
            k,
            k_groups,
            E4M3_MAX=_E4M3_MAX,
            GROUP=TILE_K,
            GROUPS_PER_PROGRAM=groups_per_program,
        )
    return q, scales


@cache_once
def _jit_w4fp8_moe_sm90_module(token_block: int) -> Module:
    if torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("w4fp8_moe_sm90 requires an SM90 (Hopper) GPU")
    args = make_cpp_args(token_block)
    return load_jit(
        "w4fp8_moe_sm90",
        *args,
        cuda_files=["gemm/w4fp8_moe_sm90/kernel.cuh"],
        cuda_wrappers=[("run", f"sglang::W4Fp8MoeSm90Kernel<{args}>::run")],
        extra_cuda_cflags=[
            "-O3",
            "-DNDEBUG",
            "-DCUTE_USE_PACKED_TUPLE=1",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        ],
        extra_dependencies=["cutlass"],
    )


@debug_kernel_api
def w4fp8_moe_sm90_gemm(
    a: torch.Tensor,
    a_scales: torch.Tensor,
    out: torch.Tensor,
    weights: W4A16MoeWeights,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor | None,
    a_row_divisor: int,
) -> None:
    """``w4a16_moe_sm90_gemm`` on ``quantize_activations(a)``, routed by TOKEN_BLOCK.

    ``sorted_token_ids``, ``expert_ids`` and ``num_tokens_post_padded`` come from
    ``moe_align_block_size(..., TOKEN_BLOCK, ...)``; the routing and
    ``a_row_divisor`` semantics are those of ``w4a16_moe_sm90_gemm``.
    """
    qweight, scales, zeros = weights
    if a.dtype != torch.float8_e4m3fn or a_scales.dtype != torch.float32:
        raise TypeError("w4fp8_moe_sm90 takes quantize_activations output")
    if scales.dtype != torch.bfloat16:
        raise TypeError("w4fp8_moe_sm90 supports bfloat16 weight scales")
    assert a.is_contiguous() and a_scales.is_contiguous() and out.is_contiguous()
    assert a.shape[1] == qweight.shape[2] * TILE_K, (a.shape, qweight.shape)
    assert out.shape[1] == qweight.shape[1] * TILE_N, (out.shape, qweight.shape)
    assert sorted_token_ids.dtype == torch.int32 and expert_ids.dtype == torch.int32
    if topk_weights is not None:
        assert topk_weights.dtype == torch.float32 and topk_weights.is_contiguous()
        assert topk_weights.numel() == out.shape[0]
    module = _jit_w4fp8_moe_sm90_module(TOKEN_BLOCK)
    module.run(
        a,
        a_scales,
        out,
        qweight,
        scales,
        zeros,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        (
            topk_weights
            if topk_weights is not None
            else torch.empty(0, dtype=torch.float32, device=out.device)
        ),
        a_row_divisor,
    )
