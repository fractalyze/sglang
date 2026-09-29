from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Below this many tokens the cuBLASLt route was not measured; callers keep their kernel.
MIN_CUBLASLT_M = 1024
# cuBLASLt's fastest FP8 algos on SM120 need a large workspace; 32 MiB was measured.
_WORKSPACE_BYTES = 32 << 20
# Arbitrary: enough repetitions to rank algos that differ by >= 5% at M >= 1024.
_TUNE_REPS = 5

_algo_cache: Dict[Tuple[int, int, int, int], int] = {}


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "fp8_cublaslt_unit_scale_gemm",
        cuda_files=["gemm/fp8_cublaslt/fp8_unit_scale_gemm.cuh"],
        cuda_wrappers=[
            ("gemm", "fp8_unit_scale_gemm"),
            ("num_algos", "fp8_unit_scale_gemm_num_algos"),
        ],
        extra_ldflags=["-lcublasLt"],
    )


@cache_once
def _workspace(device_index: int) -> torch.Tensor:
    return torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=f"cuda:{device_index}")


def _select_algo(module: Module, out, mat_a, w_nk, workspace) -> int:
    """Fastest of cuBLASLt's heuristic algos for this shape, timed once and cached.

    cuBLASLt's first choice is not always the fastest on SM120 (11% slower on
    M=4096, K=12288, N=4096). Under CUDA graph capture, take the first choice.
    """
    key = (mat_a.device.index, mat_a.shape[0], w_nk.shape[0], mat_a.shape[1])
    algo = _algo_cache.get(key)
    if algo is not None:
        return algo
    num_algos = module.num_algos(mat_a, w_nk, workspace)
    if num_algos == 1 or torch.cuda.is_current_stream_capturing():
        return 0
    times = []
    for idx in range(num_algos):
        module.gemm(out, mat_a, w_nk, workspace, idx)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(_TUNE_REPS):
            module.gemm(out, mat_a, w_nk, workspace, idx)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    algo = min(range(num_algos), key=times.__getitem__)
    _algo_cache[key] = algo
    return algo


@triton.jit
def _scale_rows_cols_kernel(out_ptr, sa_ptr, sb_ptr, N, stride_m, BLOCK_N: tl.constexpr):
    m = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    sa = tl.load(sa_ptr + m)
    sb = tl.load(sb_ptr + offs, mask=mask, other=0.0)
    ptrs = out_ptr + m * stride_m + offs
    x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    tl.store(ptrs, (x * (sa * sb)).to(out_ptr.dtype.element_ty), mask=mask)


@register_custom_op(
    op_name="fp8_unit_scale_gemm_cublaslt",
    mutates_args=["out"],
)
def _fp8_unit_scale_gemm_cublaslt_op(
    out: torch.Tensor, mat_a: torch.Tensor, w_nk: torch.Tensor
) -> None:
    module = _jit_module()
    workspace = _workspace(mat_a.device.index or 0)
    module.gemm(out, mat_a, w_nk, workspace, _select_algo(module, out, mat_a, w_nk, workspace))


@debug_kernel_api
def fp8_unit_scale_gemm_cublaslt(mat_a: torch.Tensor, w_nk: torch.Tensor) -> torch.Tensor:
    """bf16 out[M, N] = mat_a[M, K] @ w_nk[N, K]^T (e4m3, fp32 accumulation, unit scales).

    The caller owns the per-token and per-channel scales; see
    apply_fp8_linear_deferred_scale.
    """
    out = torch.empty(
        (mat_a.shape[0], w_nk.shape[0]), dtype=torch.bfloat16, device=mat_a.device
    )
    _fp8_unit_scale_gemm_cublaslt_op(out, mat_a, w_nk)
    return out


@register_custom_op(
    op_name="fp8_per_channel_scaled_mm_cublaslt",
    mutates_args=["out"],
)
def _fp8_per_channel_scaled_mm_cublaslt_op(
    out: torch.Tensor,
    mat_a: torch.Tensor,
    w_nk: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
) -> None:
    module = _jit_module()
    workspace = _workspace(mat_a.device.index or 0)
    algo = _select_algo(module, out, mat_a, w_nk, workspace)
    module.gemm(out, mat_a, w_nk, workspace, algo)
    m, n = out.shape
    block_n = 2048
    _scale_rows_cols_kernel[(m, triton.cdiv(n, block_n))](
        out, scales_a, scales_b, n, out.stride(0), BLOCK_N=block_n, num_warps=8
    )


@debug_kernel_api
def fp8_per_channel_scaled_mm_cublaslt(
    mat_a: torch.Tensor,
    w_nk: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
) -> torch.Tensor:
    """bf16 out[M, N] = scales_a[M] * scales_b[N] * (mat_a[M, K] @ w_nk[N, K]^T), e4m3 inputs.

    The GEMM runs in cuBLASLt with unit per-tensor scales and fp32 accumulation into
    bf16; the scales are applied by a second in-place pass. cuBLASLt on SM120 has no
    per-row or per-column FP8 scale mode, so the unscaled product is rounded to bf16
    once more than in a fused-epilogue kernel (relative error about 2.8e-3).
    """
    out = torch.empty(
        (mat_a.shape[0], w_nk.shape[0]), dtype=torch.bfloat16, device=mat_a.device
    )
    _fp8_per_channel_scaled_mm_cublaslt_op(
        out, mat_a, w_nk, scales_a.reshape(-1), scales_b.reshape(-1)
    )
    return out


def maybe_fp8_per_channel_scaled_mm_cublaslt(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """The cuBLASLt route for apply_fp8_linear's per-token x per-channel GEMM, or None.

    mat_b is the [K, N] column-major view of the stored [N, K] weight.
    """
    w_nk = mat_b.t()
    if (
        mat_a.shape[0] < MIN_CUBLASLT_M
        or out_dtype != torch.bfloat16
        or not mat_a.is_contiguous()
        or not w_nk.is_contiguous()
        or not scales_a.is_contiguous()
        or not scales_b.is_contiguous()
        or scales_a.numel() != mat_a.shape[0]
        or scales_b.numel() != w_nk.shape[0]
    ):
        return None
    return fp8_per_channel_scaled_mm_cublaslt(mat_a, w_nk, scales_a, scales_b)
