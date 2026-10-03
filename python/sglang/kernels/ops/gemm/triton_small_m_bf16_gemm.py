"""Small-M BF16 GEMM ``y = x @ w.T`` for decode batches on SM120.

On the RTX 5090 cuBLAS (and cuBLASLt's heuristics) run small-M BF16 GEMMs on
the SM80 WMMA fallback ``cutlass_80_wmma_tensorop_bf16_s161616gemm``, which
reaches 0.55-0.65 of DRAM bandwidth at M=8-16 on Gemma-4's o_proj and dense
MLP shapes. This kernel streams each weight row block once with fp32
accumulation in K order. Narrow-N shapes (the MoE router) cannot fill the GPU
from N tiles alone, so they split K and reduce fp32 partials in a second launch.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Tuple

import torch
import triton
import triton.language as tl

# Decode batches only; prefill keeps cuBLAS, which wins at M=1024.
MAX_M = 32


class _TileConfig(NamedTuple):
    block_n: int
    block_k: int
    num_stages: int
    split_k: int = 1


# Measured on RTX 5090 at M in {1, 8, 16, 32} (gemma4nv T3 and W7 microbenches).
# Shapes outside this allowlist keep cuBLAS.
_TUNED_SHAPES: Dict[Tuple[int, int], _TileConfig] = {
    # Gemma-4-26B-A4B o_proj: sliding (16 x 256 heads) and full (16 x 512 heads).
    (2816, 4096): _TileConfig(32, 256, 3),
    (2816, 8192): _TileConfig(32, 256, 3),
    # Gemma-4-26B-A4B dense MLP: gate_up (2 x 2112) and down.
    (4224, 2816): _TileConfig(32, 256, 4),
    (2816, 2112): _TileConfig(32, 128, 4),
    # Gemma-4-26B-A4B qkv_proj: sliding (16 + 2 x 8 heads of 256), full (16 + 2 x 2 of 512).
    (8192, 2816): _TileConfig(32, 128, 4),
    (10240, 2816): _TileConfig(32, 128, 4),
    # Gemma-4-26B-A4B MoE router (128 experts).
    (128, 2816): _TileConfig(16, 128, 3, split_k=8),
    # Gemma-4-26B-A4B tied lm_head (vocab 262144).
    (262144, 2816): _TileConfig(32, 256, 4),
}


def use_triton_small_m_bf16_gemm(m: int, n: int, k: int) -> bool:
    return m <= MAX_M and (n, k) in _TUNED_SHAPES


@triton.jit
def _small_m_bf16_gemm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    K,
    K_PER_SPLIT,
    stride_xm,
    stride_wn,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """out[split, m, n] = sum over this split's K range of x[m, k] * w[n, k].

    With one split ``out`` is the bf16 result; with several it is an fp32 partial.
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_s = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_ptr += pid_s * M * stride_om
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_start = pid_s * K_PER_SPLIT
    for k0 in range(k_start, k_start + K_PER_SPLIT, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
            other=0.0,
        )
        acc = tl.dot(x, tl.trans(w), acc)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _split_k_reduce_kernel(
    partial_ptr, out_ptr, MN, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        acc += tl.load(partial_ptr + s * MN + offs, mask=offs < MN, other=0.0)
    tl.store(out_ptr + offs, acc.to(tl.bfloat16), mask=offs < MN)


def triton_small_m_bf16_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``F.linear(x, weight)`` for a 2D bf16 ``x`` [M, K] and row-major ``weight`` [N, K]."""
    m, k = x.shape
    n = weight.shape[0]
    cfg = _TUNED_SHAPES[(n, k)]
    # tl.dot needs at least 16 rows; padding rows are masked loads of zero.
    block_m = max(16, triton.next_power_of_2(m))
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if cfg.split_k == 1:
        gemm_out = out
    else:
        gemm_out = torch.empty((cfg.split_k, m, n), dtype=torch.float32, device=x.device)
    k_per_split = triton.cdiv(triton.cdiv(k, cfg.split_k), cfg.block_k) * cfg.block_k
    grid = (triton.cdiv(n, cfg.block_n), triton.cdiv(m, block_m), cfg.split_k)
    _small_m_bf16_gemm_kernel[grid](
        x,
        weight,
        gemm_out,
        m,
        n,
        k,
        k_per_split,
        x.stride(0),
        weight.stride(0),
        n,
        BLOCK_M=block_m,
        BLOCK_N=cfg.block_n,
        BLOCK_K=cfg.block_k,
        num_warps=4,
        num_stages=cfg.num_stages,
    )
    if cfg.split_k > 1:
        reduce_block = 1024
        _split_k_reduce_kernel[(triton.cdiv(m * n, reduce_block),)](
            gemm_out, out, m * n, SPLIT_K=cfg.split_k, BLOCK=reduce_block
        )
    return out
