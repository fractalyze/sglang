"""Small-M BF16 GEMM ``y = x @ w.T`` for decode batches on SM120.

On the RTX 5090 cuBLAS (and cuBLASLt's heuristics) run small-M BF16 GEMMs on
the SM80 WMMA fallback ``cutlass_80_wmma_tensorop_bf16_s161616gemm``, which
reaches 0.55-0.65 of DRAM bandwidth at M=8-16 on Gemma-4's o_proj and dense
MLP shapes. This kernel streams each weight row block once with fp32
accumulation in K order and no split-K, so one launch and no workspace.

The same kernel also reads FP8 E4M3 weights with a per-output-channel scale
(weight-only quantization): it upcasts each weight tile to bf16, so the bf16
MMA and fp32 accumulation are unchanged and only the weight bytes halve.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

import torch
import triton
import triton.language as tl

# Decode batches only; prefill keeps cuBLAS, which wins at M=1024.
MAX_M = 32


class _TileConfig(NamedTuple):
    block_n: int
    block_k: int
    num_stages: int


# Measured on RTX 5090 at M in {1, 8, 16, 32} (gemma4nv T3 microbench).
# Shapes outside this allowlist keep cuBLAS.
_TUNED_SHAPES: Dict[Tuple[int, int], _TileConfig] = {
    # Gemma-4-26B-A4B o_proj: sliding (16 x 256 heads) and full (16 x 512 heads).
    (2816, 4096): _TileConfig(32, 256, 3),
    (2816, 8192): _TileConfig(32, 256, 3),
    # Gemma-4-26B-A4B dense MLP: gate_up (2 x 2112) and down.
    (4224, 2816): _TileConfig(32, 256, 4),
    (2816, 2112): _TileConfig(32, 128, 4),
}


# FP8 E4M3 weight-only shapes; quantizing changes numerics, so this is a separate opt-in.
_FP8_WEIGHT_TUNED_SHAPES: Dict[Tuple[int, int], _TileConfig] = {
    # Gemma-4-26B-A4B o_proj: sliding and full.
    (2816, 4096): _TileConfig(32, 256, 4),
    (2816, 8192): _TileConfig(32, 256, 4),
}
_FP8_E4M3_MAX = 448.0


def use_triton_small_m_bf16_gemm(m: int, n: int, k: int) -> bool:
    return m <= MAX_M and (n, k) in _TUNED_SHAPES


def use_fp8_weight_only(n: int, k: int) -> bool:
    return (n, k) in _FP8_WEIGHT_TUNED_SHAPES


def quantize_fp8_weight_per_channel(
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Row-major ``weight`` [N, K] -> (E4M3 weight [N, K], fp32 scale [N]), absmax per row."""
    scale = weight.float().abs().amax(dim=1).clamp_min(1e-12) / _FP8_E4M3_MAX
    return (weight.float() / scale[:, None]).to(torch.float8_e4m3fn), scale


@triton.jit
def _small_m_bf16_gemm_kernel(
    x_ptr,
    w_ptr,
    scale_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    """out[m, n] = sum over k of x[m, k] * w[n, k] (* scale[n] when HAS_SCALE)."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
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
        acc = tl.dot(x, tl.trans(w.to(x.dtype)), acc)
    if HAS_SCALE:
        acc *= tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def triton_small_m_bf16_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``F.linear(x, weight)`` for a 2D bf16 ``x`` [M, K] and row-major ``weight`` [N, K]."""
    n, k = weight.shape
    return _launch(x, weight, None, _TUNED_SHAPES[(n, k)])


def triton_small_m_fp8_weight_gemm(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """``(x @ weight.T) * scale`` for an E4M3 ``weight`` [N, K] and fp32 ``scale`` [N], M <= MAX_M."""
    n, k = weight.shape
    return _launch(x, weight, scale, _FP8_WEIGHT_TUNED_SHAPES[(n, k)])


def _launch(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: Optional[torch.Tensor],
    cfg: _TileConfig,
) -> torch.Tensor:
    m, k = x.shape
    n = weight.shape[0]
    # tl.dot needs at least 16 rows; padding rows are masked loads of zero.
    block_m = max(16, triton.next_power_of_2(m))
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    grid = (triton.cdiv(n, cfg.block_n), triton.cdiv(m, block_m))
    _small_m_bf16_gemm_kernel[grid](
        x,
        weight,
        scale,
        out,
        m,
        n,
        k,
        x.stride(0),
        weight.stride(0),
        out.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=cfg.block_n,
        BLOCK_K=cfg.block_k,
        HAS_SCALE=scale is not None,
        num_warps=4,
        num_stages=cfg.num_stages,
    )
    return out
