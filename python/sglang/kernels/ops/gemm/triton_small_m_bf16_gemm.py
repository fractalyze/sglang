"""Small-M GEMM ``y = (x @ w.T) * scale`` over an FP8 E4M3 vocab table on SM120.

Gemma-4's FP8 vocab table (SGLANG_OPT_GEMMA4_FP8_VOCAB_TABLE) stores the tied
embedding / LM head as E4M3 with a per-row (per-output-channel) scale. This
kernel streams each weight row block once with fp32 accumulation in K order and
no split-K, so one launch and no workspace: it upcasts each weight tile to bf16,
so the bf16 MMA and fp32 accumulation are those of a BF16 GEMM and only the
weight bytes halve.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

import torch
import triton
import triton.language as tl


class _TileConfig(NamedTuple):
    block_n: int
    block_k: int
    num_stages: int
    # Largest M routed to the tile; its shared memory must fit BLOCK_M = next_pow2(max_m).
    max_m: int


# FP8 E4M3 vocab heads with a tile measured on the RTX 5090; other shapes have none.
_FP8_HEAD_TUNED_SHAPES: Dict[Tuple[int, int], _TileConfig] = {
    # The Gemma-4-26B-A4B target's tied head: best worst case over M in 4..48 on the 5090
    # (443-525 us vs cuBLAS BF16 880-925 us; gemma4nv W12).
    (262144, 2816): _TileConfig(128, 128, 3, max_m=48),
}
_FP8_E4M3_MAX = 448.0


def quantize_fp8_weight_per_channel(
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Row-major ``weight`` [N, K] -> (E4M3 weight [N, K], fp32 scale [N]), absmax per row."""
    scale = weight.float().abs().amax(dim=1).clamp_min(1e-12) / _FP8_E4M3_MAX
    return (weight.float() / scale[:, None]).to(torch.float8_e4m3fn), scale


def quantize_fp8_weight_per_channel_chunked(
    weight: torch.Tensor, chunk_rows: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``quantize_fp8_weight_per_channel`` over ``chunk_rows`` rows at a time.

    Rows are independent, so the result is identical; peak memory holds one chunk in fp32.
    """
    rows = weight.shape[0]
    weight_fp8 = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    scale = torch.empty(rows, dtype=torch.float32, device=weight.device)
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        weight_fp8[start:end], scale[start:end] = quantize_fp8_weight_per_channel(
            weight[start:end]
        )
    return weight_fp8, scale


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


def use_fp8_vocab_head(n: int, k: int) -> bool:
    return (n, k) in _FP8_HEAD_TUNED_SHAPES


def fp8_vocab_head_max_m(n: int, k: int) -> int:
    return _FP8_HEAD_TUNED_SHAPES[(n, k)].max_m


def triton_small_m_fp8_vocab_head(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Logits ``(x @ weight.T) * scale`` for an E4M3 vocab head [V, K], M <= the shape's max_m."""
    n, k = weight.shape
    return _launch(x, weight, scale, _FP8_HEAD_TUNED_SHAPES[(n, k)])


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
