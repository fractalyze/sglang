"""Hopper grouped W4A16 MoE GEMM (see csrc/gemm/w4a16_moe_sm90/).

The kernel runs swap-AB: the dequantised 4-bit expert weights are the wgmma A
operand, fed from registers in 64-row atoms, and the expert's
routed tokens are the N operand, gathered into shared memory. A few tokens per
expert therefore cost no M padding, and the same kernel covers prefill-sized
token blocks by widening N.

Weights are repacked once at load time (``repack_awq_moe_weights``) into
contiguous per-stage blocks, so the kernel streams them with bulk async copies
and each thread reads its wgmma A fragments with one 16-byte shared-memory load.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Must match csrc/gemm/w4a16_moe_sm90/kernel.cuh.
TILE_N = 128  # rows per weight tile, one consumer warpgroup's: two 64-row wgmma atoms
TILE_K = 128  # reduction depth per pipeline stage, one AWQ group
CTA_ROWS = 2 * TILE_N  # output rows per CTA: one weight tile per consumer warpgroup
GROUP_SIZE = 128
# Ordered [64-row atom][k64 half][thread][k16 slice] so a thread's four k16
# fragments of one half are a single 16-byte load.
WORDS_PER_STAGE = TILE_N * TILE_K // 8

# AutoAWQ packs 8 nibbles per int32 in this interleaved column order.
_AWQ_REVERSE_PACK_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)
# Nibble slot of fragment value v, chosen so that marlin's
# dequant<nv_bfloat162, kU4>(q) yields registers (v0, v1), (v2, v3) and
# dequant(q >> 8) yields (v4, v5), (v6, v7).
_FRAGMENT_NIBBLE_SLOT = (0, 4, 1, 5, 2, 6, 3, 7)


class W4A16MoeWeights(NamedTuple):
    """One expert-stacked projection in the kernel's layout.

    qweight: int32 [E, N / TILE_N, K / TILE_K, WORDS_PER_STAGE]
    scales:  activation dtype [E, N / TILE_N, K / TILE_K, TILE_N]
    zeros:   uint8 [E, N / TILE_N, K / TILE_K, TILE_N], codes in [0, 15]
    """

    qweight: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor


def unpack_awq_codes(packed: torch.Tensor) -> torch.Tensor:
    """AWQ int32 [..., R, C / 8] packed along the last dim -> uint8 codes [..., R, C]."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
    codes = (packed.unsqueeze(-1) >> shifts) & 0xF
    reverse = torch.tensor(_AWQ_REVERSE_PACK_ORDER, device=packed.device)
    codes = codes[..., reverse]
    return codes.reshape(*packed.shape[:-1], -1).to(torch.uint8)


def _fragment_coordinates(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """(row, col) inside one TILE_N x TILE_K stage for every packed nibble.

    Both are int64 [WORDS_PER_STAGE, 8], indexed by (word, fragment value).
    The thread -> (row, col) map is the wgmma m64k16 register A layout, which
    per warp equals the mma.m16n8k16 A layout:
      row = 16 * warp + lane / 4 + 8 * v[1]
      col = 2 * (lane % 4) + v[0] + 8 * v[2]
    """
    atom, half, tid, step, v = torch.meshgrid(
        torch.arange(2, device=device),
        torch.arange(2, device=device),
        torch.arange(128, device=device),
        torch.arange(4, device=device),
        torch.arange(8, device=device),
        indexing="ij",
    )
    warp, lane = tid // 32, tid % 32
    row = 64 * atom + 16 * warp + lane // 4 + 8 * ((v >> 1) & 1)
    col = 16 * (4 * half + step) + 2 * (lane % 4) + (v & 1) + 8 * ((v >> 2) & 1)
    return row.reshape(-1, 8), col.reshape(-1, 8)


def pack_codes(codes: torch.Tensor) -> torch.Tensor:
    """uint8 codes [E, N, K] (output rows x reduction) -> qweight in kernel layout."""
    e, n, k = codes.shape
    assert n % TILE_N == 0 and k % TILE_K == 0, (n, k)
    row, col = _fragment_coordinates(codes.device)
    tiles = codes.view(e, n // TILE_N, TILE_N, k // TILE_K, TILE_K).permute(
        0, 1, 3, 2, 4
    )
    values = tiles[..., row, col].to(torch.int64)  # [E, Nt, Kt, words, 8]
    slots = torch.tensor(_FRAGMENT_NIBBLE_SLOT, device=codes.device)
    # Disjoint nibbles, so the sum is the bitwise OR.
    words = (values << (4 * slots)).sum(dim=-1)
    # Reinterpret the unsigned 32-bit pattern as int32.
    words = torch.where(words >= 2**31, words - 2**32, words)
    return words.to(torch.int32).contiguous()


def _tile_per_row(t: torch.Tensor) -> torch.Tensor:
    """[E, K / G, N] -> [E, N / TILE_N, K / TILE_K, TILE_N]."""
    e, kg, n = t.shape
    return t.view(e, kg, n // TILE_N, TILE_N).permute(0, 2, 1, 3).contiguous()


def repack_awq_moe_weights(
    qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor, group_size: int
) -> W4A16MoeWeights:
    """AWQ expert weights -> kernel layout.

    qweight: int32 [E, K, N / 8], scales: [E, K / G, N], qzeros: int32 [E, K / G, N / 8].
    """
    if group_size != GROUP_SIZE:
        raise ValueError(
            f"w4a16_moe_sm90 needs AWQ group size {GROUP_SIZE}, got {group_size}"
        )
    e, k, packed_n = qweight.shape
    n = packed_n * 8
    # A CTA covers two weight tiles; checked here so a bad shape fails at load.
    if n % CTA_ROWS != 0 or k % TILE_K != 0:
        raise ValueError(
            f"w4a16_moe_sm90 needs N divisible by {CTA_ROWS} and K by {TILE_K}, "
            f"got N={n}, K={k}"
        )
    packed = torch.empty(
        (e, n // TILE_N, k // TILE_K, WORDS_PER_STAGE),
        dtype=torch.int32,
        device=qweight.device,
    )
    # Per expert, so the unpacked intermediates stay one expert large.
    for expert in range(e):
        codes = unpack_awq_codes(qweight[expert]).t().contiguous()  # [N, K]
        packed[expert] = pack_codes(codes.unsqueeze(0))[0]
    return W4A16MoeWeights(
        qweight=packed,
        scales=_tile_per_row(scales),
        zeros=_tile_per_row(unpack_awq_codes(qzeros)),
    )


def dequantize_reference(weights: W4A16MoeWeights) -> torch.Tensor:
    """Kernel-layout weights -> dense [E, N, K] in the scales' dtype.

    Computes (q - z) * s in the activation dtype, the rounding the kernel uses.
    """
    qweight, scales, zeros = weights
    e, nt, kt, _ = qweight.shape
    row, col = _fragment_coordinates(qweight.device)
    slots = torch.tensor(_FRAGMENT_NIBBLE_SLOT, device=qweight.device)
    values = (qweight.unsqueeze(-1) >> (4 * slots)) & 0xF  # [E, Nt, Kt, words, 8]
    codes = torch.empty(
        e, nt, kt, TILE_N, TILE_K, dtype=torch.uint8, device=qweight.device
    )
    codes[..., row, col] = values.to(torch.uint8)
    z = zeros.unsqueeze(-1).to(scales.dtype)
    s = scales.unsqueeze(-1)
    dense = (codes.to(scales.dtype) - z) * s  # [E, Nt, Kt, TILE_N, TILE_K]
    return dense.permute(0, 1, 3, 2, 4).reshape(e, nt * TILE_N, kt * TILE_K)


# Token-block widths the kernel is instantiated for (wgmma N, a multiple of 8).
# No 128-token block: it spills registers and fits only four pipeline stages.
TOKEN_BLOCKS = (8, 16, 32, 64)


def select_token_block(num_tokens: int, top_k: int, num_experts: int) -> int:
    """Smallest block that holds most experts' routed tokens.

    Per-expert counts are roughly binomial; a block below mean + 2.5 sd sends
    many experts into a second block that streams their weights again, while a
    wider one makes every wgmma wider. The margin minimises gate-up plus down
    time in an H100 sweep of 4 to 64 tokens per expert; re-sweep when the
    kernel changes.
    """
    mean = num_tokens * top_k / num_experts
    target = mean + 2.5 * math.sqrt(mean)
    for block in TOKEN_BLOCKS:
        if target <= block:
            return block
    return TOKEN_BLOCKS[-1]


@cache_once
def _jit_w4a16_moe_sm90_module(
    token_block: int, zero_unrouted: bool, half_tile_tail: bool
) -> Module:
    if torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("w4a16_moe_sm90 requires an SM90 (Hopper) GPU")
    args = make_cpp_args(token_block, zero_unrouted, half_tile_tail)
    return load_jit(
        "w4a16_moe_sm90",
        *args,
        cuda_files=["gemm/w4a16_moe_sm90/kernel.cuh"],
        cuda_wrappers=[("run", f"sglang::W4A16MoeSm90Kernel<{args}>::run")],
        extra_cuda_cflags=[
            "-O3",
            "-DNDEBUG",
            "-DCUTE_USE_PACKED_TUPLE=1",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        ],
        extra_dependencies=["cutlass"],
    )


@debug_kernel_api
def w4a16_moe_sm90_gemm(
    a: torch.Tensor,
    out: torch.Tensor,
    weights: W4A16MoeWeights,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor | None,
    token_block: int,
    a_row_divisor: int,
    zero_unrouted: bool = False,
    half_tile_tail: bool = False,
) -> None:
    """out[r] = W[expert(r)] @ a[r / a_row_divisor] for every routed row r.

    Routed rows r are the flat (token * top_k + slot) indices that
    ``moe_align_block_size(..., token_block, ...)`` lists in sorted_token_ids;
    padding entries (r >= out.shape[0]) are skipped, and rows no block lists
    are left untouched. With topk_weights, row r is scaled by
    topk_weights.view(-1)[r]. ``a_row_divisor`` is top_k when ``a`` holds one
    row per token, 1 when it holds one row per routed row.

    With ``zero_unrouted``, rows of blocks routed to no local expert (expert id
    -1) are written as zeros instead of skipped, so ``out`` needs no zero fill
    when every row is listed.

    With ``half_tile_tail``, the tiles past the last full wave of SMs run as
    128-row halves, so a launch whose tile count just passes a multiple of the
    SM count finishes half a tile sooner; the output is bitwise unchanged.
    """
    qweight, scales, zeros = weights
    if a.dtype != torch.bfloat16 or scales.dtype != torch.bfloat16:
        raise TypeError("w4a16_moe_sm90 supports bfloat16 activations and scales")
    assert a.is_contiguous() and out.is_contiguous()
    assert a.shape[1] == qweight.shape[2] * TILE_K, (a.shape, qweight.shape)
    assert out.shape[1] == qweight.shape[1] * TILE_N, (out.shape, qweight.shape)
    assert out.shape[1] % CTA_ROWS == 0, out.shape
    assert token_block in TOKEN_BLOCKS, token_block
    assert sorted_token_ids.dtype == torch.int32 and expert_ids.dtype == torch.int32
    if topk_weights is not None:
        assert topk_weights.dtype == torch.float32 and topk_weights.is_contiguous()
        assert topk_weights.numel() == out.shape[0]
    module = _jit_w4a16_moe_sm90_module(token_block, zero_unrouted, half_tile_tail)
    module.run(
        a,
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
