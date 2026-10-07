from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    cuda_stubs_dir,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# W4A16 GEMM for M <= 1024 over Marlin-repacked AWQ weights (see
# csrc/gemm/w4a16_sm90.cuh). It reads the same qweight / scales / qzeros
# tensors AWQMarlinLinearKernel prepares, so it needs no load-time work of its own.
# The bound covers decode batches gathered across DP-attention ranks; larger M
# (prefill) stays on Marlin.

GROUP_SIZE = 128
MAX_M = 1024
_TILE_N = 64
# wgmma N per launch. M up to the widest rounds up to the next one and runs as
# one token block; larger M runs in several blocks of one tile.
_TOKEN_TILES = (8, 16, 32, 48, 64, 96, 128, 192)
_SMEM_LIMIT = 227 * 1024
# Upper bound on the persistent grid (SMs x resident CTAs per SM); the launcher
# checks it.
_MAX_CTAS = 2048


def supports_w4a16_sm90(
    m: int, n: int, k: int, group_size: int, dtype: torch.dtype
) -> bool:
    return (
        1 <= m <= MAX_M
        and dtype == torch.bfloat16
        and group_size == GROUP_SIZE
        and k % GROUP_SIZE == 0
        # One group makes marlin_permute_scales use its channelwise layout.
        and k >= 2 * GROUP_SIZE
        and n % _TILE_N == 0
    )


def _fits_registers(token_tile: int, tiles: int, ping: int) -> bool:
    # Mirrors the register static_assert in w4a16_sm90::Trait.
    threads = tiles * ping * 128 + 32
    return token_tile // 2 + 64 <= 65536 // threads


def _smem_bytes(
    token_tile: int, tiles: int, ping: int, cluster_k: int, stages: int
) -> int:
    # Mirrors w4a16_sm90::Trait.
    cta_n = _TILE_N * tiles
    stage = 16 * token_tile * 16 + 8 * tiles * 512 + cta_n * 2 + cta_n // 2
    stage = (stage + 1023) // 1024 * 1024
    handoff = tiles * token_tile // 2 * 128 * 4 if ping > 1 else 0
    inbox = cta_n * token_tile * 4 if cluster_k > 1 else 0
    return 1024 + stages * stage + handoff + inbox


@cache_once
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@cache_once
def _launch_config(m: int, n: int, k: int, device_index: int) -> tuple:
    """(token_tile, tiles, ping, cluster_k, stages, min_groups_per_cta) for one shape.

    cluster_k 0 is stream-K, 1 one CTA per unit, above 1 cluster split-K. Fit to
    H100 sweeps over DeepSeek V3.2's dense AWQ shapes; re-tune when the kernel
    changes.
    """
    if m > _TOKEN_TILES[-1]:
        return _large_m_launch_config(m, n, k)
    token_tile = next(t for t in _TOKEN_TILES if t >= m)
    num_groups = k // GROUP_SIZE
    n_tiles = n // _TILE_N
    num_sms = _num_sms(device_index)
    min_groups = 0
    if num_groups <= 16:
        # Short K: splitting costs more than it balances, except for the
        # latency-bound smallest token tiles.
        if token_tile <= 16 and num_groups >= 2:
            tiles, ping, cluster_k = 1, 2 if n_tiles < num_sms else 1, 2
        else:
            tiles, ping, cluster_k = 1, 2 if n_tiles <= 2 * num_sms else 1, 1
    elif n_tiles <= 8:
        # Few column tiles: split K across a full cluster.
        tiles, ping, cluster_k = 1, 2 if token_tile <= 96 else 1, min(8, num_groups)
    elif num_groups >= 64:
        # Long K: stream-K over wide units, or a 4-way cluster for the
        # latency-bound smallest token tiles.
        if token_tile <= 16:
            tiles, ping, cluster_k = 4, 1, 4
        else:
            tiles, ping, cluster_k, min_groups = 4 if token_tile <= 64 else 2, 1, 0, 2
    elif token_tile <= 64:
        tiles, ping, cluster_k = 1, 1, 8
    else:
        tiles, ping, cluster_k = 1, 2 if token_tile <= 96 else 1, 0
        min_groups = max(2, num_groups // 8) if token_tile <= 96 else 2

    while n % (_TILE_N * tiles):
        tiles //= 2
    if not _fits_registers(token_tile, tiles, ping):
        ping = 1
    while not _fits_registers(token_tile, tiles, ping):
        tiles //= 2

    many_ctas = cluster_k != 0 and n_tiles // tiles * cluster_k >= num_sms
    stages = 2 if token_tile >= 48 and many_ctas else 4
    if _smem_bytes(token_tile, tiles, ping, cluster_k, stages) > _SMEM_LIMIT:
        stages = 2
    return token_tile, tiles, ping, cluster_k, stages, min_groups


def _large_m_launch_config(m: int, n: int, k: int) -> tuple:
    """_launch_config for M above one token block, run as several token blocks.

    Fit to H100 sweeps over DeepSeek V3.2's MLP shapes at M = 256 to 1024.
    """
    if n // _TILE_N <= 8:
        # Few column tiles: narrow token blocks give stream-K enough units.
        token_tile, cluster_k, min_groups = 64, 0, 2
    else:
        # The tile that pads M least; on a tie, the wider one, which dequantizes
        # each weight fewer times.
        token_tile = min((128, 192), key=lambda t: (-m % t, -t))
        # Short K: a unit's few groups are not worth splitting.
        cluster_k, min_groups = (1, 0) if k // GROUP_SIZE <= 16 else (0, 2)
    tiles, ping = 2, 1
    while n % (_TILE_N * tiles):
        tiles //= 2
    stages = 4
    while _smem_bytes(token_tile, tiles, ping, cluster_k, stages) > _SMEM_LIMIT:
        stages -= 1
    return token_tile, tiles, ping, cluster_k, stages, min_groups


_flags: dict = {}


def _flags_for(device: torch.device) -> torch.Tensor:
    # Zero between launches (each launch resets what it set); one buffer per
    # stream, since launches on different streams may overlap. A buffer first
    # needed inside graph capture is zeroed only when that graph replays, so it
    # stays the graph's own rather than being shared with later launches.
    stream = torch.cuda.current_stream(device)
    key = (device.index, stream.cuda_stream)
    flags = _flags.get(key)
    if flags is None:
        flags = torch.zeros(_MAX_CTAS, dtype=torch.int32, device=device)
        if not torch.cuda.is_current_stream_capturing():
            _flags[key] = flags
    return flags


@cache_once
def _jit_w4a16_sm90_module(
    token_tile: int, tiles: int, ping: int, cluster_k: int, stages: int
) -> Module:
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("w4a16_sm90_gemm requires SM90 (Hopper)")
    args = make_cpp_args(
        token_tile, tiles, ping, cluster_k, stages, is_arch_support_pdl()
    )
    return load_jit(
        "w4a16_sm90",
        *args,
        cuda_files=["gemm/w4a16_sm90.cuh"],
        cuda_wrappers=[("run", f"w4a16_sm90_gemm<{args}>")],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=[f"-L{cuda_stubs_dir()}", "-lcuda"],
        extra_dependencies=["cutlass"],
    )


def w4a16_sm90_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    b_zeros: torch.Tensor,
    size_n: int,
) -> torch.Tensor:
    """out[M, N] = a[M, K] @ dequant(w)[K, N] for Marlin-layout AWQ g128 weights.

    a: [M, K] bf16 with unit inner stride and 16-byte aligned rows; b_q_weight:
    awq_marlin_repack output; b_scales: marlin_permute_scales output; b_zeros:
    awq_to_marlin_zero_points output. Callers gate on `supports_w4a16_sm90`.
    """
    m, k = a.shape
    out = torch.empty((m, size_n), dtype=a.dtype, device=a.device)
    *config, min_groups = _launch_config(m, size_n, k, a.device.index)
    _jit_w4a16_sm90_module(*config).run(
        out, a, b_q_weight, b_scales, b_zeros, _flags_for(a.device), min_groups
    )
    return out
