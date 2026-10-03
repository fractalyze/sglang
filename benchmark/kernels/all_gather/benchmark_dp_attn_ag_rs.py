"""Benchmark the DP-attention MoE gather / combine collectives.

Under DP attention with attn_tp_size == 1 (e.g. tp8 dp8), every layer gathers
each rank's hidden states into a global buffer before the MoE
(`_dp_gather_via_all_gather` -> `tp_group.all_gather_into_tensor`) and
reduce-scatters them back after (`dp_reduce_scatter_tensor` ->
`tp_group.reduce_scatter_tensor`). This script times exactly those two calls
inside a captured CUDA graph, at the global token counts the server runs, and
checks every result bit-for-bit against the expected gather / sum.

NCCL reads its environment once, at communicator init, so each NCCL setup is
its own process. `--config` picks the setup the server would build:

  default   the engine defaults (NCCL_CUMEM_ENABLE=0, NCCL_NVLS_ENABLE=0)
  nvls      `--enable-nccl-nvls` (NCCL_NVLS_ENABLE=1)
  symm      `--enable-symm-mem` (ncclMemAlloc buffers registered with the
            communicator, NCCL_CUMEM_ENABLE=1, NCCL_NVLS_ENABLE=1)

`--multimem` adds the in-tree multimem all-gather (`triton_symm_mem_ag`, a
one-shot NVLS store). There is no in-tree reduce-scatter kernel on CUDA, so
that row stays NCCL-only. Set NCCL_PROTO in the environment to pin a protocol.

Full sweep on one 8-GPU node:

  for cfg in default nvls symm; do
    for proto in LL LL128 Simple; do
      NCCL_PROTO=$proto torchrun --nproc_per_node 8 \
        benchmark/kernels/all_gather/benchmark_dp_attn_ag_rs.py \
        --config $cfg --out dp_attn_ag_rs.jsonl
    done
    torchrun --nproc_per_node 8 \
      benchmark/kernels/all_gather/benchmark_dp_attn_ag_rs.py \
      --config $cfg --multimem --out dp_attn_ag_rs.jsonl
  done
"""

import argparse
import json
import os
from typing import Callable, List

# The engine sets CUDA_DEVICE_MAX_CONNECTIONS=8 for every server.
_CONFIG_ENV = {
    "default": {"NCCL_CUMEM_ENABLE": "0", "NCCL_NVLS_ENABLE": "0"},
    "nvls": {"NCCL_CUMEM_ENABLE": "0", "NCCL_NVLS_ENABLE": "1"},
    "symm": {"NCCL_CUMEM_ENABLE": "1", "NCCL_NVLS_ENABLE": "1"},
}
_SERVER_ENV = {"CUDA_DEVICE_MAX_CONNECTIONS": "8"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", choices=sorted(_CONFIG_ENV), default="default")
    parser.add_argument(
        "--multimem",
        action="store_true",
        help="Also time the in-tree multimem all-gather.",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048, 4096],
        help="Global token counts (summed over DP ranks).",
    )
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument(
        "--nvlink-gbps",
        type=float,
        default=370.0,
        help=(
            "Bus bandwidth (GB/s) for the floor_us column. The default is the "
            "NCCL Simple-protocol figure on 8x H100 SXM (NVLink 4, NVSwitch); "
            "pass the figure for your machine."
        ),
    )
    parser.add_argument("--graph-loop", type=int, default=20)
    parser.add_argument("--test-loop", type=int, default=20)
    parser.add_argument("--out", help="Append one JSON line per row (rank 0).")
    return parser.parse_args()


# NCCL reads these at communicator init, before any import creates one.
ARGS = _parse_args()
os.environ.update(_SERVER_ENV)
os.environ.update(_CONFIG_ENV[ARGS.config])

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from sglang.srt.distributed import init_distributed_environment  # noqa: E402
from sglang.srt.distributed.device_communicators import (  # noqa: E402
    triton_symm_mem_ag,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (  # noqa: E402
    use_symmetric_memory,
)
from sglang.srt.distributed.parallel_state import (  # noqa: E402
    graph_capture,
    initialize_model_parallel,
)
from sglang.srt.runtime_context import get_parallel  # noqa: E402
from sglang.test.test_utils import publish_build_topology  # noqa: E402


def _rank_input(rank: int, rows: int, hidden: int, device) -> torch.Tensor:
    # Small integers keep a bf16 sum over 8 ranks exact, so RS checks bit-for-bit.
    gen = torch.Generator(device=device).manual_seed(rank)
    return torch.randint(1, 16, (rows, hidden), generator=gen, device=device).to(
        torch.bfloat16
    )


def _graph_latencies_us(
    *, fn: Callable[[], object], graph_loop: int, test_loop: int
) -> List[float]:
    """Per-call latency of each replay, sorted ascending."""
    with graph_capture() as ctx:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=ctx.stream):
            for _ in range(graph_loop):
                fn()
    graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    latencies: List[float] = []
    for _ in range(test_loop):
        dist.barrier()
        torch.cuda.synchronize()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        latencies.append(start.elapsed_time(end) * 1000 / graph_loop)
    graph.reset()
    return sorted(latencies)


def _row(
    op: str, impl: str, tokens: int, nbytes: int, latencies_us: List[float]
) -> dict:
    us = latencies_us[len(latencies_us) // 2]
    world = dist.get_world_size()
    # nccl-tests bus bandwidth: each rank moves (n - 1) / n of the full buffer.
    bus_bytes = nbytes * (world - 1) / world
    return {
        "config": ARGS.config,
        "nccl_proto": os.environ.get("NCCL_PROTO", "auto"),
        "op": op,
        "impl": impl,
        "global_tokens": tokens,
        "MiB": round(nbytes / 2**20, 2),
        "us": round(us, 1),
        "min_us": round(latencies_us[0], 1),
        "max_us": round(latencies_us[-1], 1),
        "bus_GBps": round(bus_bytes / us / 1e3, 1),
        "floor_us": round(bus_bytes / ARGS.nvlink_gbps / 1e3, 1),
    }


def _init_groups(*, symm: bool):
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", torch.cuda.current_device())
    init_distributed_environment(world_size=world, rank=rank, local_rank=device.index)
    publish_build_topology(world_rank=rank, tp_size=world, enable_symm_mem=symm)
    initialize_model_parallel(enable_symm_mem=symm)
    return get_parallel().tp_group, device


def _create_multimem_state(*, tp_group, device):
    # The kernel gathers along the last dim; viewing each rank's contiguous
    # [T, H] slab as one [1, T * H] row makes that the token-dim gather.
    max_local_tokens = max(ARGS.tokens) // tp_group.world_size
    return triton_symm_mem_ag.create_state(
        group=tp_group.device_group,
        rank_in_group=tp_group.rank_in_group,
        max_tokens=1,
        hidden_size=tp_group.world_size * max_local_tokens * ARGS.hidden_size,
        device=device,
    )


def _time_checked(
    *, fn, out: torch.Tensor, expected: torch.Tensor, what: str
) -> List[float]:
    latencies_us = _graph_latencies_us(
        fn=fn, graph_loop=ARGS.graph_loop, test_loop=ARGS.test_loop
    )
    assert torch.equal(out.view_as(expected), expected), f"{what} mismatch"
    return latencies_us


def _bench_tokens(*, tokens: int, tp_group, device, symm: bool, multimem_state):
    rank, world = tp_group.rank_in_group, tp_group.world_size
    hidden = ARGS.hidden_size
    assert tokens % world == 0, f"{tokens=} must split evenly over {world} ranks"
    local_tokens = tokens // world

    ag_inputs = [_rank_input(r, local_tokens, hidden, device) for r in range(world)]
    expected_ag = torch.cat(ag_inputs)
    rs_inputs = [_rank_input(r, tokens, hidden, device) for r in range(world)]
    expected_rs = torch.stack(rs_inputs).sum(0)[
        rank * local_tokens : (rank + 1) * local_tokens
    ]
    with use_symmetric_memory(tp_group, disabled=not symm):
        ag_in = ag_inputs[rank].clone()
        ag_out = torch.empty_like(expected_ag)
        rs_in = rs_inputs[rank].clone()
        rs_out = torch.empty_like(expected_rs)

    ag_latencies = _time_checked(
        fn=lambda: tp_group.all_gather_into_tensor(ag_out, ag_in),
        out=ag_out,
        expected=expected_ag,
        what=f"nccl all_gather at {tokens=}",
    )
    rs_latencies = _time_checked(
        fn=lambda: tp_group.reduce_scatter_tensor(rs_out, rs_in),
        out=rs_out,
        expected=expected_rs,
        what=f"nccl reduce_scatter at {tokens=}",
    )
    rows = [
        _row("all_gather", "nccl", tokens, expected_ag.nbytes, ag_latencies),
        _row("reduce_scatter", "nccl", tokens, rs_in.nbytes, rs_latencies),
    ]

    if multimem_state is not None:
        flat_in = ag_in.view(1, -1)
        # Kernel output is a view into the symmetric buffer: stable across calls.
        mm_out = multimem_state.comm_buff.view(-1)[: expected_ag.numel()]
        mm_latencies = _time_checked(
            fn=lambda: triton_symm_mem_ag.all_gather_inner(
                multimem_state,
                flat_in,
                tp_hidden_dim=world * flat_in.shape[1],
                safe=False,
            ),
            out=mm_out,
            expected=expected_ag,
            what=f"multimem all_gather at {tokens=}",
        )
        rows.append(
            _row("all_gather", "multimem", tokens, expected_ag.nbytes, mm_latencies)
        )
    return rows


def _report(rows: List[dict]) -> None:
    headers = list(rows[0])
    print("| " + " | ".join(headers) + " |")
    print("|" + " --- |" * len(headers))
    for row in rows:
        print("| " + " | ".join(str(row[h]) for h in headers) + " |")
    if ARGS.out:
        with open(ARGS.out, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


def main() -> None:
    symm = ARGS.config == "symm"
    tp_group, device = _init_groups(symm=symm)
    multimem_state = (
        _create_multimem_state(tp_group=tp_group, device=device)
        if ARGS.multimem
        else None
    )
    rows = []
    for tokens in ARGS.tokens:
        rows += _bench_tokens(
            tokens=tokens,
            tp_group=tp_group,
            device=device,
            symm=symm,
            multimem_state=multimem_state,
        )
    if tp_group.rank_in_group == 0:
        print("All results match the expected gather / sum.")
        _report(rows)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
