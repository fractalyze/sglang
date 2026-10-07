"""Count which kernel serves each dense AWQ GEMM in profiled DeepSeek V3.2 decode steps.

With `SGLANG_USE_W4A16_SM90_GEMM=1`, `AWQMarlinLinearKernel` runs a dense AWQ
projection on the W4A16 SM90 GEMM when its M is at most 192 and on Marlin
otherwise. Under DP attention the attention projections see the rank's own
tokens, while the MLP side (the shared expert, or the dense MLP of the first
layers) runs after the all-gather on every rank's tokens. This script reads the
torch-profiler decode traces SGLang writes under `/start_profile` (one
`*-DECODE.trace.json.gz` per rank) and counts, per projection, how many calls
each kernel served.

A layer's dense GEMMs are told apart by position. A decode step runs one
all-gather and one reduce-scatter per layer: the AWQ GEMMs before the
all-gather are the attention projections, the ones between it and the
reduce-scatter are the MLP. Within a side, the n-th AWQ GEMM is the n-th
projection that side runs. That holds while each projection is one kernel; Marlin
can split a large M into several launches, which shows up as `<side> #n` rows.

    python benchmark/dsv32_stack/dense_dispatch.py <trace dir> [--json]
    python benchmark/dsv32_stack/dense_dispatch.py <trace dir> --stage VERIFY --tokens-per-request 3
"""

import argparse
import collections
import gzip
import json
import os
import re
import statistics
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import msgspec

# The AWQ projections each side runs, in order: the MLA and indexer projections
# before the all-gather, the shared expert (or dense MLP) between it and the
# reduce-scatter.
ATTENTION = ("q_a+kv_a", "q_b", "indexer wq_b", "indexer wk", "o_proj")
MLP = ("gate_up", "down")

_SM90 = re.compile(r"w4a16_sm90_kernel")
_MARLIN = re.compile(r"marlin::Marlin<")
_ALL_GATHER = re.compile(r"AllGather|all_gather")
_REDUCE_SCATTER = re.compile(r"ReduceScatter|reduce_scatter")
_STEP = re.compile(r"^step\[(DECODE|VERIFY) bs=(\d+)\]")


def kernel_family(name: str) -> Optional[str]:
    """`sm90` or `marlin` for a dense AWQ GEMM kernel, None for anything else."""
    if _SM90.search(name) and "moe" not in name:
        return "sm90"
    if _MARLIN.search(name):
        return "marlin"
    return None


class Call(msgspec.Struct, frozen=True):
    side: str  # "attention" or "mlp"
    index: int  # position within the side's AWQ GEMMs in this layer
    family: str  # "sm90" or "marlin"
    dur_us: float


class Step(msgspec.Struct, frozen=True):
    stage: str  # DECODE, or VERIFY for a speculative step's target forward
    bs: int  # the rank's own requests, from the step annotation
    layers: int  # all-gathers seen in the step
    calls: List[Call]


def split_layers(kernels: Sequence[Tuple[str, float]]) -> Tuple[int, List[Call]]:
    """(layers, calls) for one step's kernels, given as (name, duration) in launch order."""
    calls: List[Call] = []
    layers = 0
    side, index = "attention", 0
    for name, dur in kernels:
        if _ALL_GATHER.search(name):
            layers += 1
            side, index = "mlp", 0
            continue
        if _REDUCE_SCATTER.search(name):
            side, index = "attention", 0
            continue
        family = kernel_family(name)
        if family is not None:
            calls.append(Call(side=side, index=index, family=family, dur_us=dur))
            index += 1
    return layers, calls


def _outer_ranges(
    ranges: Iterable[Tuple[float, float, str]],
) -> List[Tuple[float, float, str]]:
    """Ranges not nested in another: the profiler repeats a step's GPU range per stream."""
    outer: List[Tuple[float, float, str]] = []
    for start, end, name in sorted(ranges, key=lambda r: (r[0], -r[1])):
        if outer and end <= outer[-1][1]:
            continue
        outer.append((start, end, name))
    return outer


def load_steps(trace_path: str) -> List[Step]:
    with gzip.open(trace_path) as f:
        events = json.load(f)["traceEvents"]
    ranges = _outer_ranges(
        (e["ts"], e["ts"] + e["dur"], e["name"])
        for e in events
        if e.get("cat") == "gpu_user_annotation" and _STEP.match(e["name"])
    )
    kernels = sorted(
        (e["ts"], e["name"], e["dur"]) for e in events if e.get("cat") == "kernel"
    )
    steps = []
    for start, end, name in ranges:
        in_step = [(n, d) for ts, n, d in kernels if start <= ts < end]
        layers, calls = split_layers(in_step)
        match = _STEP.match(name)
        steps.append(
            Step(
                stage=match.group(1),
                bs=int(match.group(2)),
                layers=layers,
                calls=calls,
            )
        )
    return steps


def projection(side: str, index: int) -> str:
    names = ATTENTION if side == "attention" else MLP
    return names[index] if index < len(names) else f"{side} #{index}"


class Row(msgspec.Struct, frozen=True):
    projection: str
    side: str
    m: str  # the M the projection runs at: the rank's batch, or the gathered batch
    sm90_calls: int
    marlin_calls: int
    sm90_us: float  # mean per call, 0 when none
    marlin_us: float


def tally(
    ranks: Dict[str, List[Step]], *, dp_size: int, tokens_per_request: int = 1
) -> List[Row]:
    """One row per (projection, M), summed over every rank and step.

    M on the attention side is the rank's own tokens: its requests times
    `tokens_per_request` (the draft tokens plus one on a VERIFY step). On the MLP
    side it is that times `dp_size`: a CUDA-graph step gathers in DP attention's
    max-len padding mode, which pads every rank to the largest batch.
    """
    durs: Dict[Tuple[str, str, int, str], List[float]] = collections.defaultdict(list)
    for steps in ranks.values():
        for step in steps:
            for call in step.calls:
                m = step.bs * tokens_per_request
                if call.side == "mlp":
                    m *= dp_size
                durs[
                    (call.side, projection(call.side, call.index), m, call.family)
                ].append(call.dur_us)
    rows = []
    for side, name, m in sorted({k[:3] for k in durs}):
        sm90 = durs.get((side, name, m, "sm90"), [])
        marlin = durs.get((side, name, m, "marlin"), [])
        rows.append(
            Row(
                projection=name,
                side=side,
                m=f"{m} ({'own' if side == 'attention' else 'gathered'})",
                sm90_calls=len(sm90),
                marlin_calls=len(marlin),
                sm90_us=statistics.fmean(sm90) if sm90 else 0.0,
                marlin_us=statistics.fmean(marlin) if marlin else 0.0,
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("trace_dir")
    parser.add_argument("--dp-size", type=int, default=8)
    parser.add_argument("--stage", choices=("DECODE", "VERIFY"), default="DECODE")
    # The tokens a step runs per request: 1 for DECODE, draft tokens plus one for VERIFY.
    parser.add_argument("--tokens-per-request", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    ranks = {
        name: [
            s
            for s in load_steps(os.path.join(args.trace_dir, name))
            if s.stage == args.stage
        ]
        for name in sorted(os.listdir(args.trace_dir))
        if name.endswith("-DECODE.trace.json.gz")
    }
    rows = tally(
        ranks, dp_size=args.dp_size, tokens_per_request=args.tokens_per_request
    )
    if args.json:
        print(msgspec.json.encode(rows).decode())
        return
    layers = sorted({s.layers for steps in ranks.values() for s in steps})
    n_steps = sum(len(s) for s in ranks.values())
    print(f"{len(ranks)} ranks, {n_steps} steps, all-gathers per step: {layers}")
    print(
        "| projection | M | W4A16 SM90 calls | Marlin calls | SM90 µs/call | Marlin µs/call |"
    )
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r.projection} | {r.m} | {r.sm90_calls} | {r.marlin_calls} | "
            f"{r.sm90_us:.1f} | {r.marlin_us:.1f} |"
        )


if __name__ == "__main__":
    main()
