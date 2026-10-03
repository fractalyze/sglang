"""Split measured DeepSeek V3.2 decode steps into op classes and idle gaps.

Reads the torch-profiler decode traces SGLang writes under `--profile` (one
`*-DECODE.trace.json.gz` per rank), finds each `step[DECODE ...]` range, and
books every microsecond of it to exactly one bucket:

  * a kernel class (moe, dense, attention, routing, comm, small), where kernels
    on parallel streams split the overlapped time evenly, so the buckets add up
    to the step's wall time;
  * idle, when no kernel runs at all (HBM sits unused).

Then it sets each class beside its floor from `decode_floor`, splitting the
measured step into floor, kernel inefficiency and serialization. It takes the
same floor inputs as `decode_floor.py`; README.md has the command.
"""

import argparse
import gzip
import json
import os
import re
import statistics
from typing import Dict, Iterable, List, Sequence, Tuple

import decode_floor
import msgspec

# First match wins, so the norm kernels flashinfer builds with cutlass land in
# small and the FlashAttention cutlass kernel lands in attention, not dense.
_CLASS_PATTERNS: Sequence[Tuple[str, str]] = (
    ("comm", r"nccl|all_reduce|all_gather|reduce_scatter|multimem"),
    ("small", r"norm"),
    ("moe", r"marlin_moe|moe_sum_reduce|fused_moe|moe_wna16"),
    ("routing", r"deepseek_v3_topk|moe_align|count_and_sort|grouped_topk"),
    (
        "attention",
        r"flash|mqa_logits|topk_main|topk_persistent|prepare_varlen|"
        r"set_mla_kv|store_indexer|hadamard|_act_quant|dsa",
    ),
    ("dense", r"Marlin|nvjet|cublas|splitKreduce|gemm"),
)
_COMPILED = [(name, re.compile(pat)) for name, pat in _CLASS_PATTERNS]
CLASSES = tuple(name for name, _ in _CLASS_PATTERNS) + ("idle",)

_STEP_PREFIX = "step[DECODE"


def classify(kernel_name: str) -> str:
    for name, pattern in _COMPILED:
        if pattern.search(kernel_name):
            return name
    return "small"


class Kernel(msgspec.Struct, frozen=True):
    start_us: float
    end_us: float
    op_class: str


def _outer_ranges(ranges: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Ranges not nested in another: the profiler repeats a step's GPU range per stream."""
    outer: List[Tuple[float, float]] = []
    for start, end in sorted(ranges, key=lambda r: (r[0], -r[1])):
        if outer and end <= outer[-1][1]:
            continue
        outer.append((start, end))
    return outer


def split_step(kernels: Sequence[Kernel]) -> Dict[str, float]:
    """Microseconds per class over [first kernel start, last kernel end]."""
    edges = sorted({k.start_us for k in kernels} | {k.end_us for k in kernels})
    buckets = dict.fromkeys(CLASSES, 0.0)
    by_start = sorted(kernels, key=lambda k: k.start_us)
    active: List[Kernel] = []
    next_kernel = 0
    for lo, hi in zip(edges, edges[1:]):
        while next_kernel < len(by_start) and by_start[next_kernel].start_us <= lo:
            active.append(by_start[next_kernel])
            next_kernel += 1
        active = [k for k in active if k.end_us > lo]
        if not active:
            buckets["idle"] += hi - lo
            continue
        share = (hi - lo) / len(active)
        for k in active:
            buckets[k.op_class] += share
    return buckets


def load_steps(trace_path: str) -> List[Dict[str, float]]:
    """Per-class microseconds for every decode step in one rank's trace."""
    with gzip.open(trace_path) as f:
        events = json.load(f)["traceEvents"]
    steps = _outer_ranges(
        (e["ts"], e["ts"] + e["dur"])
        for e in events
        if e.get("cat") == "gpu_user_annotation" and e["name"].startswith(_STEP_PREFIX)
    )
    kernels = [
        Kernel(
            start_us=e["ts"],
            end_us=e["ts"] + e["dur"],
            op_class=classify(e["name"]),
        )
        for e in events
        if e.get("cat") == "kernel"
    ]
    out = []
    for start, end in steps:
        in_step = [k for k in kernels if start <= k.start_us < end]
        if in_step:
            out.append(split_step(in_step))
    return out


def mean_step_ms(steps: Sequence[Dict[str, float]]) -> Dict[str, float]:
    return {c: statistics.fmean(s[c] for s in steps) / 1e3 for c in CLASSES}


class GapSplit(msgspec.Struct, frozen=True, kw_only=True):
    """The measured step against its floor, in ms."""

    measured_ms: float
    floor_ms: Dict[str, float]  # per class; the classes with no HBM floor are 0
    inefficiency_ms: Dict[str, float]  # measured class time minus its floor
    idle_ms: float
    comm_symm_saving_ms: float  # NCCL time `--enable-symm-mem` removes
    baseline_ms: float  # the measured step with symm-mem collectives
    # Slice 1 ceiling: idle gaps, small ops and routing fused away entirely.
    slice1_ms: float
    # Slice 2 ceiling: each collective and each layer's attention excess hidden
    # behind next-GEMM weight prefetch, at most an L2 of weights per window.
    slice2_ms: float


def split_gap(
    measured: Dict[str, float],
    *,
    shape: decode_floor.ModelShape,
    point: decode_floor.DecodePoint,
    hbm_gbps: float,
    comm: decode_floor.CommTable,
    prefetch_bytes: int = decode_floor.H100_L2_BYTES,
) -> GapSplit:
    nbytes = decode_floor.step_bytes(shape, point)
    to_ms = 1.0 / (hbm_gbps * 1e6)
    comm_windows_us = decode_floor.comm_windows_us(shape, point, comm)
    comm_floor = sum(comm_windows_us) / 1e3
    # Routing and small ops move only activations, a few MB per step.
    floor = {
        "moe": nbytes.routed_experts * to_ms,
        "dense": nbytes.dense_gemm * to_ms,
        "attention": nbytes.kv * to_ms,
        "routing": 0.0,
        "comm": comm_floor,
        "small": 0.0,
    }
    inefficiency = {c: measured[c] - floor[c] for c in floor}
    # Attention at or under its floor leaves nothing for prefetch to hide.
    attention_excess_us = max(0.0, 1e3 * inefficiency["attention"] / shape.num_layers)
    windows_us = comm_windows_us + [attention_excess_us] * shape.num_layers
    measured_ms = sum(measured.values())
    return GapSplit(
        measured_ms=measured_ms,
        floor_ms=floor,
        inefficiency_ms=inefficiency,
        idle_ms=measured["idle"],
        comm_symm_saving_ms=inefficiency["comm"],
        baseline_ms=measured_ms - inefficiency["comm"],
        slice1_ms=measured["idle"] + measured["small"] + measured["routing"],
        slice2_ms=decode_floor.hidden_behind_prefetch_us(
            windows_us, prefetch_bytes=prefetch_bytes, hbm_gbps=hbm_gbps
        )
        / 1e3,
    )


def _report(measured: Dict[str, float], gap: GapSplit, n_steps: int) -> None:
    print(f"{n_steps} steps, mean step {gap.measured_ms:.2f} ms")
    print(f"{'class':>10} {'measured':>9} {'floor':>7} {'excess':>7} {'share':>6}")
    for c in CLASSES:
        if c == "idle":
            floor_s, excess = "-", measured[c]
        else:
            floor_s, excess = f"{gap.floor_ms[c]:7.2f}", gap.inefficiency_ms[c]
        share = 100 * measured[c] / gap.measured_ms
        print(f"{c:>10} {measured[c]:9.2f} {floor_s:>7} {excess:7.2f} {share:5.1f}%")
    base = gap.baseline_ms
    print(f"with symm-mem collectives: {base:.2f} ms")
    for name, ms in (("slice 1", gap.slice1_ms), ("slice 2", gap.slice2_ms)):
        print(f"{name} ceiling: {ms:.2f} ms ({100 * ms / base:.1f}% of it)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("trace_dir")
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--json", action="store_true")
    decode_floor.add_floor_args(parser)
    args = parser.parse_args()

    paths = sorted(
        os.path.join(args.trace_dir, name)
        for name in os.listdir(args.trace_dir)
        if name.endswith("-DECODE.trace.json.gz")
    )
    steps = [s for p in paths for s in load_steps(p)]
    measured = mean_step_ms(steps)
    gap = split_gap(
        measured,
        shape=decode_floor.shape_from_args(args),
        point=decode_floor.point_from_args(args, args.concurrency),
        hbm_gbps=args.hbm_gbps,
        comm=decode_floor.load_comm_table(args.comm_jsonl),
    )
    if args.json:
        print(msgspec.json.encode({"measured_ms": measured, "gap": gap}).decode())
    else:
        _report(measured, gap, len(steps))


if __name__ == "__main__":
    main()
