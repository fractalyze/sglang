"""Per-round breakdown of the MTP draft loop and verify from a spec_probe `profile` trace.

Reads the DECODE torch-profiler trace that `spec_probe --mode profile` writes and
sums the GPU kernels inside each `step[DRAFT_LOOP ...]` and `step[VERIFY ...]`
annotation, grouped into the parts the T-SPEC4 analysis uses:
  python draft_loop_profile.py <trace.json.gz> [--json out.json]
"""

import argparse
import collections
import gzip
import json
import statistics
from typing import Dict, List


def _part(name: str, grid: List[int]) -> str:
    """Draft-loop kernel class; the head is the GEMV/GEMM whose grid spans the 262144-row vocab."""
    # cuBLAS gemvx covers 8 rows per block (grid 32768); the FP8 head's Triton tile covers 128 (grid 2048);
    # every other GEMV/GEMM on the B=1 path (hidden <= 8192 rows) launches at most 4096 / 132 blocks.
    if "gemvx" in name:
        return "lm_head" if grid and grid[0] >= 16384 else "layer GEMV/GEMM"
    if "small_m_bf16_gemm" in name:
        return "lm_head" if grid and grid[0] >= 1024 else "layer GEMV/GEMM"
    if "SoftMax" in name or "reduce_kernel" in name:
        return "softmax + max over the vocab"
    if "wmma" in name or "splitKreduce" in name:
        return "cuBLAS WMMA + split-K"
    if "tensorrt_llm" in name or "GemmUniversal" in name:
        return "NVFP4 MoE (CUTLASS)"
    if "_fwd_" in name or "_verify_" in name:
        return "attention"
    return "norms, RoPE, glue"


def breakdown(trace_path: str) -> Dict:
    events = json.load(gzip.open(trace_path))["traceEvents"]
    kernels = sorted((e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    steps = collections.defaultdict(list)
    for e in events:
        if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("step["):
            steps[e["name"].split()[0]].append(e)
    out = {}
    for step, windows in steps.items():
        per_part = collections.defaultdict(list)
        totals, spans = [], []
        for w in windows:
            lo, hi = w["ts"], w["ts"] + w["dur"]
            parts = collections.Counter()
            for k in kernels:
                if lo <= k["ts"] and k["ts"] + k["dur"] <= hi + 1:
                    parts[_part(k["name"], k.get("args", {}).get("grid", []))] += k["dur"]
            for p in set(parts) | set(per_part):
                per_part[p].append(parts.get(p, 0.0))
            totals.append(sum(parts.values()))
            spans.append(w["dur"])
        out[step] = {
            "rounds": len(windows),
            "span_ms_median": statistics.median(spans) / 1e3,
            "kernel_ms_median": statistics.median(totals) / 1e3,
            "parts_ms_median": {p: statistics.median(v) / 1e3 for p, v in sorted(per_part.items())},
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--json")
    args = ap.parse_args()
    res = breakdown(args.trace)
    for step, r in res.items():
        print(f"{step}: {r['rounds']} rounds, span {r['span_ms_median']:.3f} ms, kernels {r['kernel_ms_median']:.3f} ms")
        for p, ms in sorted(r["parts_ms_median"].items(), key=lambda kv: -kv[1]):
            print(f"  {ms:7.3f} ms  {p}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
