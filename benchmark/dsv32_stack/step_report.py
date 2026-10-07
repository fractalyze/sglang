"""Split a DP-attention serving run into step types from the per-rank step records.

Reads the `steps-dp<r>-tp<t>.csv` files the `step_profile` hook writes (one per
scheduler) and lines them up by row: under DP attention every rank launches
every step, so row n is the same step on each rank. A step is

- DECODE when no rank prefills;
- PREFILL-ALL when every rank with work prefills (EXTEND or MIXED);
- PREFILL-SOME when at least one rank prefills and another only decodes
  (its decode batch runs as-is or viewed as 1-token extends, CONVERTED).

A step's time is the gap to the next launch on the first rank. With the overlap
scheduler a launch waits for the previous step's result, so the gap tracks the
GPU step; gaps over `--max-gap-s` are idle server time and are dropped.

The prefill step time is also broken down by how many ranks prefill in the step.
When it follows the largest per-rank prefill rather than the rank count,
lining the ranks' prefills up into shared steps cuts the stall below.

The decode stall is what PREFILL-SOME steps cost the decode-only ranks: each
such step takes its own time instead of a decode step's (the median DECODE
time), summed over those steps.

    python3 step_report.py <dir with steps-*.csv> [--start-ts T0 --end-ts T1] [--skip-head-s 60] [--json out.json]

Launch times are CLOCK_MONOTONIC, which a container shares with its host, so a
bench's window can be taken on the host with `time.monotonic()`.
"""

import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

PREFILL_KINDS = ("EXTEND", "MIXED")


def _read_ranks(directory: str) -> list[list[dict]]:
    """One row list per DP rank, keeping the lowest TP rank of each."""
    by_dp = {}
    for path in glob.glob(os.path.join(directory, "steps-dp*-tp*.csv")):
        dp, tp = map(int, re.search(r"steps-dp(\d+)-tp(\d+)\.csv$", path).groups())
        if dp not in by_dp or tp < by_dp[dp][0]:
            by_dp[dp] = (tp, path)
    if not by_dp:
        sys.exit(f"no steps-dp*-tp*.csv in {directory}")
    ranks = []
    for dp in sorted(by_dp):
        with open(by_dp[dp][1]) as f:
            ranks.append(list(csv.DictReader(f)))
    return ranks


def _classify(kinds: list[str]) -> str:
    prefilling = sum(kind in PREFILL_KINDS for kind in kinds)
    if prefilling == 0:
        return "DECODE"
    decoding = sum(kind in ("DECODE", "CONVERTED") for kind in kinds)
    return "PREFILL-SOME" if decoding else "PREFILL-ALL"


def _steps(ranks: list[list[dict]], max_gap_s: float) -> tuple[list[dict], int]:
    """Aligned steps with a time, and how many rows disagreed on the global batch."""
    n = min(len(rows) for rows in ranks)
    steps, mismatched = [], 0
    for i in range(n - 1):
        rows = [r[i] for r in ranks]
        if len({row["global_tokens"] for row in rows}) > 1:
            mismatched += 1
        gap = float(ranks[0][i + 1]["launch_ts"]) - float(rows[0]["launch_ts"])
        if gap > max_gap_s:
            continue
        kinds = [row["local_kind"] for row in rows]
        steps.append(
            {
                "ts": float(rows[0]["launch_ts"]),
                "ms": gap * 1e3,
                "type": _classify(kinds),
                "kinds": kinds,
                "prefill_ranks": sum(kind in PREFILL_KINDS for kind in kinds),
                "prefill_tokens": sum(
                    int(row["local_tokens"])
                    for row in rows
                    if row["local_kind"] in PREFILL_KINDS
                ),
                "decode_bs": sum(
                    int(row["bs"])
                    for row in rows
                    if row["local_kind"] in ("DECODE", "CONVERTED")
                ),
                "decode_graph": rows[0]["decode_graph"] == "1",
                "prefill_graph": rows[0]["prefill_graph"] == "1",
            }
        )
    return steps, mismatched


def _median_ms_by_prefill_ranks(steps: list[dict]) -> dict[int, float]:
    """Prefill step time against how many ranks prefill in it."""
    by_ranks = defaultdict(list)
    for s in steps:
        if s["prefill_ranks"]:
            by_ranks[s["prefill_ranks"]].append(s["ms"])
    return {n: statistics.median(ms) for n, ms in sorted(by_ranks.items())}


def _summary(steps: list[dict]) -> dict:
    wall_s = sum(s["ms"] for s in steps) / 1e3
    by_type = defaultdict(list)
    for s in steps:
        by_type[s["type"]].append(s)
    decode_ms = (
        statistics.median(s["ms"] for s in by_type["DECODE"])
        if by_type["DECODE"]
        else 0.0
    )
    types = {}
    for name in ("DECODE", "PREFILL-SOME", "PREFILL-ALL"):
        group = by_type[name]
        ms = [s["ms"] for s in group]
        types[name] = {
            "steps": len(group),
            "share_of_steps": len(group) / len(steps) if steps else 0.0,
            "median_ms": statistics.median(ms) if ms else None,
            "mean_ms": statistics.fmean(ms) if ms else None,
            "share_of_time": sum(ms) / 1e3 / wall_s if wall_s else 0.0,
            "graph_replay": (
                sum(
                    s["decode_graph"] if name == "DECODE" else s["prefill_graph"]
                    for s in group
                )
                / len(group)
                if group
                else None
            ),
        }
    some = by_type["PREFILL-SOME"]
    stall_s = sum(max(0.0, s["ms"] - decode_ms) for s in some) / 1e3
    return {
        "wall_s": wall_s,
        "steps": len(steps),
        "steps_per_s": len(steps) / wall_s if wall_s else 0.0,
        "types": types,
        "decode_step_median_ms": decode_ms,
        "decode_stall_s": stall_s,
        "decode_stall_share": stall_s / wall_s if wall_s else 0.0,
        "prefill_ranks_per_prefill_step": dict(
            sorted(
                Counter(s["prefill_ranks"] for s in steps if s["prefill_ranks"]).items()
            )
        ),
        "median_ms_by_prefill_ranks": _median_ms_by_prefill_ranks(steps),
        "converted_share_of_some": (
            sum("CONVERTED" in s["kinds"] for s in some) / len(some) if some else None
        ),
        "prefill_tokens_per_prefill_step": (
            statistics.fmean(s["prefill_tokens"] for s in steps if s["prefill_ranks"])
            if any(s["prefill_ranks"] for s in steps)
            else None
        ),
        "decode_bs_median": (
            statistics.median(s["decode_bs"] for s in by_type["DECODE"])
            if by_type["DECODE"]
            else None
        ),
    }


def _print(summary: dict, mismatched: int) -> None:
    print(
        f"steps {summary['steps']} over {summary['wall_s']:.1f} s ({summary['steps_per_s']:.1f}/s)"
    )
    if mismatched:
        print(
            f"WARNING: {mismatched} rows disagree on the global batch; ranks are misaligned"
        )
    print(
        f"{'type':<13}{'steps':>8}{'of steps':>10}{'median ms':>11}{'mean ms':>9}{'of time':>9}{'graph':>7}"
    )
    for name, t in summary["types"].items():
        if not t["steps"]:
            continue
        print(
            f"{name:<13}{t['steps']:>8}{t['share_of_steps']:>10.1%}{t['median_ms']:>11.1f}"
            f"{t['mean_ms']:>9.1f}{t['share_of_time']:>9.1%}{t['graph_replay']:>7.0%}"
        )
    print(
        f"decode stall on non-prefill ranks: {summary['decode_stall_s']:.1f} s "
        f"({summary['decode_stall_share']:.1%} of wall), against a "
        f"{summary['decode_step_median_ms']:.1f} ms decode step at median batch "
        f"{summary['decode_bs_median']}"
    )
    print(
        f"prefilling ranks per prefill step: {summary['prefill_ranks_per_prefill_step']}"
    )
    by_ranks = summary["median_ms_by_prefill_ranks"]
    print(
        "median prefill step ms by prefilling ranks: "
        + ", ".join(f"{n}: {ms:.1f}" for n, ms in by_ranks.items())
    )
    if summary["converted_share_of_some"] is not None:
        print(
            f"PREFILL-SOME steps with a CONVERTED rank: {summary['converted_share_of_some']:.0%}"
        )
    if summary["prefill_tokens_per_prefill_step"] is not None:
        print(
            f"prefill tokens per prefill step: {summary['prefill_tokens_per_prefill_step']:.0f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory")
    parser.add_argument(
        "--start-ts",
        type=float,
        help="first launch time (host CLOCK_MONOTONIC) to keep",
    )
    parser.add_argument("--end-ts", type=float, help="last launch time to keep")
    parser.add_argument(
        "--skip-head-s",
        type=float,
        default=0.0,
        help="drop the first seconds (warm-up, ramp)",
    )
    parser.add_argument(
        "--skip-tail-s", type=float, default=0.0, help="drop the last seconds (drain)"
    )
    parser.add_argument(
        "--max-gap-s", type=float, default=2.0, help="longer gaps are idle time"
    )
    parser.add_argument("--json", help="also write the summary here")
    args = parser.parse_args()

    steps, mismatched = _steps(_read_ranks(args.directory), max_gap_s=args.max_gap_s)
    if args.start_ts is not None:
        steps = [s for s in steps if s["ts"] >= args.start_ts]
    if args.end_ts is not None:
        steps = [s for s in steps if s["ts"] <= args.end_ts]
    if steps:
        start, end = (
            steps[0]["ts"] + args.skip_head_s,
            steps[-1]["ts"] - args.skip_tail_s,
        )
        steps = [s for s in steps if start <= s["ts"] <= end]
    summary = _summary(steps)
    summary["misaligned_rows"] = mismatched
    _print(summary, mismatched)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
