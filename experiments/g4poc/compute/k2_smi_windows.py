"""GPU SM clock, power and temperature per gate run, from a 1 s nvidia-smi sample file.

Samples come from `nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,
utilization.gpu,memory.used --format=csv,noheader,nounits -l 1`. A run's span is its directory's start stamp
(`<label>-YYYYMMDD-HHMMSS-<host>-<id>`) to the newest file in it; only busy samples (utilization >= BUSY_PCT) count,
so server start-up and idle gaps between points do not dilute the means.

  python compute/k2_smi_windows.py --samples smi.csv --runs <run dir> [<run dir> ...]
"""

import argparse
import json
import os
import re
import statistics
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

BUSY_PCT = 50.0
FIELDS = ("clocks_sm", "clocks_mem", "power_w", "temp_c", "util_pct", "mem_mib")


def parse(lines: Sequence[str]) -> List[Tuple[float, Dict[str, float]]]:
    out = []
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != 1 + len(FIELDS):
            continue
        try:
            t = datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            vals = {k: float(v) for k, v in zip(FIELDS, parts[1:])}
        except ValueError:
            continue
        out.append((t, vals))
    return out


def run_span(run_dir: str) -> Tuple[float, float]:
    m = re.search(r"-(\d{8}-\d{6})-", os.path.basename(run_dir.rstrip("/")))
    if not m:
        raise ValueError(f"no start stamp in {run_dir}")
    start = datetime.strptime(m.group(1), "%Y%m%d-%H%M%S").timestamp()
    end = max(os.path.getmtime(os.path.join(root, f)) for root, _, files in os.walk(run_dir) for f in files)
    return start, end


def summarize(samples: List[Tuple[float, Dict[str, float]]], t0: float, t1: float) -> Dict:
    busy = [v for t, v in samples if t0 <= t <= t1 and v["util_pct"] >= BUSY_PCT]
    if not busy:
        return {"n_busy": 0}
    return {"n_busy": len(busy), **{f"{k}_mean": statistics.fmean(v[k] for v in busy)
                                    for k in ("clocks_sm", "clocks_mem", "power_w", "temp_c")},
            "clocks_sm_p10": sorted(v["clocks_sm"] for v in busy)[len(busy) // 10]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    args = p.parse_args()
    with open(args.samples) as f:
        samples = parse(f.read().splitlines())
    res = {os.path.basename(r.rstrip("/")): summarize(samples, *run_span(r)) for r in args.runs}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
