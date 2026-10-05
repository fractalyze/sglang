"""GPU memory against the deployable rule: the plateau must stay at least 512 MiB under what torch can use.

Reads `nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 100` samples taken
while a server ran, and reports the peak, how long the GPU sat within 64 MiB of it, and the margin to
torch's capacity (`torch.cuda.get_device_properties(0).total_memory`, which is below what nvidia-smi
calls total). A peak shorter than SPIKE_MAX_S is reported as a spike next to the plateau (the highest
level held for at least that long).

  python compute/mem_check.py --samples mem.csv [--capacity-mib 32111]
"""

import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple

MARGIN_MIB = 512
SPIKE_MAX_S = 1.0
NEAR_PEAK_MIB = 64


def parse(lines: List[str]) -> List[Tuple[float, int]]:
    out = []
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        t = datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
        out.append((t, int(parts[1])))
    return out


def plateau(samples: List[Tuple[float, int]], hold_s: float) -> int:
    """The highest memory level the GPU held for at least `hold_s` without dropping below it."""
    best = 0
    for i, (t0, _) in enumerate(samples):
        lo = None
        for t, m in samples[i:]:
            lo = m if lo is None else min(lo, m)
            if t - t0 >= hold_s:
                best = max(best, lo)
                break
    return best


def check(samples: List[Tuple[float, int]], capacity_mib: int) -> Dict:
    peak = max(m for _, m in samples)
    near = [t for t, m in samples if m >= peak - NEAR_PEAK_MIB]
    level = plateau(samples, SPIKE_MAX_S)
    limit = capacity_mib - MARGIN_MIB
    return {"n_samples": len(samples), "peak_mib": peak, "plateau_mib": level, "limit_mib": limit,
            "capacity_mib": capacity_mib, "plateau_ok": level <= limit, "peak_ok": peak <= limit,
            "seconds_near_peak": (near[-1] - near[0]) if near else 0.0}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", required=True)
    p.add_argument("--capacity-mib", type=int, default=32111, help="bs2: torch total_memory 33,670,758,400 B")
    args = p.parse_args()
    with open(args.samples) as f:
        print(json.dumps(check(parse(f.readlines()), args.capacity_mib), indent=1))


if __name__ == "__main__":
    main()
