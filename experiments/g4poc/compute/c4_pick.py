"""C4: the candidate to confirm from the nested sweeps against the final stack, or "none".

Rule (registered in compute/PREREG.md before the sweeps): over the in-flight counts POINTS, the candidate
with the highest mean log E2E p90 gain, if that mean exceeds log(MIN_MEAN_GAIN) and no point's gain is
below MIN_POINT_GAIN.

  python compute/c4_pick.py --dir <sweep_nested out dir> <cand1> <cand2> ...
"""

import argparse
import json
import math
import os
from typing import Dict, List, Sequence

POINTS = (24, 32)
MIN_MEAN_GAIN = 1.01
MIN_POINT_GAIN = 0.99


def pick(results: Dict[str, List[Dict]], points: Sequence[int] = POINTS) -> str:
    best, best_score = "none", math.log(MIN_MEAN_GAIN)
    for cand, rows in results.items():
        by_c = {r["concurrency"]: r["e2e_p90_gain"] for r in rows}
        if any(c not in by_c for c in points) or min(by_c[c] for c in points) < MIN_POINT_GAIN:
            continue
        score = sum(math.log(by_c[c]) for c in points) / len(points)
        if score > best_score:
            best, best_score = cand, score
    return best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("cands", nargs="+")
    args = p.parse_args()
    results = {}
    for c in args.cands:
        with open(os.path.join(args.dir, f"{c}.json")) as f:
            results[c] = json.load(f)["points"]
    print(pick(results))


if __name__ == "__main__":
    main()
