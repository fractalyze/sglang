"""C4: the candidate to gate from the nested sweeps at 12 and 20 in flight, or "none".

Rule (registered in compute/PREREG.md before the sweeps): the highest E2E p90 gain at 12 in flight among
candidates whose gain there exceeds MIN_GAIN_C12 and whose gain at 20 is at least MIN_GAIN_C20.

  python compute/c4_pick.py --dir <sweep_nested out dir> <cand1> <cand2> ...
"""

import argparse
import json
import os
from typing import Dict, List

MIN_GAIN_C12 = 1.01
MIN_GAIN_C20 = 0.99


def pick(results: Dict[str, List[Dict]]) -> str:
    best, best_gain = "none", MIN_GAIN_C12
    for cand, rows in results.items():
        by_c = {r["concurrency"]: r for r in rows}
        if 12 not in by_c or 20 not in by_c:
            continue
        g12, g20 = by_c[12]["e2e_p90_gain"], by_c[20]["e2e_p90_gain"]
        if g12 > best_gain and g20 >= MIN_GAIN_C20:
            best, best_gain = cand, g12
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
