"""Flags kept from the nested sweeps against the final stack, and the ref that combines them.

Rule (registered in compute/PREREG.md before the sweeps): a candidate is kept if its mean log E2E p90 gain
over the in-flight counts POINTS exceeds log(MIN_MEAN_GAIN) and no point's gain is below MIN_POINT_GAIN.
Candidates are `<control>-<flag>` refs; the combined ref is `<control>-<kept flags sorted, joined by ->`.

  python compute/c4_pick.py --dir <sweep_nested out dir> --control final-hc <cand1> <cand2> ...
  (prints the combined ref, or "none")
"""

import argparse
import json
import math
import os
from typing import Dict, List, Sequence

POINTS = (24, 32)
MIN_MEAN_GAIN = 1.01
MIN_POINT_GAIN = 0.99


def kept(results: Dict[str, List[Dict]], points: Sequence[int] = POINTS) -> List[str]:
    out = []
    for cand, rows in results.items():
        by_c = {r["concurrency"]: r["e2e_p90_gain"] for r in rows}
        if any(c not in by_c for c in points) or min(by_c[c] for c in points) < MIN_POINT_GAIN:
            continue
        if sum(math.log(by_c[c]) for c in points) / len(points) > math.log(MIN_MEAN_GAIN):
            out.append(cand)
    return out


def combined(control: str, cands: Sequence[str]) -> str:
    if not cands:
        return "none"
    return control + "-" + "-".join(sorted(c[len(control) + 1:] for c in cands))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--control", required=True)
    p.add_argument("cands", nargs="+")
    args = p.parse_args()
    results = {}
    for c in args.cands:
        with open(os.path.join(args.dir, f"{c}.json")) as f:
            results[c] = json.load(f)["points"]
    print(combined(args.control, kept(results)))


if __name__ == "__main__":
    main()
