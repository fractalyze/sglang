"""Paired in-flight sweeps: control and candidate sweeps run A B B A, compared point by point.

The gate's ABBA (`gate run`) decides at one load. A lever that should move the hit-rate cliff
(C3) or be checked at a second load (C1/C2 at 8 in flight) is compared across several in-flight
counts instead: `gate sweep` runs control, candidate, candidate, control, and each in-flight
count gets two pairs (sweep 1 vs 2, sweep 4 vs 3). Gains are control/candidate for E2E p90
(above 1 = faster) and candidate/control for output tok/s, geometric mean over the pairs.

  python compute/sweep_abba.py --sweeps <A1 dir> <B1 dir> <B2 dir> <A2 dir>
"""

import argparse
import json
import math
import os
from typing import Dict, List


def _points(sweep: Dict) -> Dict[int, Dict]:
    return {p["load"]["concurrency"] if isinstance(p["load"], dict) else int(p["load"].rsplit("C", 1)[1]):
            p for p in sweep["points"]}


def compare(a1: Dict, b1: Dict, b2: Dict, a2: Dict) -> List[Dict]:
    pairs = [(_points(a1), _points(b1)), (_points(a2), _points(b2))]
    rows = []
    for c in sorted(set.intersection(*(set(p) for pair in pairs for p in pair))):
        def gain(metric: str, lower_better: bool) -> float:
            logs = []
            for ctrl, cand in pairs:
                x, y = ctrl[c]["summary"][metric], cand[c]["summary"][metric]
                logs.append(math.log(x / y if lower_better else y / x))
            return math.exp(sum(logs) / len(logs))

        def both(points_role: int, metric: str) -> List[float]:
            return [pair[points_role][c]["summary"][metric] for pair in pairs]

        rows.append({
            "concurrency": c,
            "e2e_p90_gain": gain("e2e_p90_s", True),
            "output_tput_gain": gain("output_tok_s_per_gpu", False),
            "control_e2e_p90_s": both(0, "e2e_p90_s"), "candidate_e2e_p90_s": both(1, "e2e_p90_s"),
            "control_out_tok_s": both(0, "output_tok_s_per_gpu"),
            "candidate_out_tok_s": both(1, "output_tok_s_per_gpu"),
            "control_hit": both(0, "prefix_cache_hit_rate"), "candidate_hit": both(1, "prefix_cache_hit_rate"),
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweeps", nargs=4, required=True, metavar=("A1", "B1", "B2", "A2"))
    args = p.parse_args()
    sweeps = []
    for d in args.sweeps:
        with open(os.path.join(d, "sweep.json")) as f:
            sweeps.append(json.load(f))
    print(json.dumps({"refs": [s["ref"] for s in sweeps], "sweeps": args.sweeps, "points": compare(*sweeps)},
                     indent=1))


if __name__ == "__main__":
    main()
