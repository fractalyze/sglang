"""Re-derive a `gate pd-measure` run's decode points from its server log (per-step decode rates).

pd-measure runs before 2026-10-05 ~20:00 read the decode rate from wall clock over the second pass, which needs
the first pass's prefixes cached; on Gemma-4's hybrid sliding-window pool they were not (hit ~0 from batch 16),
so no point qualified. Each decode point runs after a cache flush, so the log splits at the flushes: the last
len(decode) segments are the decode points in order, and their full-batch decode steps give the rate.

  python compute/pd_steps_from_log.py --run <pd run dir> [--max-tpot 0.030]   # writes <run>/pd-steps.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gate import pd  # noqa: E402

FLUSH = "Cache flushed successfully"


def decode_segments(log_text: str, n_decode: int) -> List[str]:
    segments = log_text.split(FLUSH)
    if len(segments) < n_decode:
        raise ValueError(f"{len(segments)} log segments for {n_decode} decode points")
    return segments[-n_decode:]


def rederive(pdm: Dict, log_text: str, max_tpot_s: float) -> Dict:
    out = dict(pdm)
    points = []
    for point, seg in zip(pdm["decode"], decode_segments(log_text, len(pdm["decode"]))):
        p = {k: v for k, v in point.items()}
        p.update(pd.step_point(seg, p["batch"], point))
        points.append(p)
    out["decode"] = points
    out["best_decode"] = pd.best_decode(points, max_tpot_s)
    out["max_tpot_s"] = max_tpot_s
    out["decode_rederived_from_log"] = True
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--max-tpot", type=float, default=0.030)
    args = p.parse_args()
    with open(os.path.join(args.run, "pd.json")) as f:
        pdm = json.load(f)
    with open(os.path.join(args.run, "server.log"), errors="replace") as f:
        log_text = f.read()
    res = rederive(pdm, log_text, args.max_tpot)
    with open(os.path.join(args.run, "pd-steps.json"), "w") as f:
        json.dump(res, f, indent=1)
    for d in res["decode"]:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()})
    print("best_decode:", res["best_decode"])


if __name__ == "__main__":
    main()
