"""Paired role-play comparison of two `gate rp-quality` runs over the same items.

The rp guard's language-adherence count moves by about one item between numerics-neutral configs (round 1: batched
outputs re-roll on any config change, ~1/80 language flip), so a candidate is judged per item against a control run
on the same host: items that were adherent under the control and are not under the candidate (out), the reverse
(in), and the net. The NLL rise is the candidate's mean reference NLL minus the control's (nats/token).

  python glue/rp_pair.py <control rp.json> <candidate rp.json>
"""

import json
import sys
from typing import Dict, List


def _adherent(results: List[Dict]) -> Dict[str, bool]:
    return {r["id"]: r["reply_language"] == r["language"] for r in results}


def pair(control: List[Dict], candidate: List[Dict]) -> Dict:
    a, b = _adherent(control), _adherent(candidate)
    if set(a) != set(b):
        raise ValueError("the two runs cover different items")
    out = sorted(i for i in a if a[i] and not b[i])
    into = sorted(i for i in a if b[i] and not a[i])
    nll = {r["id"]: r["ref_nll"] for r in control}
    rise = sum(r["ref_nll"] - nll[r["id"]] for r in candidate) / len(candidate)
    return {"n": len(a), "control_adherent": sum(a.values()), "candidate_adherent": sum(b.values()),
            "flips_out": out, "flips_in": into, "net_out": len(out) - len(into), "nll_rise": rise}


def main() -> None:
    control, candidate = (json.load(open(p))["results"] for p in sys.argv[1:3])
    print(json.dumps(pair(control, candidate), indent=1))


if __name__ == "__main__":
    main()
