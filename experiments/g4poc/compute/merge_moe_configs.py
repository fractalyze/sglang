"""Merge fused-MoE tuner outputs that cover disjoint token counts into one config file.

C1 tunes decode-size and prefill-size token counts in separate runs (each with its own
search space), so each run writes a partial `E=...,device_name=...json`. SGLang loads one
file per shape and picks the nearest token count, so the parts are merged by key:

  python compute/merge_moe_configs.py --out <dir>/configs/triton_<ver>/<name>.json part1.json part2.json
"""

import argparse
import json
import os
from typing import Dict, List


def merge(parts: List[Dict[str, Dict]]) -> Dict[str, Dict]:
    merged: Dict[str, Dict] = {}
    for part in parts:
        for m, cfg in part.items():
            if m in merged and merged[m] != cfg:
                raise ValueError(f"token count {m} tuned twice with different configs")
            merged[m] = cfg
    return {m: merged[m] for m in sorted(merged, key=int)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("parts", nargs="+")
    args = p.parse_args()
    parts = []
    for path in args.parts:
        with open(path) as f:
            parts.append(json.load(f))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(merge(parts), f, indent=4)
        f.write("\n")


if __name__ == "__main__":
    main()
