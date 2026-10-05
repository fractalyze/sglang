"""Join an rp-quality run's per-request prefix split (PFXDBG server-log lines) to its items.

The debug tree (G4POC_DEBUG_PREFIX_SPLIT=1) logs, per prefill admission, the device hit, the HiCache
host hit (full and SWA), what the load-back restored and how many restored host slots were stale
from SWA tombstone recovery, keyed by an md5 of the request's input ids. The free-generation pass of
rp-quality sends each item's prompt as-is, so its admission carries the item's hash; the forced-NLL
pass appends a continuation and hashes differently.

  python -m hicache.prefix_split <rp-quality run dir> [--baseline <rp.json>] [--ids s000135/0,...]
"""

import argparse
import hashlib
import json
import os
import re
from typing import Dict, List

import numpy as np

from gate import config, rpquality, runner
from workload import schema

_LINE = re.compile(r"PFXDBG (.*)$")


def ids_hash(ids: List[int]) -> str:
    return hashlib.md5(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()[:12]


def parse_log(path: str) -> Dict[str, List[Dict]]:
    """hash -> admissions (a request can be admitted more than once, e.g. after a retraction)."""
    by_hash: Dict[str, List[Dict]] = {}
    with open(path, errors="replace") as f:
        for line in f:
            m = _LINE.search(line)
            if not m:
                continue
            fields = dict(kv.split("=", 1) for kv in m.group(1).split())
            row = {
                k: (int(v) if v.lstrip("-").isdigit() else v) for k, v in fields.items()
            }
            by_hash.setdefault(row["hash"], []).append(row)
    return by_hash


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument(
        "--baseline",
        help="rp.json of the baseline arm, to mark adherent->non-adherent flips",
    )
    ap.add_argument(
        "--ids", default="", help="comma-separated item ids to print in full"
    )
    a = ap.parse_args()

    with open(os.path.join(a.run_dir, "rp.json")) as f:
        results = {r["id"]: r for r in json.load(f)["results"]}
    baseline = {}
    if a.baseline:
        with open(a.baseline) as f:
            baseline = {r["id"]: r for r in json.load(f)["results"]}
    items = rpquality.build_items(
        schema.read(config.SESSIONS_PATH), runner.load_tokenizer()
    )
    log = parse_log(os.path.join(a.run_dir, "server.log"))

    rows = []
    for it in items:
        res = results.get(it["id"])
        if res is None:
            continue
        adm = log.get(ids_hash(it["input_ids"]), [])
        first = adm[0] if adm else {}
        base = baseline.get(it["id"])
        flipped = (
            bool(base)
            and base["reply_language"] == base["language"] != res["reply_language"]
        )
        rows.append(
            {
                "id": it["id"],
                "language": res["language"],
                "reply": res["reply_language"],
                "flip": flipped,
                "admissions": len(adm),
                "fill": first.get("fill"),
                "device_hit": first.get("device_hit"),
                "host_hit": first.get("host_hit"),
                "swa_host_hit": first.get("swa_host_hit"),
                "loaded": first.get("loaded"),
                "lb_full": first.get("lb_full"),
                "lb_swa": first.get("lb_swa"),
                "stale_full": first.get("stale_full"),
                "stale_swa": first.get("stale_swa"),
            }
        )

    loaded = [r for r in rows if (r["lb_full"] or 0) + (r["lb_swa"] or 0) > 0]
    stale = [r for r in rows if (r["stale_full"] or 0) + (r["stale_swa"] or 0) > 0]
    print(
        json.dumps(
            {
                "items": len(rows),
                "unmatched": sum(r["admissions"] == 0 for r in rows),
                "adherent": sum(r["reply"] == r["language"] for r in rows),
                "with_load_back": len(loaded),
                "with_stale_load_back": len(stale),
                "flips": [r["id"] for r in rows if r["flip"]],
                "flips_with_load_back": [
                    r["id"] for r in rows if r["flip"] and r in loaded
                ],
                "flips_with_stale_load_back": [
                    r["id"] for r in rows if r["flip"] and r in stale
                ],
            },
            indent=1,
        )
    )
    wanted = {i for i in a.ids.split(",") if i}
    for r in rows:
        if r["id"] in wanted or r["flip"] or r in stale:
            print(json.dumps(r))


if __name__ == "__main__":
    main()
