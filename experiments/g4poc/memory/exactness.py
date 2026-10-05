"""Greedy outputs of one ref on fixed long prompts, and an exact comparison of two such runs.

A lever that claims unchanged numerics must reproduce its control's greedy outputs token for
token where the batch composition is the same (concurrency 1). Prompts are seeded random token
ids (no shared prefix), decoded to exactly `output_len` tokens.

  python -m memory.exactness run --ref mem-stack1 --n 6 --concurrency 1,8
  python -m memory.exactness compare <control exactness.json> <candidate exactness.json>
"""

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import requests

from gate import config, fidelity, hostwatch, runner, server


def generate(url: str, input_len: int, output_len: int, seed: int) -> Dict:
    rng = random.Random(seed)
    ids = [rng.randrange(1000, 200000) for _ in range(input_len)]
    r = requests.post(f"{url}/generate", json={
        "input_ids": ids,
        "sampling_params": {"max_new_tokens": output_len, "ignore_eos": True, "temperature": 0},
    }, timeout=3600)
    r.raise_for_status()
    out = r.json()
    return {"seed": seed, "text": out["text"], "output_ids": out.get("output_ids")}


def compare(control: Dict, candidate: Dict) -> Dict:
    """Per concurrency: how many outputs match exactly, and where the first mismatch starts."""
    res = {}
    for conc, ctl_outs in control["outputs"].items():
        cand_outs = candidate["outputs"].get(conc)
        if cand_outs is None:
            continue
        key = "output_ids" if all(o["output_ids"] is not None for o in ctl_outs + cand_outs) else "text"
        rows = []
        for a, b in zip(ctl_outs, cand_outs):
            assert a["seed"] == b["seed"], (a["seed"], b["seed"])
            x, y = a[key], b[key]
            first = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None)
            if first is None and len(x) != len(y):
                first = min(len(x), len(y))
            rows.append({"seed": a["seed"], "match": first is None, "first_mismatch": first})
        res[conc] = {"compared_on": key, "n": len(rows), "exact": sum(r["match"] for r in rows), "rows": rows}
    return res


def _run(a) -> None:
    ref = server.load_ref(a.ref)
    out_dir = os.path.join(config.RUNS_DIR, runner.new_exp_id(f"exact-{a.ref}"))
    os.makedirs(out_dir)
    seeds = [7000 + i for i in range(a.n)]
    res: Dict = {"ref": ref, "input_len": a.input_len, "output_len": a.output_len, "outputs": {}}
    with hostwatch.host_lock(), server.Server(ref, os.path.join(out_dir, "server.log"),
                                              extra_args=runner.server_extra_args(ref)) as srv:
        res["commit"] = srv.commit
        for conc in (int(x) for x in a.concurrency.split(",")):
            requests.post(f"{srv.url}/flush_cache", timeout=60)
            with ThreadPoolExecutor(conc) as ex:
                outs: List[Dict] = list(ex.map(lambda s: generate(srv.url, a.input_len, a.output_len, s), seeds))
            res["outputs"][str(conc)] = outs
            print(f"concurrency {conc}: {len(outs)} outputs", file=sys.stderr, flush=True)
    res["host"] = srv.host_summary
    fidelity.save_json(os.path.join(out_dir, "exactness.json"), res)
    print(f"run dir: {out_dir}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--ref", required=True)
    r.add_argument("--n", type=int, default=6)
    r.add_argument("--input-len", type=int, default=5000)
    r.add_argument("--output-len", type=int, default=300)
    r.add_argument("--concurrency", default="1")
    c = sub.add_parser("compare")
    c.add_argument("control")
    c.add_argument("candidate")
    a = ap.parse_args()
    if a.cmd == "run":
        _run(a)
    else:
        with open(a.control) as f, open(a.candidate) as g:
            print(json.dumps(compare(json.load(f), json.load(g)), indent=1))


if __name__ == "__main__":
    main()
