"""Logprob/KL check of a candidate against the base on long role-play prompts, at the base's A/A level.

The gate's fidelity prompts are mostly short; a change to the extend (prefill) kernels shows on
long contexts. This takes one first turn per language from the role-play items (`rpquality`:
persona card + ~5K tokens of history + user turn), decodes the base's greedy reply as the
reference, and compares teacher-forced top-k logprobs over that reply (every reply token is
scored in one extend over the ~5K-token prompt):

- A/A level: the base's forced pass with every prompt in one batch against the base one prompt
  at a time (the batch-composition noise the gate's calibration also uses);
- candidate: the candidate's forced pass, every prompt in one batch, against the base's.

Pass: the candidate's KL mean and p99 are at most FACTOR x the A/A level (with the gate's KL
floors), and its worst per-prompt top-1 agreement is at most AGREEMENT_SLACK below the A/A's.

Greedy token identity is checked one prompt at a time only (batched outputs are not
run-to-run deterministic on this engine): the base twice (its own determinism) and the
candidate once. It is reported, not gated: a change that reorders a reduction may flip a
near-tie, so each divergence carries the base's top-1 minus top-2 logprob at that token.

  python compute/kl_check.py --candidate <ref> [--control base] [--n 8]
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FACTOR = 2.0
AGREEMENT_SLACK = 0.02
MAX_NEW_TOKENS = 192


def pick_items(items: List[Dict], n: int) -> List[Dict]:
    """The first item of each language in turn, until n."""
    by_lang: Dict[str, List[Dict]] = {}
    for it in items:
        by_lang.setdefault(it["language"], []).append(it)
    picked, k = [], 0
    while len(picked) < n and any(k < len(v) for v in by_lang.values()):
        picked += [v[k] for _, v in sorted(by_lang.items()) if k < len(v)][: n - len(picked)]
        k += 1
    return picked


def verdict(aa: Dict, cand: Dict, kl_mean_floor: float, kl_p99_floor: float) -> Dict:
    limits = {
        "kl_mean_max": max(FACTOR * aa["kl_mean"], kl_mean_floor),
        "kl_p99_max": max(FACTOR * aa["kl_p99"], kl_p99_floor),
        "min_top1_agreement_min": aa["min_top1_agreement"] - AGREEMENT_SLACK,
    }
    checks = {
        "kl_mean": cand["kl_mean"] <= limits["kl_mean_max"],
        "kl_p99": cand["kl_p99"] <= limits["kl_p99_max"],
        "min_top1_agreement": cand["min_top1_agreement"] >= limits["min_top1_agreement_min"],
    }
    return {"pass": all(checks.values()), "checks": checks, "limits": limits}


def serial_identity(control: List[Dict], other: List[Dict]) -> Dict:
    """Greedy outputs one prompt at a time: identical prompts, and per divergence the control's top-2 margin."""
    from gate import fidelity

    rows = []
    for c, o in zip(control, other):
        i = fidelity.first_divergence(c["output_ids"], o["output_ids"])
        row = {"id": c["id"], "first_divergence": i}
        if 0 <= i < len(c["top_logprobs"]):
            top = sorted((lp for lp, _ in c["top_logprobs"][i]), reverse=True)
            row["control_top2_margin"] = top[0] - top[1] if len(top) > 1 else None
        rows.append(row)
    return {"n_identical": sum(r["first_divergence"] < 0 for r in rows), "n": len(rows), "per_prompt": rows}


def _summary(cmp: Dict) -> Dict:
    return {k: cmp[k] for k in ("kl_mean", "kl_p99", "min_top1_agreement", "mean_top1_agreement")}


def main() -> None:
    from gate import client, config, fidelity, hostwatch, rpquality, runner, server
    from workload import schema

    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--control", default="base")
    p.add_argument("--n", type=int, default=8)
    args = p.parse_args()

    out_dir = os.path.join(config.RUNS_DIR, runner.new_exp_id(f"kl-{args.candidate}"))
    os.makedirs(out_dir)
    items = pick_items(rpquality.build_items(schema.read(config.SESSIONS_PATH), runner.load_tokenizer()), args.n)
    prompts = [it["input_ids"] for it in items]
    n, k = len(prompts), config.TOP_LOGPROBS

    def serve(ref_name: str, tag: str):
        ref = server.load_ref(ref_name)
        return server.Server(ref, os.path.join(out_dir, f"server-{tag}.log"), extra_args=runner.server_extra_args(ref))

    def forced(url: str, reference: List[Dict], concurrency: int) -> List[Dict]:
        rows = asyncio.run(client.forced_batch(url, prompts, [r["output_ids"] for r in reference], k, concurrency))
        return [{"id": r["id"], "top_logprobs": row} for r, row in zip(reference, rows)]

    with hostwatch.host_lock():
        with serve(args.control, "control") as srv:
            outs = asyncio.run(client.greedy_batch(srv.url, prompts, MAX_NEW_TOKENS, k, n))
            reference = [{"id": it["id"], "category": it["language"], "prompt_tokens": len(it["input_ids"]), **o}
                         for it, o in zip(items, outs)]
            srv.flush_cache()
            ctrl_batched = forced(srv.url, reference, n)
            srv.flush_cache()
            ctrl_serial = forced(srv.url, reference, 1)
            srv.flush_cache()
            ctrl_greedy_1 = asyncio.run(client.greedy_batch(srv.url, prompts, MAX_NEW_TOKENS, k, 1))
            srv.flush_cache()
            ctrl_greedy_1b = asyncio.run(client.greedy_batch(srv.url, prompts, MAX_NEW_TOKENS, k, 1))
            control_commit = srv.commit
        with serve(args.candidate, "candidate") as srv:
            cand_batched = forced(srv.url, reference, n)
            srv.flush_cache()
            cand_greedy_1 = asyncio.run(client.greedy_batch(srv.url, prompts, MAX_NEW_TOKENS, k, 1))
            candidate_commit = srv.commit

    aa = fidelity.compare_forced(reference, ctrl_batched, ctrl_serial)
    cand = fidelity.compare_forced(reference, ctrl_batched, cand_batched)
    res = {
        "control": args.control, "candidate": args.candidate,
        "commits": {"control": control_commit, "candidate": candidate_commit},
        "items": [{"id": it["id"], "language": it["language"], "prompt_tokens": len(it["input_ids"]),
                   "reply_tokens": len(r["output_ids"])} for it, r in zip(items, reference)],
        "aa": _summary(aa), "candidate_vs_control": _summary(cand),
        "serial_token_identity": {
            "aa": serial_identity([{"id": it["id"], **o} for it, o in zip(items, ctrl_greedy_1)], ctrl_greedy_1b),
            "candidate": serial_identity([{"id": it["id"], **o} for it, o in zip(items, ctrl_greedy_1)],
                                         cand_greedy_1),
        },
        "per_prompt": {"aa": aa["per_prompt"], "candidate": cand["per_prompt"]},
        "verdict": verdict(_summary(aa), _summary(cand), config.KL_MEAN_FLOOR, config.KL_P99_FLOOR),
    }
    with open(os.path.join(out_dir, "kl_check.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("items", "aa", "candidate_vs_control", "verdict")}, indent=1))
    print(json.dumps({role: {k: v for k, v in r.items() if k != "per_prompt"} | {
        "divergences": [p for p in r["per_prompt"] if p["first_divergence"] >= 0]}
        for role, r in res["serial_token_identity"].items()}, indent=1))
    print(f"run dir: {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
