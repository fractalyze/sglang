"""W8 speculative-decoding probe (gemma4nv, T-SPEC1 groundwork).

Screens only: every number here is a single unpaired run on one server
lifetime. They size the T-SPEC1 prediction; the verdict comes from `gate run`.

Each mode is one server lifetime of a gate ref plus extra flags, under the
gate's host-safety path (host.lock, preflight, 24G no-swap scope, watchdog):

  spec    -- the ref with speculative flags. Accept length on the hidden
             prompt set (per category, contents never printed) and on the
             gate's timing corpus, plus client TPOT at B=1 and B=8.
  sweep   -- target only. Decode step time with exactly M running requests,
             from the per-step decode log (#running-req / gen throughput),
             for M in --rows. cost(M rows) for the verify-cost curve.
  experts -- target only with --enable-return-routed-experts. Raw routing
             [tokens, layers, top_k] per stream, saved for verify_cost.py.
  profile -- torch-profiler trace of --profile-steps decode steps at B=1
             (W2's start_profile shape), after one warm-up request.
  info    -- launch only. Saves /get_server_info, so a candidate's resolved
             args can be diffed against its control with the gate's own check
             before a gate run is spent on an undeclared derived field.

Run on a gemma4nv host from the deployed experiments dir:
  python -m trials.spec.spec_probe --mode spec --ref smallm-gemm --out DIR -- <extra sglang flags>
"""

import argparse
import asyncio
import base64
import collections
import json
import os
import re
import statistics
import time
from typing import Dict, List

import numpy as np
import requests

from gate import client, config, hostwatch, prompts, server

N_LAYERS, TOP_K = 30, 8
BOS = 2


def _timing_prompts(seed: str, n: int, length: int = 1024) -> List[List[int]]:
    return prompts.timing_prompts(prompts.load_corpus(), BOS, seed, n, length)


def _hidden() -> List[Dict]:
    with open(os.path.join(config.HIDDEN_DIR, "fidelity_prompts.jsonl")) as f:
        return [json.loads(line) for line in f]


def _generate(url: str, ids: List[int], max_new: int, **extra) -> Dict:
    payload = {
        "input_ids": ids,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new, "ignore_eos": True},
        **extra,
    }
    t0 = time.perf_counter()
    r = requests.post(f"{url}/generate", json=payload, timeout=1800)
    r.raise_for_status()
    body = r.json()
    body["_e2e_s"] = time.perf_counter() - t0
    return body


def _accept_stats(metas: List[Dict]) -> Dict:
    toks = sum(m["completion_tokens"] for m in metas)
    verify = sum(m.get("spec_verify_ct", 0) for m in metas)
    # Index i counts verify rounds that kept i drafts (bonus excluded).
    hist: List[int] = []
    for m in metas:
        for i, c in enumerate(m.get("spec_correct_drafts_histogram") or []):
            hist.extend([0] * (i + 1 - len(hist)))
            hist[i] += c
    return {"n": len(metas), "tokens": toks, "verify_ct": verify,
            "accept_length": round(toks / verify, 3) if verify else None,
            "correct_drafts_histogram": hist}


def mode_spec(srv: server.Server, out: Dict, decode_new: int) -> None:
    by_cat = collections.defaultdict(list)
    per_prompt = []
    for p in _hidden():
        srv.flush_cache()
        body = _generate(srv.url, p["input_ids"], decode_new)
        by_cat[p["category"]].append(body["meta_info"])
        # Index and category only; hidden prompt contents are never written out.
        per_prompt.append({"index": len(per_prompt), "category": p["category"],
                           "accept_length": _accept_stats([body["meta_info"]])["accept_length"]})
    out["hidden"] = {cat: _accept_stats(m) for cat, m in sorted(by_cat.items())}
    out["hidden_per_prompt"] = per_prompt
    out["hidden_all"] = _accept_stats([m for ms in by_cat.values() for m in ms])

    for batch, reps, new in ((1, 3, 256), (8, 2, 128), (32, 1, 128)):
        rows, wall_s = [], 0.0
        for rep in range(reps + 1):  # rep 0 warms up
            srv.flush_cache()
            ps = _timing_prompts(f"w8-spec-b{batch}-r{rep}", batch)
            res = asyncio.run(client.run_batch(srv.url, ps, new))
            if rep == 0:
                continue
            wall_s += res["wall_s"]
            for s in res["streams"]:
                rows.append({"tpot_ms": 1e3 * (s["e2e_s"] - s["ttft_s"]) / (s["output_tokens"] - 1),
                             "ttft_s": s["ttft_s"]})
        # Acceptance on the same corpus, read back per request (non-streamed).
        metas = []
        for p in _timing_prompts(f"w8-spec-acc-b{batch}", 4 if batch == 1 else 8):
            srv.flush_cache()
            metas.append(_generate(srv.url, p, new)["meta_info"])
        out[f"timing_b{batch}"] = {
            "tpot_ms_median": statistics.median(r["tpot_ms"] for r in rows),
            "ttft_s_median": statistics.median(r["ttft_s"] for r in rows),
            "wall_tok_s": round(batch * new * reps / wall_s, 1),
            "tpot_ms_all": [round(r["tpot_ms"], 3) for r in rows],
            "accept": _accept_stats(metas),
        }


_DECODE_LINE = re.compile(r"Decode batch.*?#running-req: (\d+).*?cuda graph: (\w+).*?gen throughput \(token/s\): ([\d.]+)")


def mode_sweep(srv: server.Server, out: Dict, rows: List[int]) -> None:
    for m in rows:
        steps = []
        for rep in range(2):
            srv.flush_cache()
            off = srv.log_offset()
            ps = _timing_prompts(f"w8-sweep-m{m}-r{rep}", m)
            asyncio.run(client.run_batch(srv.url, ps, 96))
            time.sleep(0.5)
            with open(srv.log_path, errors="replace") as f:
                f.seek(off)
                text = f.read()
            for n, graph, tput in _DECODE_LINE.findall(text):
                if int(n) == m and graph == "True" and float(tput) > 0:
                    steps.append(1e3 * m / float(tput))
        # Drop the first steps after the last admission (they share the
        # interval with prefill); the median over the rest is the step time.
        out.setdefault("sweep", {})[str(m)] = {
            "n_steps": len(steps),
            "step_ms_median": round(statistics.median(steps[5:] or steps), 4) if steps else None,
        }
        print(m, out["sweep"][str(m)], flush=True)


def mode_experts(srv: server.Server, out: Dict, out_dir: str, decode_new: int) -> None:
    sets = {"timing": _timing_prompts("w8-experts", 24), "hidden": [p["input_ids"] for p in _hidden()]}
    for name, ps in sets.items():
        arrs = []
        for ids in ps:
            body = _generate(srv.url, ids, decode_new, return_routed_experts=True, routed_experts_start_len=len(ids))
            flat = np.frombuffer(base64.b64decode(body["meta_info"]["routed_experts"]), dtype=np.int32)
            arrs.append(flat.reshape(-1, N_LAYERS, TOP_K))
        np.savez_compressed(os.path.join(out_dir, f"routing_{name}.npz"), *arrs)
        out[f"experts_{name}"] = {"streams": len(arrs), "tokens": [int(a.shape[0]) for a in arrs]}


def mode_profile(srv: server.Server, out: Dict, out_dir: str, steps: int) -> None:
    srv.flush_cache()
    asyncio.run(client.run_batch(srv.url, _timing_prompts("w8-prof-warm", 1), 64))
    srv.flush_cache()
    trace_dir = os.path.join(out_dir, "trace_b1")
    requests.post(f"{srv.url}/start_profile", json=dict(
        output_dir=trace_dir, num_steps=steps, activities=["CPU", "GPU"], profile_by_stage=True,
        record_shapes=True, with_stack=False), timeout=60).raise_for_status()
    res = asyncio.run(client.run_batch(srv.url, _timing_prompts("w8-prof", 1), 96))
    out["profile"] = {"trace_dir": trace_dir, "stream": res["streams"][0] | {"output_ids": None}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("spec", "sweep", "experts", "profile", "info"))
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default="1,2,3,4,5,6,8,12,16,24,32,40,48")
    ap.add_argument("--decode-new", type=int, default=256)
    ap.add_argument("--profile-steps", type=int, default=12)
    ap.add_argument("extra", nargs="*")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ref = server.load_ref(args.ref)
    out = {"mode": args.mode, "ref": args.ref, "extra_args": args.extra, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with hostwatch.host_lock(), server.Server(ref, os.path.join(args.out, "server.log"), args.extra) as srv:
        out["commit"] = srv.commit
        out["preflight"] = srv.preflight
        out["server_info"] = srv.server_info()
        if args.mode == "spec":
            mode_spec(srv, out, args.decode_new)
        elif args.mode == "sweep":
            mode_sweep(srv, out, [int(x) for x in args.rows.split(",")])
        elif args.mode == "experts":
            mode_experts(srv, out, args.out, args.decode_new)
        elif args.mode == "profile":
            mode_profile(srv, out, args.out, args.profile_steps)
    out["host_summary"] = srv.host_summary
    with open(os.path.join(args.out, "probe.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("preflight", "host_summary", "server_info")}, indent=1))


if __name__ == "__main__":
    main()
