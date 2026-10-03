"""Distinct routed experts per MoE layer per decode step, and same-expert run
lengths, at B concurrent streams over diverse prompts.

Gemma4 has no ExpertLocationMetadata, so SGLang's expert distribution recorder
cannot run on it; routing comes from --enable-return-routed-experts instead
(same route as W1's `gate sol`). Request r returns int32 [steps, layers, top_k]
from routed_experts_start_len = prompt length, so row j is the decode pass that
consumed generated token j; step j is the union over the B streams (lockstep
approximation of one forward pass).

Usage: python experts.py --url URL --prompts prompts.json --batch B --groups G --out out.json
"""

import argparse
import base64
import collections
import json
import statistics
import threading

import numpy as np
import requests

N_LAYERS, TOP_K = 30, 8


def routed(url, ids, decode_len):
    payload = {
        "input_ids": ids,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": decode_len, "ignore_eos": True},
        "return_routed_experts": True,
        "routed_experts_start_len": len(ids),
    }
    r = requests.post(f"{url}/generate", json=payload, timeout=1800)
    r.raise_for_status()
    flat = np.frombuffer(base64.b64decode(r.json()["meta_info"]["routed_experts"]), dtype=np.int32)
    return flat.reshape(-1, N_LAYERS, TOP_K)


def run_group(url, prompts, decode_len):
    out = [None] * len(prompts)

    def work(i):
        out[i] = routed(url, prompts[i], decode_len)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(prompts))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--groups", type=int, default=3)
    ap.add_argument("--decode-len", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompts = [p["input_ids"] for p in json.load(open(args.prompts))]
    distinct_steps = []  # [layers] per step
    runs = collections.Counter()
    for g in range(args.groups):
        idx = [(g * args.batch + j) % len(prompts) for j in range(args.batch)]
        requests.post(f"{args.url}/flush_cache", timeout=60)
        routes = run_group(args.url, [prompts[i] for i in idx], args.decode_len)
        steps = min(r.shape[0] for r in routes)
        for k in range(steps):
            stacked = np.concatenate([r[k] for r in routes], axis=1)  # [layers, B*top_k]
            per_layer = []
            for layer in range(N_LAYERS):
                _, counts = np.unique(stacked[layer], return_counts=True)
                per_layer.append(len(counts))
                runs.update(counts.tolist())
            distinct_steps.append(per_layer)

    arr = np.array(distinct_steps, dtype=float)
    total_runs = sum(runs.values())
    out = dict(
        batch=args.batch,
        groups=args.groups,
        decode_steps=len(distinct_steps),
        distinct_experts_mean=float(arr.mean()),
        distinct_experts_by_layer=arr.mean(axis=0).round(2).tolist(),
        uniform_routing_expectation=128 * (1 - (1 - TOP_K / 128) ** args.batch),
        mean_run_length=sum(k * v for k, v in runs.items()) / total_runs,
        run_length_hist={int(k): v for k, v in sorted(runs.items())},
    )
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "distinct_experts_by_layer"}))


if __name__ == "__main__":
    main()
