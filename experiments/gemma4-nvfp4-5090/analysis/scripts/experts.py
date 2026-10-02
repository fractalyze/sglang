"""Distinct routed experts touched per MoE layer per decode step.

Reads dumps from a server launched with
--expert-distribution-recorder-mode per_token (SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR
set) after drive.py ran B diverse streams. Only decode passes whose token count
equals B are kept, so mixed prefill/decode steps do not pollute the statistic.

Usage: python experts.py --batch 8 dump.pt [dump.pt ...]
"""

import argparse
import collections
import json
import statistics

import torch

DECODE_MODE = 2  # ForwardMode.DECODE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("dumps", nargs="+")
    args = ap.parse_args()

    distinct_per_layer = collections.defaultdict(list)
    run_hist = collections.Counter()
    n_pass = 0
    for path in args.dumps:
        data = torch.load(path, weights_only=False)
        for rec in data["records"]:
            if rec.get("forward_mode") != DECODE_MODE:
                continue
            ids = rec["topk_ids_of_layer"]  # [layers, tokens, topk]
            if ids.shape[1] != args.batch:
                continue
            n_pass += 1
            for layer in range(ids.shape[0]):
                flat = ids[layer].flatten()
                flat = flat[flat >= 0]
                counts = torch.bincount(flat, minlength=128)
                used = counts[counts > 0]
                distinct_per_layer[layer].append(int(used.numel()))
                run_hist.update(used.tolist())

    per_layer = {l: statistics.mean(v) for l, v in sorted(distinct_per_layer.items())}
    overall = statistics.mean(per_layer.values())
    total_runs = sum(run_hist.values())
    mean_run = sum(k * v for k, v in run_hist.items()) / total_runs
    out = dict(
        batch=args.batch,
        decode_passes=n_pass,
        distinct_experts_mean=overall,
        distinct_experts_min_layer=min(per_layer.values()),
        distinct_experts_max_layer=max(per_layer.values()),
        mean_run_length=mean_run,
        run_length_hist={k: run_hist[k] for k in sorted(run_hist)},
        per_layer=per_layer,
    )
    print(json.dumps({k: v for k, v in out.items() if k != "per_layer"}, indent=1))
    if args.json_out:
        json.dump(out, open(args.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
