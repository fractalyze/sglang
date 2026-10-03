"""W14: decompose an existing `gate run` under speculation into per-round time x acceptance.

No GPU: it reads each leg's server log (--decode-log-interval 1 logs every verify
round with #running-req) and leg.json. Episodes are split at a prefill that follows
a decode. A leg's log runs warm-up (W8, W1, W32), then the timed W8 reps, the
timed W1 reps, then W32 and fidelity. The timed W1 block is the run of
len(W1 reps) single-stream episodes, and the timed W8 block is the len(W8 reps)
episodes before it.

Per workload and leg:
  stream_rounds  W1: decode lines; W8: sum of #running-req over decode lines
  tau            stream tokens after the first / stream_rounds
  round_ms       summed stream decode time / stream_rounds

Each arm decodes its own text here, so round time still carries text-dependent
routing; gate spec-run's replay removes that too. This is the estimate for runs that
cannot be rerun (bs3) and a cross-check of the replay for the others.

  python -m trials.spec.legs_decompose RUN_DIR [RUN_DIR ...] --out FILE
"""

import argparse
import json
import math
import os
import re
from typing import Dict, List

from gate import stats

_DECODE = re.compile(r"Decode batch, #running-req: (\d+)")


def episodes(log_path: str) -> List[List[int]]:
    """Per episode, the #running-req of each decode line."""
    eps: List[List[int]] = []
    last = None
    with open(log_path, errors="replace") as f:
        for ln in f:
            if "Prefill batch" in ln:
                if last != "P":
                    eps.append([])
                last = "P"
            elif eps and "Decode batch" in ln:
                eps[-1].append(int(_DECODE.search(ln).group(1)))
                last = "D"
    return eps


def timed_blocks(eps: List[List[int]], n_w1: int, n_w8: int, w8_concurrency: int) -> Dict[str, List[List[int]]]:
    single = [bool(e) and max(e) == 1 for e in eps]
    for i in range(len(eps) - n_w1 + 1):
        if all(single[i:i + n_w1]) and (i + n_w1 == len(eps) or not single[i + n_w1]) and (i == 0 or not single[i - 1]):
            w8 = eps[i - n_w8:i]
            if len(w8) == n_w8 and all(max(e) == w8_concurrency for e in w8):
                return {"W1": eps[i:i + n_w1], "W8": w8}
    raise ValueError("no timed W8 + W1 block in the log")


def leg_decomposition(leg_dir: str, workloads: Dict[str, Dict]) -> Dict:
    leg = json.load(open(os.path.join(leg_dir, "leg.json")))
    # A retried leg's leg.json sits in the first attempt's directory, its log in the last one's.
    log_dir = f"{leg_dir}-retry{len(leg['attempts'])}" if leg.get("attempts") else leg_dir
    blocks = timed_blocks(episodes(os.path.join(log_dir, "server.log")), workloads["W1"]["reps"],
                          workloads["W8"]["reps"], workloads["W8"]["concurrency"])
    out = {}
    for wl in ("W1", "W8"):
        streams = [s for r in leg["workloads"][wl] for s in r["streams"]]
        decode_s = sum(s["e2e_s"] - s["ttft_s"] for s in streams)
        tokens = sum(s["output_tokens"] - 1 for s in streams)
        rounds = sum(sum(e) for e in blocks[wl])
        out[wl] = {"decode_s": decode_s, "tokens": tokens, "stream_rounds": rounds, "tau": tokens / rounds,
                   "round_ms": 1e3 * decode_s / rounds}
    return out


def run_decomposition(run_dir: str) -> Dict:
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    workloads = {w["name"]: w for w in meta["config"]["workloads"]}
    per = {role: [leg_decomposition(os.path.join(run_dir, f"pair{k}-{role}"), workloads)
                  for k in range(meta["n_pairs"])]
           for role in ("control", "candidate")}
    out = {"run": os.path.basename(run_dir.rstrip("/")), "control": meta["control"]["name"],
           "candidate": meta["candidate"]["name"], "per_leg": per}
    for wl in ("W1", "W8"):
        def tot(role, key):
            return sum(leg[wl][key] for leg in per[role])

        time_gain = tot("control", "decode_s") / tot("candidate", "decode_s")
        tau_c = tot("control", "tokens") / tot("control", "stream_rounds")
        tau_k = tot("candidate", "tokens") / tot("candidate", "stream_rounds")
        round_c = tot("control", "decode_s") / tot("control", "stream_rounds")
        round_k = tot("candidate", "decode_s") / tot("candidate", "stream_rounds")
        logs = [math.log(c[wl]["round_ms"] / k[wl]["round_ms"]) for c, k in zip(per["control"], per["candidate"])]
        out[wl] = {"decode_time_gain": time_gain,
                   "round_time_gain": round_c / round_k, "round_time_ci95": stats.pair_ci95(logs),
                   "tau_control": tau_c, "tau_candidate": tau_k, "tau_ratio": tau_k / tau_c,
                   "round_ms_control": 1e3 * round_c, "round_ms_candidate": 1e3 * round_k}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = [run_decomposition(r) for r in args.runs]
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    for r in res:
        print(r["run"], r["control"], "->", r["candidate"])
        for wl in ("W1", "W8"):
            w = r[wl]
            print(f"  {wl}: time x{w['decode_time_gain']:.4f} = round x{w['round_time_gain']:.4f} "
                  f"(CI {w['round_time_ci95']['low']:.4f}-{w['round_time_ci95']['high']:.4f}; "
                  f"{w['round_ms_control']:.3f} -> {w['round_ms_candidate']:.3f} ms) "
                  f"x tau {w['tau_ratio']:.4f} ({w['tau_control']:.3f} -> {w['tau_candidate']:.3f})")


if __name__ == "__main__":
    main()
