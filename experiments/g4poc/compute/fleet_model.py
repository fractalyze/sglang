"""Fleet model: GPUs and $/1M tokens for S concurrent sessions that think between turns, two cache policies.

The in-flight sweeps have no think time. A real session alternates a turn (one request, E2E e) and an
idle think time T. By Little's law a GPU that runs a sweep point (r turns/s, mean E2E e) serves
n = r (T + e) sessions, of which n e / (T + e) are in flight at a time.

(a) sticky routing, every history cached: a session's history stays in the device pool while it
    thinks, so a GPU holds at most N_cap histories (full-attention pool tokens / tokens per session);
    its turns run at the cached sweep's rate (points below the hit-rate cliff only).
    n_a(T) = min(N_cap, max over SLO-meeting points of r (T + e)).
(b) idle histories fall out: nothing is held between turns, and every turn re-prefills its whole
    history; the rate comes from a sweep with the radix cache off.
    n_b(T) = max over SLO-meeting points of r (T + e).

GPUs = ceil(S / n); $/1M output tokens = price x GPUs / (S x O / (T + e) x 3600) x 1e6 (O: output
tokens per turn). n_a stops growing once memory binds, n_b keeps growing with T, so (b) is cheaper
past a crossover think time T*. An LRU pool without sticky guarantees sits between the two.

  python compute/fleet_model.py --cached <sweep.json> --nocache <sweep.json> [--sessions 2200] [--slo 10]
"""

import argparse
import json
import math
from typing import Dict, List, Optional, Sequence

PRICES_USD_PER_HR = (0.40, 0.70, 1.00, 1.50)
THINK_S = (0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300)
# A cached-sweep point is below the cliff when its hit rate is at least this share of the sweep's best.
BELOW_CLIFF_HIT_SHARE = 0.9


def points(sweep: Dict) -> List[Dict]:
    out = []
    for p in sweep["points"]:
        s = p["summary"]
        if s.get("n_failed", 0) or p.get("abandoned", 0):
            continue
        out.append({"inflight": s["inflight_mean"], "turns_per_s": s["requests_per_s"], "e2e_mean_s": s["e2e_mean_s"],
                    "e2e_p90_s": s["e2e_p90_s"], "output_tokens": s["mean_output_tokens"],
                    "prompt_tokens": s["mean_prompt_tokens"], "hit": s["prefix_cache_hit_rate"]})
    return out


def full_pool_tokens(sweep: Dict) -> int:
    return int(sweep["server_info"]["max_total_num_tokens"])


def best(pts: Sequence[Dict], slo_s: float, think_s: float) -> Optional[Dict]:
    """The SLO-meeting point that serves the most sessions at this think time."""
    ok = [p for p in pts if p["e2e_p90_s"] <= slo_s]
    return max(ok, key=lambda p: p["turns_per_s"] * (think_s + p["e2e_mean_s"]), default=None)


def policy_row(p: Optional[Dict], think_s: float, sessions: int, cap: Optional[float] = None) -> Optional[Dict]:
    if p is None:
        return None
    per_gpu = p["turns_per_s"] * (think_s + p["e2e_mean_s"])
    memory_bound = cap is not None and cap < per_gpu
    if memory_bound:
        per_gpu = cap
    gpus = math.ceil(sessions / per_gpu)
    turns_per_s = sessions / (think_s + p["e2e_mean_s"])
    out_tok_s = turns_per_s * p["output_tokens"]
    total_tok_s = turns_per_s * (p["output_tokens"] + p["prompt_tokens"])
    return {
        "sessions_per_gpu": per_gpu, "gpus": gpus, "memory_bound": memory_bound,
        "point_inflight": p["inflight"], "e2e_mean_s": p["e2e_mean_s"], "e2e_p90_s": p["e2e_p90_s"],
        "usd_per_mtok_output": {f"{c:.2f}": c * gpus / (out_tok_s * 3600) * 1e6 for c in PRICES_USD_PER_HR},
        "usd_per_mtok_total": {f"{c:.2f}": c * gpus / (total_tok_s * 3600) * 1e6 for c in PRICES_USD_PER_HR},
    }


def model(cached: Dict, nocache: Dict, sessions: int = 2200, slo_s: float = 10.0,
          think: Sequence[float] = THINK_S) -> Dict:
    c_pts, n_pts = points(cached), points(nocache)
    top_hit = max(p["hit"] for p in c_pts)
    below_cliff = [p for p in c_pts if p["hit"] >= BELOW_CLIFF_HIT_SHARE * top_hit]
    per_session = max(p["prompt_tokens"] + p["output_tokens"] for p in below_cliff)
    n_cap = full_pool_tokens(cached) / per_session
    rows = []
    for t in think:
        a = policy_row(best(below_cliff, slo_s, t), t, sessions, cap=n_cap)
        b = policy_row(best(n_pts, slo_s, t), t, sessions)
        rows.append({"think_s": t, "sticky_cached": a, "drop_idle": b})
    return {"sessions": sessions, "slo_s": slo_s, "full_pool_tokens": full_pool_tokens(cached),
            "tokens_per_session": per_session, "cached_sessions_per_gpu_cap": n_cap,
            "crossover_think_s": crossover(below_cliff, n_pts, n_cap, slo_s), "rows": rows}


def crossover(below_cliff: Sequence[Dict], n_pts: Sequence[Dict], n_cap: float, slo_s: float,
              step_s: float = 0.25, max_s: float = 3600.0) -> Optional[float]:
    """The shortest think time at which dropping idle histories serves at least as many sessions per GPU."""
    t = 0.0
    while t <= max_s:
        a, b = best(below_cliff, slo_s, t), best(n_pts, slo_s, t)
        if a is not None and b is not None:
            n_a = min(n_cap, a["turns_per_s"] * (t + a["e2e_mean_s"]))
            if b["turns_per_s"] * (t + b["e2e_mean_s"]) >= n_a:
                return t
        t += step_s
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cached", required=True, help="sweep.json with the prefix cache on")
    p.add_argument("--nocache", required=True, help="sweep.json of the same stack with --disable-radix-cache")
    p.add_argument("--sessions", type=int, default=2200)
    p.add_argument("--slo", type=float, default=10.0)
    p.add_argument("--out")
    args = p.parse_args()
    with open(args.cached) as f:
        cached = json.load(f)
    with open(args.nocache) as f:
        nocache = json.load(f)
    res = model(cached, nocache, args.sessions, args.slo)
    res["inputs"] = {"cached": args.cached, "nocache": args.nocache}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    print(f"full pool {res['full_pool_tokens']} tokens / {res['tokens_per_session']:.0f} per session = "
          f"{res['cached_sessions_per_gpu_cap']:.1f} cached sessions per GPU; SLO p90 <= {args.slo:g} s; "
          f"crossover think time {res['crossover_think_s']} s")
    print("think s | sticky cached: sess/GPU GPUs $/1M out@0.70 | drop idle: sess/GPU GPUs $/1M out@0.70")
    for r in res["rows"]:
        a, b = r["sticky_cached"], r["drop_idle"]
        fa = (f"{a['sessions_per_gpu']:6.1f}{'*' if a['memory_bound'] else ' '} {a['gpus']:5d} "
              f"{a['usd_per_mtok_output']['0.70']:.3f}") if a else "n/a"
        fb = f"{b['sessions_per_gpu']:6.1f} {b['gpus']:5d} {b['usd_per_mtok_output']['0.70']:.3f}" if b else "n/a"
        print(f"{r['think_s']:7g} | {fa} | {fb}")


if __name__ == "__main__":
    main()
