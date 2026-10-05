"""Fleet model: GPUs and $/1M tokens for S concurrent sessions that think between turns, by cache policy.

The in-flight sweeps have no think time. A real session alternates a turn (one request, E2E e) and an
idle think time T. By Little's law a GPU that runs a sweep point (r turns/s, mean E2E e) serves
n = r (T + e) sessions, n e / (T + e) of them in flight at a time. Per policy, the sessions one GPU
serves at think time T are the best SLO-meeting point's r (T + e), capped where idle histories must be
held:

(a) sticky, device-cached: every history stays in the device pool while its user thinks, so a GPU holds
    at most N_cap = full-attention pool tokens / tokens per session; turns run at the cached sweep's
    rate below its hit-rate cliff.
(b) drop idle: nothing is held between turns and every turn re-prefills its whole history (the same
    stack with the radix cache off). No cap.
(b') LRU-oversubscribed: the plain radix cache past its cliff (the cached sweep's points whose hit rate
    collapsed), i.e. what (b) costs when the cache is left on and thrashes. No cap.
(c) sticky, host-tier (HiCache): idle histories live in host RAM and load back on the next turn; a GPU
    holds at most host-pool tokens / tokens per session, for each host RAM size per GPU. Rates from a
    HiCache in-flight sweep. Pending HiCache exactness (multi-turn load-back is not yet bit-exact).

GPUs = ceil(S / n); $/1M output tokens = price x GPUs / (S x O / (T + e) x 3600) x 1e6 (O: output tokens
per turn). A capped policy stops growing with T while (b) keeps growing, so (b) wins past a crossover T*.
T = 0 is the reading where S counts requests in flight rather than sessions.

  python compute/fleet_model.py --cached <sweep.json> --nocache <sweep.json> [--hicache <sweep.json>] \
      [--host-gb 12,32,64,128] [--sessions 2200] [--slo 6,10,15] [--out fleet.json]
"""

import argparse
import json
import math
from typing import Dict, List, Optional, Sequence

PRICES_USD_PER_HR = (0.40, 0.70, 1.00, 1.50)
THINK_S = (0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300)
# A cached-sweep point is past the cliff when its hit rate has collapsed below this share of the sweep's best
# (base: 0.22 at 16 in flight against 0.76; mem-final: 0.11 at 28 against 0.76, while 20 and 24 hold 0.66/0.61).
BELOW_CLIFF_HIT_SHARE = 0.5
# PC3's bs3 HiCache run (sweep-mem-hc-20261005-145932, --hicache-size 12): "Allocating full hierarchical KV
# host pool: 318445 tokens, 3.26 GB" -- full-layer tokens per GB of --hicache-size (the rest of the size
# holds sliding-window KV in the device pools' ratio).
HOST_FULL_TOKENS_PER_GB = 318445 / 12


def points(sweep: Dict) -> List[Dict]:
    out = []
    for p in sweep["points"]:
        s = p["summary"]
        if s.get("n_failed", 0) or p.get("abandoned", 0) or not s.get("requests_per_s"):
            continue
        out.append({"inflight": s["inflight_mean"], "turns_per_s": s["requests_per_s"], "e2e_mean_s": s["e2e_mean_s"],
                    "e2e_p90_s": s["e2e_p90_s"], "output_tokens": s["mean_output_tokens"],
                    "prompt_tokens": s["mean_prompt_tokens"], "hit": s["prefix_cache_hit_rate"]})
    return out


def split_at_cliff(pts: Sequence[Dict]):
    top = max(p["hit"] for p in pts)
    below = [p for p in pts if p["hit"] >= BELOW_CLIFF_HIT_SHARE * top]
    return below, [p for p in pts if p["hit"] < BELOW_CLIFF_HIT_SHARE * top]


def tokens_per_session(pts: Sequence[Dict]) -> float:
    return max(p["prompt_tokens"] + p["output_tokens"] for p in pts)


def best(pts: Sequence[Dict], slo_s: float, think_s: float, cap: Optional[float] = None) -> Optional[Dict]:
    """The SLO-meeting point that serves the most sessions at this think time (ties: the lower E2E)."""
    ok = [p for p in pts if p["e2e_p90_s"] <= slo_s]
    serve = lambda p: min(cap, p["turns_per_s"] * (think_s + p["e2e_mean_s"])) if cap else \
        p["turns_per_s"] * (think_s + p["e2e_mean_s"])
    return max(ok, key=lambda p: (serve(p), -p["e2e_mean_s"]), default=None)


def sessions_per_gpu(p: Dict, think_s: float, cap: Optional[float]) -> float:
    n = p["turns_per_s"] * (think_s + p["e2e_mean_s"])
    return min(cap, n) if cap else n


def policy_row(p: Optional[Dict], think_s: float, sessions: int, cap: Optional[float] = None) -> Optional[Dict]:
    if p is None:
        return None
    per_gpu = sessions_per_gpu(p, think_s, cap)
    gpus = math.ceil(sessions / per_gpu)
    turns_per_s = sessions / (think_s + p["e2e_mean_s"])
    out_tok_s = turns_per_s * p["output_tokens"]
    total_tok_s = turns_per_s * (p["output_tokens"] + p["prompt_tokens"])
    return {
        "sessions_per_gpu": per_gpu, "gpus": gpus, "capped": bool(cap) and per_gpu == cap,
        "point_inflight": p["inflight"], "e2e_mean_s": p["e2e_mean_s"], "e2e_p90_s": p["e2e_p90_s"],
        "usd_per_mtok_output": {f"{c:.2f}": c * gpus / (out_tok_s * 3600) * 1e6 for c in PRICES_USD_PER_HR},
        "usd_per_mtok_total": {f"{c:.2f}": c * gpus / (total_tok_s * 3600) * 1e6 for c in PRICES_USD_PER_HR},
    }


def policies(cached: Dict, nocache: Dict, hicache: Optional[Dict] = None,
             host_gb: Sequence[float] = (12, 32, 64, 128)) -> List[Dict]:
    below, past = split_at_cliff(points(cached))
    per_session = tokens_per_session(below)
    n_cap = int(cached["server_info"]["max_total_num_tokens"]) / per_session
    out = [
        {"name": "a_sticky_device", "points": below, "cap": n_cap},
        {"name": "b_drop_idle", "points": points(nocache), "cap": None},
        {"name": "b_lru_oversubscribed", "points": past, "cap": None},
    ]
    if hicache is not None:
        hc = points(hicache)
        hc_per_session = tokens_per_session(hc)
        for gb in host_gb:
            out.append({"name": f"c_sticky_host_{gb:g}gb", "points": hc,
                        "cap": max(n_cap, HOST_FULL_TOKENS_PER_GB * gb / hc_per_session),
                        "pending": "HiCache exactness (multi-turn load-back not yet bit-exact)"})
    return out


def crossover(capped: Dict, other: Dict, slo_s: float, step_s: float = 0.25, max_s: float = 3600.0) -> Optional[float]:
    """The shortest think time at which `other` serves at least as many sessions per GPU as `capped`."""
    t = 0.0
    while t <= max_s:
        a, b = best(capped["points"], slo_s, t, capped["cap"]), best(other["points"], slo_s, t, other["cap"])
        if a is not None and b is not None and \
                sessions_per_gpu(b, t, other["cap"]) >= sessions_per_gpu(a, t, capped["cap"]):
            return t
        t += step_s
    return None


def model(pols: List[Dict], sessions: int = 2200, slo_s: float = 10.0, think: Sequence[float] = THINK_S) -> Dict:
    rows = [{"think_s": t, **{p["name"]: policy_row(best(p["points"], slo_s, t, p["cap"]), t, sessions, p["cap"])
                              for p in pols}} for t in think]
    drop = next(p for p in pols if p["name"] == "b_drop_idle")
    return {
        "sessions": sessions, "slo_s": slo_s,
        "caps_sessions_per_gpu": {p["name"]: p["cap"] for p in pols},
        "pending": {p["name"]: p["pending"] for p in pols if p.get("pending")},
        "crossover_vs_drop_idle_s": {p["name"]: crossover(p, drop, slo_s) for p in pols if p["cap"]},
        "rows": rows,
    }


def _fmt(r: Optional[Dict]) -> str:
    if r is None:
        return "        n/a        "
    return f"{r['sessions_per_gpu']:6.1f}{'*' if r['capped'] else ' '} {r['gpus']:5d} {r['usd_per_mtok_output']['0.70']:6.3f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cached", required=True, help="sweep.json with the prefix cache on")
    p.add_argument("--nocache", required=True, help="sweep.json of the same stack with --disable-radix-cache")
    p.add_argument("--hicache", help="sweep.json of a HiCache stack")
    p.add_argument("--host-gb", default="12,32,64,128")
    p.add_argument("--sessions", type=int, default=2200)
    p.add_argument("--slo", default="6,10,15")
    p.add_argument("--out")
    args = p.parse_args()
    load = lambda path: json.load(open(path)) if path else None
    pols = policies(load(args.cached), load(args.nocache), load(args.hicache),
                    [float(x) for x in args.host_gb.split(",")])
    res = {"inputs": {"cached": args.cached, "nocache": args.nocache, "hicache": args.hicache},
           "by_slo": {s: model(pols, args.sessions, float(s)) for s in args.slo.split(",")}}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    names = [q["name"] for q in pols]
    for s, m in res["by_slo"].items():
        print(f"\nSLO p90 <= {s} s, {args.sessions} sessions; caps (sessions/GPU): "
              + ", ".join(f"{k} {v:.1f}" for k, v in m["caps_sessions_per_gpu"].items() if v))
        print("crossover think time vs drop-idle (s): " + json.dumps(m["crossover_vs_drop_idle_s"]))
        print("think s | " + " | ".join(f"{n[:21]:^21}" for n in names) + "   (sess/GPU, GPUs, $/1M out @0.70)")
        for r in m["rows"]:
            print(f"{r['think_s']:7g} | " + " | ".join(_fmt(r[n]) for n in names))


if __name__ == "__main__":
    main()
