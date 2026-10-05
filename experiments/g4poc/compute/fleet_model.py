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
    holds at most (device full-pool tokens + host-pool tokens) / tokens per session, for each host RAM size
    per GPU (the storage bound); rates from a HiCache in-flight sweep (the compute bound). Past the storage
    bound the pool thrashes and every turn re-prefills (measured on bs3: 120 sessions at 30 s think time on
    final-hc, hit rate 0.002, p90 24 s), so the storage bound is a hard cap at any SLO under ~20 s.

Sessions per GPU = min(compute bound, storage bound). `host_gb_needed` gives, per think time, the host pool per
GPU at which storage stops binding (the sizing rule).

GPUs = ceil(S / n); $/1M output tokens = price x GPUs / (S x O / (T + e) x 3600) x 1e6 (O: output tokens
per turn). A capped policy stops growing with T while (b) keeps growing, so (b) wins past a crossover T*.
T = 0 is the reading where S counts requests in flight rather than sessions.

  python compute/fleet_model.py --cached <sweep.json> --nocache <sweep.json> [--hicache <sweep.json>] \
      [--host-gb 12,24,48,96,128] [--measured ...] [--sessions 2200] [--slo 6,10,15] [--out fleet.json]
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
# PC3's bs3 HiCache run (sweep-mem-hc-20261005-145932, --hicache-size 12; the same on bs2): "Allocating full
# hierarchical KV host pool: 318445 tokens, 3.26 GB" -- full-layer tokens per GB of --hicache-size (the other
# 8.74 GB hold sliding-window KV in the device pools' ratio), i.e. ~37.7 KB of host pool per history token.
HOST_FULL_TOKENS_PER_GB = 318445 / 12
HOST_GB = (12, 24, 48, 96, 128)


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


def points_from_triples(spec: str, like: Sequence[Dict]) -> List[Dict]:
    """Points from "C:p90:out_tok_s,..." (a sweep reported only as in-flight count, E2E p90 and output tok/s).

    With C slots always in flight, Little's law gives turns/s = out tok/s / output tokens per turn and mean
    E2E = C / turns/s; tokens per turn and the hit rate are taken from `like` (the same workload's points).
    """
    out_tokens = sum(p["output_tokens"] for p in like) / len(like)
    prompt_tokens = sum(p["prompt_tokens"] for p in like) / len(like)
    pts = []
    for item in spec.split(","):
        c, p90, tok_s = (float(x) for x in item.split(":"))
        turns = tok_s / out_tokens
        pts.append({"inflight": c, "turns_per_s": turns, "e2e_mean_s": c / turns, "e2e_p90_s": p90,
                    "output_tokens": out_tokens, "prompt_tokens": prompt_tokens, "hit": None})
    return pts


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


def storage_cap(device_tokens: float, host_gb: float, per_session: float) -> float:
    """Histories one GPU can hold: its device full pool plus a host pool of `host_gb` (--hicache-size)."""
    return (device_tokens + HOST_FULL_TOKENS_PER_GB * host_gb) / per_session


def host_gb_needed(compute_sessions: float, device_tokens: float, per_session: float) -> float:
    """The host pool per GPU at which the storage bound reaches the compute bound."""
    return max(0.0, compute_sessions * per_session - device_tokens) / HOST_FULL_TOKENS_PER_GB


def policies(cached: Dict, nocache: Optional[Dict] = None, hicache: Optional[Dict] = None,
             host_gb: Sequence[float] = HOST_GB, hicache_points: Optional[str] = None) -> List[Dict]:
    below, past = split_at_cliff(points(cached))
    per_session = tokens_per_session(below)
    n_cap = int(cached["server_info"]["max_total_num_tokens"]) / per_session
    out = [{"name": "a_sticky_device", "points": below, "cap": n_cap}]
    if nocache is not None:
        out.append({"name": "b_drop_idle", "points": points(nocache), "cap": None})
    out.append({"name": "b_lru_oversubscribed", "points": past, "cap": None})
    if hicache is not None or hicache_points:
        hc = points(hicache) if hicache is not None else points_from_triples(hicache_points, below)
        hc_per_session = tokens_per_session(hc)
        hc_device = int((hicache or cached)["server_info"]["max_total_num_tokens"])
        for gb in host_gb:
            out.append({"name": f"c_sticky_host_{gb:g}gb", "points": hc, "host_gb": gb,
                        "device_tokens": hc_device, "tokens_per_session": hc_per_session,
                        "cap": storage_cap(hc_device, gb, hc_per_session)})
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
    drop = next((p for p in pols if p["name"] == "b_drop_idle"), None)
    host = [p for p in pols if p.get("host_gb") is not None]
    sizing = []
    if host:
        h = host[0]
        for t in think:
            b = best(h["points"], slo_s, t)
            if b is not None:
                n = sessions_per_gpu(b, t, None)
                sizing.append({"think_s": t, "compute_bound_sessions_per_gpu": n,
                               "host_gb_needed": host_gb_needed(n, h["device_tokens"], h["tokens_per_session"]),
                               "gpus_for_sessions": math.ceil(sessions / n)})
    return {
        "host_sizing": sizing,
        "sessions": sessions, "slo_s": slo_s,
        "caps_sessions_per_gpu": {p["name"]: p["cap"] for p in pols},
        "pending": {p["name"]: p["pending"] for p in pols if p.get("pending")},
        "crossover_vs_drop_idle_s": {p["name"]: crossover(p, drop, slo_s) for p in pols if p["cap"]} if drop else {},
        "rows": rows,
    }


def validate(measured: Sequence[str], pols: List[Dict], slo_s: float) -> List[Dict]:
    """Measured think-time session loads against the model's bounds for the same host pool.

    Each spec is "label:sessions:mean_think_s:turns_per_session:out_tok_s:p90_s:hit[:host_gb]" (host_gb 12 by
    default). A closed population starts a new session the moment one ends, and a session's first turn has no
    think time, so the mean think per turn is mean_think x (1 - 1/turns_per_session).
    """
    host = {p["host_gb"]: p for p in pols if p.get("host_gb") is not None}
    rows = []
    for spec in measured:
        parts = spec.split(":")
        label, (n, think, turns, tok_s, p90, hit) = parts[0], (float(x) for x in parts[1:7])
        gb = float(parts[7]) if len(parts) > 7 else 12.0
        t_eff = think * (1 - 1 / turns)
        row = {"label": label, "sessions": n, "think_eff_s": t_eff, "out_tok_s": tok_s, "p90_s": p90, "hit": hit}
        h = host.get(gb)
        if h is not None:
            b = best(h["points"], slo_s, t_eff)
            row.update({"host_gb": gb, "storage_bound": h["cap"],
                        "compute_bound": sessions_per_gpu(b, t_eff, None) if b else None,
                        "past_storage_bound": n > h["cap"]})
        rows.append(row)
    return rows


def _fmt(r: Optional[Dict]) -> str:
    if r is None:
        return "        n/a        "
    return f"{r['sessions_per_gpu']:6.1f}{'*' if r['capped'] else ' '} {r['gpus']:5d} {r['usd_per_mtok_output']['0.70']:6.3f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cached", required=True, help="sweep.json with the prefix cache on")
    p.add_argument("--nocache", help="sweep.json of the same stack with --disable-radix-cache (policy b)")
    p.add_argument("--hicache", help="sweep.json of a HiCache stack")
    p.add_argument("--hicache-points", help='instead of --hicache: "C:p90:out_tok_s,..." of a HiCache sweep')
    p.add_argument("--sessions", type=int, default=2200)
    p.add_argument("--slo", default="6,10,15")
    p.add_argument("--measured", action="append", default=[],
                   help="label:sessions:mean_think_s:turns_per_session:out_tok_s:p90_s:hit[:host_gb] (repeatable)")
    p.add_argument("--host-gb", default=",".join(f"{g:g}" for g in HOST_GB), help="host pool per GPU (--hicache-size)")
    p.add_argument("--out")
    args = p.parse_args()
    load = lambda path: json.load(open(path)) if path else None
    pols = policies(load(args.cached), load(args.nocache), load(args.hicache),
                    [float(x) for x in args.host_gb.split(",")], args.hicache_points)
    res = {"inputs": {"cached": args.cached, "nocache": args.nocache, "hicache": args.hicache,
                      "hicache_points": args.hicache_points},
           "by_slo": {s: model(pols, args.sessions, float(s)) for s in args.slo.split(",")}}
    res["validation"] = {s: validate(args.measured, pols, float(s)) for s in args.slo.split(",")}
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
        print("host sizing (HiCache): think s -> compute-bound sessions/GPU, host GB/GPU to hold them, GPUs")
        for z in m["host_sizing"]:
            print(f"  {z['think_s']:5g} s: {z['compute_bound_sessions_per_gpu']:6.1f} sessions/GPU, "
                  f"{z['host_gb_needed']:5.1f} GB host, {z['gpus_for_sessions']} GPUs")
        for v in res["validation"][s]:
            print("  measured:", json.dumps({k: (round(x, 2) if isinstance(x, float) else x) for k, x in v.items()}))


if __name__ == "__main__":
    main()
