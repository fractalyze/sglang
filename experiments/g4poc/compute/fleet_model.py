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

Storage (PC4, tree a0491db764, 10-05/06): under write_through the host pool is an inclusive mirror of the device
(host eviction removes only nodes already gone from the device), so the distinct histories one GPU holds are the
larger of what the device and the host pool hold. PC4 measured the host cost of an idle session at 12 GB (final-hc,
think30 x 48, bs3): ~0.23 GB, one third full-layer KV and two thirds sliding-window KV (1.4-1.8K window tokens at
102 KB/token: node-granular windows, the dead reply leaf, chunk-boundary windows). That is ~52 sessions at 12 GB
with perfect packing (~44 with ended-session garbage), and the measured hit rate collapses there. The sizing rule
(host GB per GPU ~ 0.23 x sessions) is that per-session cost, measured at 12 GB, extrapolated linearly. There is
no config lever (the full/SWA host split is near balance); code levers, open: exclusive tiering for hybrid SWA
(~+50% distinct capacity) and not caching decode-output tokens, which Gemma-4's template never reuses (~10-15% of
the SWA host share).

Measured session loads (`--session-point`) give the capacity that holds today: per config and think time, the most
sessions per GPU that meet the SLO (the largest measured point, and the interpolated point where p90 = SLO).

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
# Device tokens a stored session occupies in the radix tree: its history plus the dead reply leaves of earlier turns
# (the next prompt re-renders the reply, so the generated tokens stay as a branch) and ended sessions not yet aged out.
TOKENS_PER_STORED_SESSION = 6500
# Host pool per idle session under write_through, measured (PC4, final-hc at 12 GB, think30 x 48, bs3).
HOST_GB_PER_SESSION = 0.23


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


def storage_cap(device_tokens: float, host_gb: float) -> float:
    """Histories one GPU holds: the host pool mirrors the device (write_through), so the larger of the two."""
    return max(device_tokens / TOKENS_PER_STORED_SESSION, host_gb / HOST_GB_PER_SESSION)


def host_gb_needed(compute_sessions: float, device_tokens: float) -> float:
    """Host pool per GPU at which storage reaches the compute bound (0 if the device alone holds the sessions)."""
    if compute_sessions <= device_tokens / TOKENS_PER_STORED_SESSION:
        return 0.0
    return compute_sessions * HOST_GB_PER_SESSION


def policies(cached: Dict, nocache: Optional[Dict] = None, hicache: Optional[Dict] = None,
             host_gb: Sequence[float] = HOST_GB, hicache_points: Optional[str] = None) -> List[Dict]:
    below, past = split_at_cliff(points(cached))
    n_cap = int(cached["server_info"]["max_total_num_tokens"]) / TOKENS_PER_STORED_SESSION
    out = [{"name": "a_sticky_device", "points": below, "cap": n_cap}]
    if nocache is not None:
        out.append({"name": "b_drop_idle", "points": points(nocache), "cap": None})
    out.append({"name": "b_lru_oversubscribed", "points": past, "cap": None})
    if hicache is not None or hicache_points:
        hc = points(hicache) if hicache is not None else points_from_triples(hicache_points, below)
        hc_device = int((hicache or cached)["server_info"]["max_total_num_tokens"])
        for gb in host_gb:
            out.append({"name": f"c_sticky_host_{gb:g}gb", "points": hc, "host_gb": gb,
                        "device_tokens": hc_device, "tokens_per_session": TOKENS_PER_STORED_SESSION,
                        "cap": storage_cap(hc_device, gb)})
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
                               "host_gb_needed": host_gb_needed(n, h["device_tokens"]),
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


def measured_capacity(specs: Sequence[str], slo_s: float, sessions: int) -> List[Dict]:
    """Sessions per GPU meeting the SLO from measured think-time loads, per (config, think time).

    Each spec is "config:mean_think_s:turns_per_session:sessions:out_tok_s:p90_s:hit". Conservative: the largest
    measured point meeting the SLO. Interpolated: where p90 crosses the SLO between that point and the next one
    (linear in sessions, tok/s interpolated the same way). The effective think per turn is mean_think x
    (1 - 1/turns_per_session) (a closed population; turn 0 has no think time).
    """
    groups: Dict[tuple, List[Dict]] = {}
    for spec in specs:
        cfg, think, turns, n, tok_s, p90, hit = spec.split(":")
        groups.setdefault((cfg, float(think), float(turns)), []).append(
            {"sessions": float(n), "out_tok_s": float(tok_s), "p90_s": float(p90), "hit": float(hit)})
    out = []
    for (cfg, think, turns), pts in groups.items():
        pts.sort(key=lambda q: q["sessions"])
        ok = [q for q in pts if q["p90_s"] <= slo_s]
        row = {"config": cfg, "think_s": think, "think_eff_s": think * (1 - 1 / turns), "slo_s": slo_s,
               "points": pts}
        if ok:
            cons = ok[-1]
            nxt = next((q for q in pts if q["sessions"] > cons["sessions"]), None)
            if nxt is not None and nxt["p90_s"] > slo_s:
                f = (slo_s - cons["p90_s"]) / (nxt["p90_s"] - cons["p90_s"])
                interp = {"sessions": cons["sessions"] + f * (nxt["sessions"] - cons["sessions"]),
                          "out_tok_s": cons["out_tok_s"] + f * (nxt["out_tok_s"] - cons["out_tok_s"])}
            else:
                interp = {"sessions": cons["sessions"], "out_tok_s": cons["out_tok_s"]}
            for name, q in (("conservative", cons), ("interpolated", interp)):
                gpus = math.ceil(sessions / q["sessions"])
                # $/1M from the per-GPU output rate at that load (each GPU runs the measured point).
                row[name] = {"sessions_per_gpu": q["sessions"], "out_tok_s_per_gpu": q["out_tok_s"],
                             "gpus_for_sessions": gpus,
                             "usd_per_mtok_output": {f"{c:.2f}": c / (q["out_tok_s"] * 3600) * 1e6
                                                     for c in PRICES_USD_PER_HR}}
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Retention model (PC4's SW1, 10-06): under write_through a host pool keeps an idle session's cache for a roughly
# fixed time, retention = host GB / aggregate write rate, the sliding-window share binding first. A returning turn
# hits only if its idle gap (think + E2E) is shorter than the retention. Measured at 12 GB, think30: SWA retention
# ~25 s at 48 sessions and ~12 s at 72 (the misses' re-prefills add writes, so retention falls faster than 1/n).
# ---------------------------------------------------------------------------
THINK_MEDIAN_S, THINK_MEAN_S, THINK_CLIP_S = 15.0, 17.9, (2.0, 120.0)  # the session file's think time (WORKLOAD.md)
THINK_SIGMA = math.sqrt(2 * math.log(THINK_MEAN_S / THINK_MEDIAN_S))


def think_cdf(t_s: float, scale: float) -> float:
    """P(think x scale <= t) for the session file's lognormal think time, clipped before scaling."""
    y = t_s / scale
    if y < THINK_CLIP_S[0]:
        return 0.0
    if y >= THINK_CLIP_S[1]:
        return 1.0
    return 0.5 * (1 + math.erf((math.log(y) - math.log(THINK_MEDIAN_S)) / (THINK_SIGMA * math.sqrt(2))))


def write_gb_per_turn(sessions: float, mean_think_s: float, e2e_s: float, retention_s: float, host_gb: float) -> float:
    """Calibrates the per-turn host write from a measured retention: retention = host / (sessions x w / interval)."""
    return host_gb * (mean_think_s + e2e_s) / (sessions * retention_s)


def retention_s(sessions: float, mean_think_s: float, e2e_s: float, host_gb: float, w_gb: float) -> float:
    return host_gb * (mean_think_s + e2e_s) / (sessions * w_gb)


def retention_capacity(cached: Dict, nocache: Dict, host_gb: float, think_scale: float, slo_s: float,
                       w_gb: float, turns_per_session: float = 5.15, max_sessions: int = 2000) -> Optional[Dict]:
    """The most sessions per GPU meeting the SLO when the hit rate follows the retention model.

    The GPU's turn rate at the SLO interpolates by hit rate between its cached SLO point (final stack, hit h_max)
    and its no-cache SLO point (every turn re-prefills); a session count n is served if n / (think + E2E) does not
    exceed that rate. First turns (1 in turns_per_session) have no think time and never hit.
    """
    fastest = lambda pts: max((q for q in pts if q["e2e_p90_s"] <= slo_s), key=lambda q: q["turns_per_s"], default=None)
    c, b = fastest(points(cached)), fastest(points(nocache))
    if c is None or b is None:
        return None
    h_max = c["hit"]
    mean_think = think_scale * THINK_MEAN_S * (1 - 1 / turns_per_session)
    best_n = None
    for n in range(1, max_sessions + 1):
        e = c["e2e_mean_s"]
        for _ in range(20):  # the hit rate and E2E depend on each other: iterate to a fixed point
            r_ret = retention_s(n, mean_think, e, host_gb, w_gb)
            hit = h_max * think_cdf(max(r_ret - e, 0.0), think_scale)
            f = hit / h_max if h_max else 0.0
            rate = 1 / (f / c["turns_per_s"] + (1 - f) / b["turns_per_s"])
            e_new = f * c["e2e_mean_s"] + (1 - f) * b["e2e_mean_s"]
            if abs(e_new - e) < 1e-3:
                break
            e = e_new
        if n / (mean_think + e) <= rate:
            best_n = {"sessions_per_gpu": n, "hit": hit, "retention_s": r_ret, "turns_per_s": n / (mean_think + e),
                      "e2e_mean_s": e, "mean_think_s": mean_think, "output_tokens": c["output_tokens"]}
        else:
            break
    return best_n


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
    p.add_argument("--retention-cal", default="48:30:25:3.5:12",
                   help="sessions:mean_think_s:retention_s:e2e_s:host_gb measured (PC4 SW1: SWA retention 25 s at 48)")
    p.add_argument("--session-point", action="append", default=[],
                   help="config:mean_think_s:turns_per_session:sessions:out_tok_s:p90_s:hit (repeatable)")
    p.add_argument("--out")
    args = p.parse_args()
    load = lambda path: json.load(open(path)) if path else None
    pols = policies(load(args.cached), load(args.nocache), load(args.hicache),
                    [float(x) for x in args.host_gb.split(",")], args.hicache_points)
    res = {"inputs": {"cached": args.cached, "nocache": args.nocache, "hicache": args.hicache,
                      "hicache_points": args.hicache_points},
           "by_slo": {s: model(pols, args.sessions, float(s)) for s in args.slo.split(",")}}
    res["validation"] = {s: validate(args.measured, pols, float(s)) for s in args.slo.split(",")}
    res["measured_capacity"] = {s: measured_capacity(args.session_point, float(s), args.sessions)
                                for s in args.slo.split(",")}
    if args.nocache and args.hicache:
        n, t, r, e, h = (float(x) for x in args.retention_cal.split(":"))
        w = write_gb_per_turn(n, t * (1 - 1 / 5.15), e, r, h)
        res["retention"] = {"w_gb_per_turn": w, "calibration": args.retention_cal, "rows": [
            {"host_gb": gb, "think_s": t_mean, "slo_s": float(s), **(retention_capacity(
                load(args.hicache), load(args.nocache), gb, t_mean / THINK_MEAN_S, float(s), w) or {})}
            for s in args.slo.split(",") for gb in [float(x) for x in args.host_gb.split(",")]
            for t_mean in (10.0, 20.0, 30.0, 60.0, 120.0)]}
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
        for mc in res["measured_capacity"][s]:
            for name in ("conservative", "interpolated"):
                if name in mc:
                    q = mc[name]
                    print(f"  measured {mc['config']} T{mc['think_s']:g} (eff {mc['think_eff_s']:.1f} s) {name}: "
                          f"{q['sessions_per_gpu']:.1f} sessions/GPU, {q['out_tok_s_per_gpu']:.0f} tok/s, "
                          f"{q['gpus_for_sessions']} GPUs, ${q['usd_per_mtok_output']['0.70']:.3f}/1M out @0.70")
        if "retention" in res:
            print(f"  retention model (w = {res['retention']['w_gb_per_turn']:.3f} GB/turn): host GB, mean think "
                  "-> sessions/GPU, hit, retention, GPUs for 2,200, $/1M out @0.70")
            for z in res["retention"]["rows"]:
                if z["slo_s"] == float(s) and "sessions_per_gpu" in z:
                    tok_s = z["turns_per_s"] * z["output_tokens"]
                    print(f"    {z['host_gb']:5g} GB, T{z['think_s']:g}: {z['sessions_per_gpu']:4d} sessions, hit "
                          f"{z['hit']:.2f}, R {z['retention_s']:5.1f} s, {math.ceil(args.sessions / z['sessions_per_gpu'])} GPUs, "
                          f"${0.70 / (tok_s * 3600) * 1e6:.3f}")
        for v in res["validation"][s]:
            print("  measured:", json.dumps({k: (round(x, 2) if isinstance(x, float) else x) for k, x in v.items()}))


if __name__ == "__main__":
    main()
