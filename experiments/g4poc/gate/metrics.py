"""Decision metrics of the study, from one replay's request records.

Primary (fixed with the user, 2026-10-04): the highest in-flight request concurrency
per GPU whose E2E p90 meets the SLO, and the goodput at that point, reported at
several SLOs (capacity_at_slos()). Every result separates two layers:
(a) in-flight requests: time-average number of requests between send and reply;
(b) sessions: time-average number of live sessions, idle think time included.

- E2E latency per request (send to full non-streaming reply): p50/p90/p99 over the
  requests sent inside the measurement window; the implied per-request decode rate
  is output tokens / E2E.
- Throughput: output and total (prompt + output) tokens of the requests that
  finished inside the window, per second of window, per GPU.
- Cost: $/1M tokens = GPU $/hr / (tok/s x 3600) x 1e6 on both bases (output, total),
  at every price in config.GPU_PRICES_USD_PER_HR, never at one price.
- Prefix-cache hit rate: cached prompt tokens / prompt tokens of the window's requests.
- KV retractions: the server's retraction counter (or log lines) over the window.
"""

import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from gate import config


def percentile(xs: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    pos = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summarize(records: Sequence[Dict], warmup_s: float, window_s: float, n_gpus: int = 1) -> Dict:
    """Window metrics of one replay (records as loadgen.RequestRecord dicts, times relative to run start)."""
    return summarize_legs([records], warmup_s, window_s, n_gpus)


def summarize_legs(legs: Sequence[Sequence[Dict]], warmup_s: float, window_s: float, n_gpus: int = 1) -> Dict:
    """Window metrics pooled over replays of one load: each leg's own window, total time n x window."""
    w0, w1 = warmup_s, warmup_s + window_s
    sent: List[Dict] = []
    finished: List[Dict] = []
    for records in legs:
        sent += [r for r in records if w0 <= r["t_send"] < w1]
        finished += [r for r in records if r["ok"] and w0 <= r["t_done"] < w1]
    ok_sent = [r for r in sent if r["ok"]]
    e2e = [r["t_done"] - r["t_send"] for r in ok_sent]
    out_tok = sum(r["output_tokens"] for r in finished)
    prompt_tok = sum(r["prompt_tokens"] for r in finished)
    uncached_tok = sum(r["prompt_tokens"] - r["cached_tokens"] for r in finished)
    lag = [r["t_send"] - r["t_due"] for r in sent]
    total_s = window_s * len(legs)
    occupancy = [occupancy_in_window([(r["t_send"], r["t_done"]) for r in records], w0, w1) for records in legs]
    sessions = [occupancy_in_window(session_spans(records), w0, w1) for records in legs]
    per_req_rate = [r["output_tokens"] / (r["t_done"] - r["t_send"]) for r in ok_sent if r["t_done"] > r["t_send"]]
    return {
        "n_sent": len(sent),
        "n_ok": len(ok_sent),
        "n_failed": len(sent) - len(ok_sent),
        "n_finished_in_window": len(finished),
        "e2e_p50_s": percentile(e2e, 50),
        "e2e_p90_s": percentile(e2e, 90),
        "e2e_p99_s": percentile(e2e, 99),
        "e2e_mean_s": sum(e2e) / len(e2e) if e2e else float("nan"),
        "output_tok_s_per_gpu": out_tok / total_s / n_gpus,
        "total_tok_s_per_gpu": (out_tok + prompt_tok) / total_s / n_gpus,
        "uncached_prefill_tok_s_per_gpu": uncached_tok / total_s / n_gpus,
        "requests_per_s": len(finished) / total_s,
        "prefix_cache_hit_rate": hit_rate(ok_sent),
        "mean_prompt_tokens": sum(r["prompt_tokens"] for r in ok_sent) / max(len(ok_sent), 1),
        "mean_output_tokens": sum(r["output_tokens"] for r in ok_sent) / max(len(ok_sent), 1),
        "client_lag_p99_s": percentile(lag, 99),
        "inflight_mean": sum(o["mean"] for o in occupancy) / len(legs),
        "inflight_max": max(o["max"] for o in occupancy),
        "sessions_active_mean": sum(o["mean"] for o in sessions) / len(legs),
        "per_request_out_tok_s_p50": percentile(per_req_rate, 50),
        "per_request_out_tok_s_p10": percentile(per_req_rate, 10),
        "window_s": total_s,
        "n_gpus": n_gpus,
    }


def hit_rate(records: Iterable[Dict]) -> float:
    cached = prompt = 0
    for r in records:
        cached += r["cached_tokens"]
        prompt += r["prompt_tokens"]
    return cached / prompt if prompt else float("nan")


def meets_slo(summary: Dict, slo_p90_s: float = config.DEFAULT_SLO_E2E_P90_S) -> bool:
    return summary["n_failed"] == 0 and summary["n_ok"] > 0 and summary["e2e_p90_s"] <= slo_p90_s


def cost_per_mtok(gpu_usd_per_hr: float, tok_s_per_gpu: float) -> float:
    if tok_s_per_gpu <= 0:
        return float("inf")
    return gpu_usd_per_hr / (tok_s_per_gpu * 3600.0) * 1e6


def cost_table(tok_s_per_gpu: float, prices: Sequence[float] = config.GPU_PRICES_USD_PER_HR) -> Dict[str, float]:
    return {f"{p:.2f}": cost_per_mtok(p, tok_s_per_gpu) for p in prices}


def occupancy_in_window(spans: Sequence[Tuple[float, float]], t0: float, t1: float) -> Dict[str, float]:
    """Time-average and peak number of open [start, end) spans inside [t0, t1)."""
    events = []
    area = 0.0
    for a, b in spans:
        lo, hi = max(a, t0), min(b, t1)
        if hi > lo:
            area += hi - lo
        if b > t0 and a < t1:
            events += [(max(a, t0), 1), (min(b, t1), -1)]
    peak = cur = 0
    for _, d in sorted(events, key=lambda e: (e[0], e[1])):
        cur += d
        peak = max(peak, cur)
    return {"mean": area / (t1 - t0), "max": peak}


def session_spans(records: Sequence[Dict]) -> List[Tuple[float, float]]:
    """Per session (nonce): first send to last reply, think time included."""
    spans: Dict[str, List[float]] = {}
    for r in records:
        s = spans.setdefault(r["nonce"], [r["t_send"], r["t_done"]])
        s[0], s[1] = min(s[0], r["t_send"]), max(s[1], r["t_done"])
    return [(a, b) for a, b in spans.values()]


_POINT_KEYS = ("inflight_mean", "inflight_max", "sessions_active_mean", "e2e_p50_s", "e2e_p90_s", "e2e_p99_s",
               "output_tok_s_per_gpu", "total_tok_s_per_gpu", "per_request_out_tok_s_p50", "prefix_cache_hit_rate",
               "mean_prompt_tokens", "mean_output_tokens", "n_failed")


def capacity_at_slo(points: Sequence[Dict], slo_p90_s: float,
                    prices: Sequence[float] = config.GPU_PRICES_USD_PER_HR) -> Optional[Dict]:
    """The SLO-meeting point with the most in-flight requests per GPU, with its goodput and cost."""
    feasible = [p for p in points if meets_slo(p["summary"], slo_p90_s)]
    if not feasible:
        return None
    best = max(feasible, key=lambda p: (p["summary"]["inflight_mean"] / p["summary"]["n_gpus"],
                                        p["summary"]["output_tok_s_per_gpu"]))
    s = best["summary"]
    return {
        "load": best["load"],
        "inflight_per_gpu": s["inflight_mean"] / s["n_gpus"],
        "sessions_per_gpu": s["sessions_active_mean"] / s["n_gpus"],
        "goodput_output_tok_s_per_gpu": s["output_tok_s_per_gpu"],
        "goodput_total_tok_s_per_gpu": s["total_tok_s_per_gpu"],
        "e2e_p90_s": s["e2e_p90_s"],
        "per_request_out_tok_s_p50": s["per_request_out_tok_s_p50"],
        "prefix_cache_hit_rate": s["prefix_cache_hit_rate"],
        "usd_per_mtok_output": cost_table(s["output_tok_s_per_gpu"], prices),
        "usd_per_mtok_total": cost_table(s["total_tok_s_per_gpu"], prices),
    }


def capacity_at_slos(points: Sequence[Dict], slos: Sequence[float] = config.SLOS_E2E_P90_S,
                     prices: Sequence[float] = config.GPU_PRICES_USD_PER_HR) -> Dict:
    """The decision table of a sweep: per SLO, the capacity point; plus every point's row."""
    rows = [{"load": p["load"], **{k: p["summary"][k] for k in _POINT_KEYS},
             "meets_slo": {f"{slo:g}": meets_slo(p["summary"], slo) for slo in slos},
             "retractions": (p.get("retractions") or {}).get("requests")} for p in points]
    return {"slos_e2e_p90_s": list(slos), "prices_usd_per_gpu_hr": list(prices), "points": rows,
            "capacity": {f"{slo:g}": capacity_at_slo(points, slo, prices) for slo in slos}}


# ---------------------------------------------------------------------------
# Server-side counters (Prometheus text from /metrics, needs --enable-metrics).
# ---------------------------------------------------------------------------

_PROM_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)$")
COUNTERS = ("sglang:num_retracted_requests_total", "sglang:num_retracted_input_tokens_total",
            "sglang:num_retracted_output_tokens_total", "sglang:prompt_tokens_total",
            "sglang:generation_tokens_total", "sglang:cached_tokens_total", "sglang:num_requests_total",
            "sglang:num_aborted_requests_total", "sglang:evicted_tokens_total")
GAUGES = ("sglang:num_running_reqs", "sglang:num_queue_reqs", "sglang:token_usage", "sglang:full_token_usage",
          "sglang:swa_token_usage", "sglang:num_retracted_reqs", "sglang:gen_throughput")


def parse_prom(text: str) -> Dict[str, float]:
    """Metric name -> value summed over label sets (histogram series are skipped)."""
    out: Dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _PROM_LINE.match(line.strip())
        if not m:
            continue
        name = m.group(1)
        if name.endswith(("_bucket", "_sum", "_count")) and name not in COUNTERS:
            continue
        try:
            out[name] = out.get(name, 0.0) + float(m.group(3))
        except ValueError:
            continue
    return out


def prom_sample(text: str) -> Dict[str, float]:
    vals = parse_prom(text)
    return {k: vals[k] for k in GAUGES + COUNTERS if k in vals}


def counter_delta(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
    return {k: after[k] - before.get(k, 0.0) for k in COUNTERS if k in after}


def window_counter_delta(samples: Sequence[Dict], t0: float, t1: float) -> Optional[Dict[str, float]]:
    """Counter growth between the last sample at or before t0 and the first at or after t1."""
    before = [s for s in samples if s["t"] <= t0]
    after = [s for s in samples if s["t"] >= t1]
    if not before or not after:
        return None
    return counter_delta(before[-1], after[0])


_RETRACT_LOG = re.compile(r"KV cache pool is full\. Retract requests\. #retracted_reqs: (\d+)")


def retractions_from_log(text: str) -> Dict[str, int]:
    counts = [int(x) for x in _RETRACT_LOG.findall(text)]
    return {"events": len(counts), "requests": sum(counts)}


def gauge_summary(samples: Sequence[Dict], t0: float, t1: float) -> Dict[str, Dict[str, float]]:
    """Mean and max of each sampled gauge inside [t0, t1)."""
    inside = [s for s in samples if t0 <= s["t"] < t1]
    out = {}
    for k in GAUGES:
        xs = [s[k] for s in inside if k in s]
        if xs:
            out[k] = {"mean": sum(xs) / len(xs), "max": max(xs)}
    return out


def retractions(delta: Optional[Dict[str, float]], log_counts: Dict[str, int]) -> Dict:
    """Retracted requests over the window: the server counter when metrics are on, else the log."""
    if delta and "sglang:num_retracted_requests_total" in delta:
        return {"requests": int(delta["sglang:num_retracted_requests_total"]),
                "input_tokens": int(delta.get("sglang:num_retracted_input_tokens_total", 0)),
                "source": "metrics", "log_events": log_counts["events"]}
    return {"requests": log_counts["requests"], "source": "log", "log_events": log_counts["events"]}
