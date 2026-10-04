"""Paired statistics for ABBA legs at a fixed offered load.

A leg is one server lifetime of one ref replaying the pair's plan. Pair k is
legs 2k and 2k+1 in ABBA order. Gains are oriented so that > 1 favours the
candidate: latency gains are control/candidate, throughput gains
candidate/control. The overall gain pools every leg's window requests per arm;
the per-pair gains give the noise (sigma of the log gain) and the 95% CI.
"""

import math
from typing import Dict, List, Sequence

from gate import config, metrics

# Metric -> (summary key, True when lower is better).
PAIRED_METRICS = {
    "e2e_p90_gain": ("e2e_p90_s", True),
    "e2e_p50_gain": ("e2e_p50_s", True),
    "e2e_p99_gain": ("e2e_p99_s", True),
    "output_tput_gain": ("output_tok_s_per_gpu", False),
}
# The ones a trial may decide on; every other one of these guards against regression.
DECIDING_METRICS = ("e2e_p90_gain", "e2e_p50_gain", "output_tput_gain")
DEFAULT_DECIDING_METRIC = "e2e_p90_gain"


def _gain(ctrl: float, cand: float, lower_better: bool) -> float:
    return ctrl / cand if lower_better else cand / ctrl


def pooled_summary(legs: Sequence[Dict]) -> Dict:
    """One arm's window metrics over all its legs' requests."""
    load = legs[0]["load"]
    return metrics.summarize_legs([leg["replay"]["records"] for leg in legs], load["warmup_s"], load["window_s"])


def gains(control: Dict, candidate: Dict) -> Dict[str, float]:
    return {m: _gain(control[k], candidate[k], lb) for m, (k, lb) in PAIRED_METRICS.items()}


def summarize_pairs(control: List[Dict], candidate: List[Dict]) -> Dict:
    """`control`/`candidate` are legs in pair order, each with "load", "summary", "replay" and "retractions"."""
    if len(control) != len(candidate):
        raise ValueError(f"unpaired legs: {len(control)} control vs {len(candidate)} candidate")
    pooled_c, pooled_k = pooled_summary(control), pooled_summary(candidate)
    overall = gains(pooled_c, pooled_k)
    pairs = [gains(c["summary"], k["summary"]) for c, k in zip(control, candidate)]
    log_sigma, ci95 = {}, {}
    for m in PAIRED_METRICS:
        logs = [math.log(p[m]) for p in pairs if p[m] > 0 and math.isfinite(p[m])]
        log_sigma[m] = _sample_std(logs)
        ci95[m] = pair_ci95(logs)
    return {"overall": overall, "pairs": pairs, "per_pair_log_sigma": log_sigma, "ci95": ci95,
            "n_pairs": len(pairs), "pooled": {"control": pooled_c, "candidate": pooled_k},
            "hit_rate": {"control": pooled_c["prefix_cache_hit_rate"], "candidate": pooled_k["prefix_cache_hit_rate"]},
            "retractions": {"control": [c["retractions"]["requests"] for c in control],
                            "candidate": [k["retractions"]["requests"] for k in candidate]}}


_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
         11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
         20: 2.086, 25: 2.060, 30: 2.042}


def _t975(df: int) -> float:
    if df in _T975:
        return _T975[df]
    return _T975[max(k for k in _T975 if k <= df)] if df < 30 else 1.96


def pair_ci95(log_gains: Sequence[float]) -> Dict[str, float]:
    """95% t-interval of the mean per-pair log gain, as gains, and its relative half-width."""
    if len(log_gains) < 2:
        return {"low": float("nan"), "high": float("nan"), "half_width": float("nan")}
    mean = sum(log_gains) / len(log_gains)
    half = _t975(len(log_gains) - 1) * _sample_std(log_gains) / math.sqrt(len(log_gains))
    return {"low": math.exp(mean - half), "high": math.exp(mean + half), "half_width": math.expm1(half)}


def _sample_std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mean = sum(xs) / len(xs)
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))


def promotion_bar(sigma: float) -> float:
    """Relative gain a candidate must exceed: max(3 sigma_noise, 1%)."""
    if sigma != sigma:
        sigma = 0.0
    return max(config.NOISE_SIGMAS * sigma, config.MIN_BAR)


def timing_verdict(summary: Dict, noise: Dict[str, float], decide_on: str = DEFAULT_DECIDING_METRIC) -> Dict:
    """Promote iff the deciding metric clears its bar and no other deciding metric regresses past its own."""
    if decide_on not in DECIDING_METRICS:
        raise ValueError(f"cannot decide on {decide_on!r}; choose one of {list(DECIDING_METRICS)}")
    overall = summary["overall"]
    bars = {m: promotion_bar(noise.get(m, 0.0)) for m in PAIRED_METRICS}
    checks = {f"{decide_on}_clears_bar": overall[decide_on] - 1.0 > bars[decide_on]}
    for m in DECIDING_METRICS:
        checks[f"{m}_no_regression"] = overall[m] - 1.0 > -bars[m]
    checks["enough_pairs"] = summary["n_pairs"] >= config.MIN_PAIRS
    ci95 = summary.get("ci95", {})
    return {
        "decided_on": decide_on,
        "bars": bars,
        "ci95_narrower_than_bar": {m: ci95[m]["half_width"] < bars[m] for m in ci95},
        "noise_calibrated": all(m in noise for m in DECIDING_METRICS),
        "checks": checks,
        "promote": all(checks.values()),
        "no_regression": all(v for k, v in checks.items() if k.endswith("_no_regression")),
    }
