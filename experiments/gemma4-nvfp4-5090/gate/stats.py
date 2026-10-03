"""Paired-timing statistics: ratio of sums, per-pair skew, noise and the bar.

A leg is one server lifetime of one ref. Legs run ABBA (A B B A A B ...);
pair k is legs 2k and 2k+1, whatever their order. Gains are control/candidate,
so a faster candidate has gain > 1.
"""

import math
from typing import Dict, List, Sequence

from gate import config


def leg_sums(leg: Dict) -> Dict[str, float]:
    """Reduces one leg's raw stream records to the summed quantities."""
    out: Dict[str, float] = {}
    w8 = leg["workloads"]["W8"]
    out["w8_prefill_s"] = sum(s["ttft_s"] for rep in w8 for s in rep["streams"])
    out["w8_decode_s"] = sum(s["e2e_s"] - s["ttft_s"] for rep in w8 for s in rep["streams"])
    w1 = leg["workloads"]["W1"]
    out["w1_decode_s"] = sum(s["e2e_s"] - s["ttft_s"] for rep in w1 for s in rep["streams"])
    out["w1_decode_tokens"] = sum(s["output_tokens"] - 1 for rep in w1 for s in rep["streams"])
    w32 = leg["workloads"]["W32"]
    out["w32_wall_s"] = sum(rep["wall_s"] for rep in w32)
    out["w32_output_tokens"] = sum(s["output_tokens"] for rep in w32 for s in rep["streams"])
    return out


def composite(prefill_gain: float, decode_gain: float) -> float:
    return prefill_gain**config.PREFILL_EXPONENT * decode_gain**config.DECODE_EXPONENT


def _gains(control: Sequence[Dict[str, float]], candidate: Sequence[Dict[str, float]]) -> Dict[str, float]:
    def total(legs, key):
        return sum(leg[key] for leg in legs)

    prefill = total(control, "w8_prefill_s") / total(candidate, "w8_prefill_s")
    decode = total(control, "w8_decode_s") / total(candidate, "w8_decode_s")
    c_tpot = total(control, "w1_decode_s") / total(control, "w1_decode_tokens")
    k_tpot = total(candidate, "w1_decode_s") / total(candidate, "w1_decode_tokens")
    c_tput = total(control, "w32_output_tokens") / total(control, "w32_wall_s")
    k_tput = total(candidate, "w32_output_tokens") / total(candidate, "w32_wall_s")
    return {
        "w8_prefill_gain": prefill,
        "w8_decode_gain": decode,
        "w8_composite": composite(prefill, decode),
        "w1_tpot_gain": c_tpot / k_tpot,
        "w1_tpot_control_ms": 1e3 * c_tpot,
        "w1_tpot_candidate_ms": 1e3 * k_tpot,
        "w32_tput_gain": k_tput / c_tput,
        "w32_tput_control_tok_s": c_tput,
        "w32_tput_candidate_tok_s": k_tput,
    }


GATED_METRICS = ("w8_composite", "w8_prefill_gain", "w8_decode_gain", "w1_tpot_gain", "w32_tput_gain")


def summarize_pairs(control: List[Dict[str, float]], candidate: List[Dict[str, float]]) -> Dict:
    """Overall ratio-of-sums gains plus each pair's own gains and skew."""
    if len(control) != len(candidate):
        raise ValueError(f"unpaired legs: {len(control)} control vs {len(candidate)} candidate")
    overall = _gains(control, candidate)
    pairs = []
    for c, k in zip(control, candidate):
        g = _gains([c], [k])
        g["skew_vs_overall"] = {m: g[m] / overall[m] - 1.0 for m in GATED_METRICS}
        pairs.append(g)
    log_sigma, ci95 = {}, {}
    for m in GATED_METRICS:
        logs = [math.log(p[m]) for p in pairs]
        log_sigma[m] = _sample_std(logs)
        ci95[m] = pair_ci95(logs)
    return {"overall": overall, "pairs": pairs, "per_pair_log_sigma": log_sigma, "ci95": ci95, "n_pairs": len(pairs)}


# Two-sided 95% Student t quantiles by degrees of freedom; above 30 the normal 1.96 is close enough.
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
    return max(config.NOISE_SIGMAS * sigma, config.MIN_BAR)


# Metrics a trial may decide on (its frozen prediction names one), with their check-name stems.
DECIDING_METRICS = {"w8_composite": "w8", "w1_tpot_gain": "w1_tpot", "w32_tput_gain": "w32_tput"}
DEFAULT_DECIDING_METRIC = "w8_composite"


def timing_verdict(summary: Dict, noise: Dict[str, float], decide_on: str = DEFAULT_DECIDING_METRIC) -> Dict:
    """Promote iff the deciding metric clears its bar and no guarded metric regresses past its own bar.

    The default decides on the W8 composite and guards W1 TPOT (the rule every
    older verdict used). Any other deciding metric guards all the other
    workload metrics in DECIDING_METRICS, W32 included.

    `noise` maps metric -> sigma measured by an A/A run (per-pair sigma of the
    log gain). Missing noise falls back to the 1% floor and is flagged.
    """
    if decide_on not in DECIDING_METRICS:
        raise ValueError(f"cannot decide on {decide_on!r}; choose one of {sorted(DECIDING_METRICS)}")
    overall = summary["overall"]
    bars = {m: promotion_bar(noise.get(m, 0.0)) for m in GATED_METRICS}
    guards = ["w1_tpot_gain"] if decide_on == DEFAULT_DECIDING_METRIC else [
        m for m in DECIDING_METRICS if m != decide_on]
    stems = DECIDING_METRICS
    # Check names as older reports spell them for the default rule.
    clears = "w8_composite" if decide_on == DEFAULT_DECIDING_METRIC else stems[decide_on]
    checks = {f"{clears}_clears_bar": overall[decide_on] - 1.0 > bars[decide_on]}
    for m in guards:
        checks[f"{stems[m]}_no_regression"] = overall[m] - 1.0 > -bars[m]
    checks[f"{stems[decide_on]}_no_regression"] = overall[decide_on] - 1.0 > -bars[decide_on]
    checks["enough_pairs"] = summary["n_pairs"] >= config.MIN_PAIRS
    ci95 = summary.get("ci95", {})
    return {
        "decided_on": decide_on,
        "bars": bars,
        # Reported, never a check: older verdicts re-evaluate unchanged.
        "ci95_narrower_than_bar": {m: ci95[m]["half_width"] < bars[m] for m in ci95},
        "noise_calibrated": all(m in noise for m in GATED_METRICS),
        "checks": checks,
        "promote": all(checks.values()),
        "no_regression": all(v for k, v in checks.items() if k.endswith("_no_regression")),
    }
