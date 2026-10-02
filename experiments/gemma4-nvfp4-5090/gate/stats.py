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
    log_sigma = {}
    for m in GATED_METRICS:
        logs = [math.log(p[m]) for p in pairs]
        log_sigma[m] = _sample_std(logs)
    return {"overall": overall, "pairs": pairs, "per_pair_log_sigma": log_sigma, "n_pairs": len(pairs)}


def _sample_std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mean = sum(xs) / len(xs)
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))


def promotion_bar(sigma: float) -> float:
    """Relative gain a candidate must exceed: max(3 sigma_noise, 1%)."""
    return max(config.NOISE_SIGMAS * sigma, config.MIN_BAR)


def timing_verdict(summary: Dict, noise: Dict[str, float]) -> Dict:
    """Promote iff the W8 composite clears its bar and W1 TPOT does not regress past its own bar.

    `noise` maps metric -> sigma measured by an A/A run (per-pair sigma of the
    log gain). Missing noise falls back to the 1% floor and is flagged.
    """
    overall = summary["overall"]
    bars = {m: promotion_bar(noise.get(m, 0.0)) for m in GATED_METRICS}
    composite_gain = overall["w8_composite"] - 1.0
    w1_change = overall["w1_tpot_gain"] - 1.0
    checks = {
        "w8_composite_clears_bar": composite_gain > bars["w8_composite"],
        "w1_tpot_no_regression": w1_change > -bars["w1_tpot_gain"],
        "w8_no_regression": composite_gain > -bars["w8_composite"],
        "enough_pairs": summary["n_pairs"] >= config.MIN_PAIRS,
    }
    return {
        "bars": bars,
        "noise_calibrated": all(m in noise for m in GATED_METRICS),
        "checks": checks,
        "promote": all(checks.values()),
        "no_regression": checks["w8_no_regression"] and checks["w1_tpot_no_regression"],
    }
