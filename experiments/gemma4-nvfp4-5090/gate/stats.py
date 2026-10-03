"""Paired-timing statistics: ratio of sums, per-pair skew, noise and the bar.

A leg is one server lifetime of one ref. Legs run ABBA (A B B A A B ...);
pair k is legs 2k and 2k+1, whatever their order. Gains are control/candidate,
so a faster candidate has gain > 1.
"""

import math
from typing import Dict, List, Optional, Sequence

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


# ---------------------------------------------------------------------------
# Speculative-decoding rule (W14). Under MTP, a candidate that changes the
# target's numerics changes its greedy text, and the drafter's acceptance (tau)
# on the new text moves fixed-prompt timing by about +-10% on its own (W13,
# T3d: W1 -9.1% at unchanged per-round kernel time). The paired CI cannot see
# it: each arm reproduces its own text exactly. Such a candidate is decided on
# speedup = round-time ratio x tau ratio, measured separately:
#   round time: both arms replay the control's text with a fixed accept length
#               (gate spec-run), so their decode-time ratio is per-round cost;
#   tau:        free-running on the hidden set, paired per prompt, with a CI.
# ---------------------------------------------------------------------------

PAIRED_RULE = "paired"
SPEC_RULE = "spec-decomposed"


def is_speculative(ref: Dict) -> bool:
    return "--speculative-algorithm" in ref.get("server_args", [])


def timing_rule(control: Dict, candidate: Dict) -> str:
    """Which rule decides a ref pair: the paired fixed-prompt timing, or the spec decomposition.

    The paired rule stays for a candidate that declares numerics_unchanged and for pairs
    with no speculation. Any other pair with speculation in either arm decodes different
    text per arm, so its timing carries the acceptance on that text.
    """
    if candidate.get("numerics_unchanged") or not (is_speculative(control) or is_speculative(candidate)):
        return PAIRED_RULE
    return SPEC_RULE


def tau_ratio(control: Sequence[Dict], candidate: Sequence[Dict]) -> Dict:
    """Hidden-set acceptance per arm and the candidate/control ratio, paired by prompt.

    Rows are {"id", "completion_tokens", "verify_ct"}; a non-speculative arm has
    verify_ct == completion_tokens (tau 1). The point estimate is the ratio of the
    arms' tokens-per-round over all prompts; the 95% t-interval is over the per-prompt
    log ratios, so a change carried by a few prompts stays insignificant.
    """
    by_id = {r["id"]: r for r in candidate}
    if set(by_id) != {r["id"] for r in control}:
        raise ValueError("tau rows must cover the same prompts in both arms")

    def tau(rows):
        return sum(r["completion_tokens"] for r in rows) / sum(r["verify_ct"] for r in rows)

    logs = [math.log((by_id[c["id"]]["completion_tokens"] / by_id[c["id"]]["verify_ct"])
                     / (c["completion_tokens"] / c["verify_ct"])) for c in control]
    ci = pair_ci95(logs)
    tau_c, tau_k = tau(control), tau(list(by_id.values()))
    return {"control": tau_c, "candidate": tau_k, "ratio": tau_k / tau_c, "ci95": ci, "n_prompts": len(logs),
            "significant": not (ci["low"] <= 1.0 <= ci["high"])}


def counted_tau_ratio(tau: Dict, quality_pass: Optional[bool]) -> Dict:
    """The tau ratio the decision multiplies in: 1 unless the change is significant on the hidden set.

    A significant gain also needs passing quality (else it is text the target should not
    produce); a significant loss always counts.
    """
    if not tau["significant"]:
        return {"ratio": 1.0, "why": "not significant on the hidden set"}
    if tau["ratio"] > 1.0 and quality_pass is not True:
        return {"ratio": 1.0, "why": f"significant gain but quality pass is {quality_pass}"}
    return {"ratio": tau["ratio"], "why": "significant on the hidden set"}


def _with_decode(overall: Dict[str, float], decode_factor: float) -> Dict[str, float]:
    """Gains with every decode-time component multiplied by `decode_factor` (prefill unchanged)."""
    return {
        "w8_prefill_gain": overall["w8_prefill_gain"],
        "w8_decode_gain": overall["w8_decode_gain"] * decode_factor,
        "w8_composite": composite(overall["w8_prefill_gain"], overall["w8_decode_gain"] * decode_factor),
        "w1_tpot_gain": overall["w1_tpot_gain"] * decode_factor,
        "w32_tput_gain": overall["w32_tput_gain"] * decode_factor,
    }


def spec_verdict(replay: Dict, tau: Dict, quality_pass: Optional[bool], noise: Dict[str, float],
                 accept_len: Dict[str, int], decide_on: str = DEFAULT_DECIDING_METRIC) -> Dict:
    """Promotion under the spec rule.

    `replay` is summarize_pairs over replay legs: both arms decode the control's text,
    a speculative arm accepting accept_len[role] tokens per round (1 for a plain-decode
    arm). Its gains become per-token gains at the hidden-set tau by the factor
    (tau_k / A_k) / (tau_c / A_c).

    Both arms speculative (same accept length): the factor is the tau ratio, and the
    round-time component (the replay gains) must clear the bar on its own. The
    decomposition with the counted tau ratio must clear it too, so a significant tau loss
    can still block.

    One arm plain decode (a decoding-mode change): acceptance is the mechanism, so there
    is no round-time-only check. The tau ratio must be significant and quality must pass,
    and the decomposition must clear the bar.

    W32's replay sums a throughput window, which also scales with tokens per round, so the
    same factor applies to it.
    """
    if decide_on not in DECIDING_METRICS:
        raise ValueError(f"cannot decide on {decide_on!r}; choose one of {sorted(DECIDING_METRICS)}")
    mode_change = accept_len["control"] == 1 or accept_len["candidate"] == 1
    if not mode_change and accept_len["control"] != accept_len["candidate"]:
        raise ValueError(f"both speculative arms must replay one accept length, got {accept_len}")
    overall = replay["overall"]
    counted = counted_tau_ratio(tau, quality_pass)
    if mode_change:
        tau_factor = (tau["candidate"] / accept_len["candidate"]) / (tau["control"] / accept_len["control"])
        if not tau["significant"] or quality_pass is not True:
            tau_factor = None
    else:
        tau_factor = counted["ratio"]
    decomposed = _with_decode(overall, tau_factor) if tau_factor is not None else None
    bars = {m: promotion_bar(noise.get(m, 0.0)) for m in GATED_METRICS}
    stems = DECIDING_METRICS
    guards = [m for m in DECIDING_METRICS if m != decide_on]
    checks = {}
    if mode_change:
        checks["tau_significant"] = tau["significant"]
        checks["quality_pass"] = quality_pass is True
    else:
        checks[f"{stems[decide_on]}_round_time_clears_bar"] = overall[decide_on] - 1.0 > bars[decide_on]
        checks[f"{stems[decide_on]}_round_time_ci_above_1"] = replay["ci95"][decide_on]["low"] > 1.0
        for m in guards:
            checks[f"{stems[m]}_round_time_no_regression"] = overall[m] - 1.0 > -bars[m]
    checks[f"{stems[decide_on]}_decomposed_clears_bar"] = (
        decomposed is not None and decomposed[decide_on] - 1.0 > bars[decide_on])
    for m in guards:
        checks[f"{stems[m]}_decomposed_no_regression"] = decomposed is not None and decomposed[m] - 1.0 > -bars[m]
    checks["enough_pairs"] = replay["n_pairs"] >= config.MIN_PAIRS
    return {
        "rule": SPEC_RULE,
        "decided_on": decide_on,
        "mode_change": mode_change,
        "accept_len": accept_len,
        "round_time": {m: overall[m] for m in GATED_METRICS},
        "round_time_ci95": replay["ci95"],
        "tau": tau,
        "counted_tau": counted if not mode_change else {"factor": tau_factor},
        "decomposed": decomposed,
        "bars": bars,
        "noise_calibrated": all(m in noise for m in GATED_METRICS),
        "checks": checks,
        "promote": all(checks.values()),
    }
