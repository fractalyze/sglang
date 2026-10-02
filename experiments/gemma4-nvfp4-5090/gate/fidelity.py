"""Fidelity gate against the pinned baseline: teacher-forced top-1 agreement and logprob KL.

The prompts live only on the host (config.HIDDEN_DIR). The reference is the
pinned baseline's greedy decode with every prompt in one batch. The baseline is
not run-to-run deterministic: a repeat at the same batch composition already
flips greedy tokens on near-ties (4 of 22 prompts, 2026-10-02 calibration), so
free-running token match cannot carry a 10% budget. The gated token check is
therefore teacher-forced: the reference tokens are fed back as input and each
position's top-1 must agree with the reference token. KL is gated both
teacher-forced (prefill path, all positions) and free-running (decode path, up
to the first divergence). Thresholds come from `calibrate`: the baseline again
at a different batch composition (one prompt at a time).
"""

import asyncio
import hashlib
import json
import math
import os
from typing import Dict, List, Sequence, Tuple

from gate import client, config

PROMPTS_PATH = os.path.join(config.HIDDEN_DIR, "fidelity_prompts.jsonl")
REFERENCE_PATH = os.path.join(config.REFERENCE_DIR, "fidelity_reference.json")
THRESHOLDS_PATH = os.path.join(config.REFERENCE_DIR, "fidelity_thresholds.json")
REFERENCE_FORCED_PATH = os.path.join(config.REFERENCE_DIR, "fidelity_reference_forced.json")

REFERENCE_CONCURRENCY = 64
CALIBRATION_CONCURRENCY = 1


def load_prompts() -> List[Dict]:
    with open(PROMPTS_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def prompts_digest() -> str:
    with open(PROMPTS_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def run(url: str, concurrency: int = REFERENCE_CONCURRENCY) -> List[Dict]:
    prompts = load_prompts()
    outs = asyncio.run(
        client.greedy_batch(url, [p["input_ids"] for p in prompts], config.FIDELITY_MAX_NEW_TOKENS,
                            config.TOP_LOGPROBS, concurrency)
    )
    return [{"id": p["id"], "category": p["category"], "prompt_tokens": len(p["input_ids"]), **o}
            for p, o in zip(prompts, outs)]


def run_forced(url: str, reference: List[Dict], concurrency: int = REFERENCE_CONCURRENCY) -> List[Dict]:
    by_id = {p["id"]: p for p in load_prompts()}
    rows = asyncio.run(client.forced_batch(
        url, [by_id[r["id"]]["input_ids"] for r in reference], [r["output_ids"] for r in reference],
        config.TOP_LOGPROBS, concurrency))
    return [{"id": r["id"], "top_logprobs": row} for r, row in zip(reference, rows)]


def compare_forced(reference: List[Dict], ref_forced: List[Dict], cand_forced: List[Dict]) -> Dict:
    """Per prompt: share of reference positions where the candidate's top-1 is the reference token."""
    ref_by, cand_by = {r["id"]: r for r in ref_forced}, {c["id"]: c for c in cand_forced}
    per_prompt, all_kl = [], []
    for ref in reference:
        toks = ref["output_ids"]
        cand_rows, ref_rows = cand_by[ref["id"]]["top_logprobs"], ref_by[ref["id"]]["top_logprobs"]
        agree = sum(1 for t, row in zip(toks, cand_rows) if max(row)[1] == t) / max(len(toks), 1)
        kls = [topk_kl(a, b) for a, b in zip(ref_rows, cand_rows)]
        all_kl.extend(kls)
        per_prompt.append({"id": ref["id"], "category": ref["category"], "top1_agreement": agree,
                           "kl_mean": sum(kls) / len(kls) if kls else 0.0})
    return {
        "per_prompt": per_prompt,
        "min_top1_agreement": min(p["top1_agreement"] for p in per_prompt),
        "mean_top1_agreement": sum(p["top1_agreement"] for p in per_prompt) / len(per_prompt),
        "kl_mean": sum(all_kl) / len(all_kl) if all_kl else 0.0,
        "kl_p99": _percentile(all_kl, 0.99),
    }


def first_divergence(a: Sequence[int], b: Sequence[int]) -> int:
    """Index of the first differing token; min length if one is a prefix of the other, -1 if equal."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))


def token_match_rate(a: Sequence[int], b: Sequence[int]) -> float:
    n = max(len(a), len(b))
    if n == 0:
        return 1.0
    return sum(1 for x, y in zip(a, b) if x == y) / n


def topk_kl(p_top: Sequence[Tuple[float, int]], q_top: Sequence[Tuple[float, int]]) -> float:
    """KL(P||Q) over the union of both top-k supports.

    A token outside one side's top-k gets that side's k-th logprob (an upper
    bound on its true value); both sides are renormalized over the union. The
    estimator is biased but identical for calibration and candidates.
    """
    p = {t: lp for lp, t in p_top}
    q = {t: lp for lp, t in q_top}
    p_floor, q_floor = min(p.values()), min(q.values())
    union = set(p) | set(q)
    lp = {t: p.get(t, p_floor) for t in union}
    lq = {t: q.get(t, q_floor) for t in union}
    zp = _logsumexp(lp.values())
    zq = _logsumexp(lq.values())
    kl = 0.0
    for t in union:
        a, b = lp[t] - zp, lq[t] - zq
        kl += math.exp(a) * (a - b)
    return max(kl, 0.0)


def _logsumexp(xs) -> float:
    xs = list(xs)
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def _percentile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def compare(reference: List[Dict], candidate: List[Dict]) -> Dict:
    """Per-prompt divergence, match rate and KL; KL only where both saw the same context."""
    by_id = {c["id"]: c for c in candidate}
    per_prompt, all_kl = [], []
    for ref in reference:
        cand = by_id[ref["id"]]
        div = first_divergence(ref["output_ids"], cand["output_ids"])
        upto = len(ref["output_ids"]) if div < 0 else div + 1
        upto = min(upto, len(ref["top_logprobs"]), len(cand["top_logprobs"]))
        kls = [topk_kl(ref["top_logprobs"][i], cand["top_logprobs"][i]) for i in range(upto)]
        all_kl.extend(kls)
        per_prompt.append({
            "id": ref["id"],
            "category": ref["category"],
            "prompt_tokens": ref["prompt_tokens"],
            "first_divergence": div,
            "token_match_rate": token_match_rate(ref["output_ids"], cand["output_ids"]),
            "kl_mean": sum(kls) / len(kls) if kls else 0.0,
            "kl_max": max(kls) if kls else 0.0,
        })
    return {
        "per_prompt": per_prompt,
        "min_token_match_rate": min(p["token_match_rate"] for p in per_prompt),
        "mean_token_match_rate": sum(p["token_match_rate"] for p in per_prompt) / len(per_prompt),
        "n_diverged": sum(1 for p in per_prompt if p["first_divergence"] >= 0),
        "kl_mean": sum(all_kl) / len(all_kl) if all_kl else 0.0,
        "kl_p99": _percentile(all_kl, 0.99),
        "n_kl_positions": len(all_kl),
    }


def thresholds_from_calibration(calib: Dict, calib_forced: Dict) -> Dict:
    f = config.KL_CALIBRATION_FACTOR
    return {
        "top1_agreement_min": config.TOKEN_MATCH_MIN,
        "decode_kl_mean_max": max(f * calib["kl_mean"], config.KL_MEAN_FLOOR),
        "decode_kl_p99_max": max(f * calib["kl_p99"], config.KL_P99_FLOOR),
        "forced_kl_mean_max": max(f * calib_forced["kl_mean"], config.KL_MEAN_FLOOR),
        "forced_kl_p99_max": max(f * calib_forced["kl_p99"], config.KL_P99_FLOOR),
        "calibration": {
            "decode": {k: calib[k] for k in ("kl_mean", "kl_p99", "min_token_match_rate", "mean_token_match_rate",
                                             "n_diverged")},
            "forced": {k: calib_forced[k] for k in ("kl_mean", "kl_p99", "min_top1_agreement",
                                                    "mean_top1_agreement")},
        },
    }


def verdict(cmp: Dict, cmp_forced: Dict, thresholds: Dict) -> Dict:
    """Free-running token match is reported in `cmp` but not gated (see module docstring)."""
    failing = [p["id"] for p in cmp_forced["per_prompt"] if p["top1_agreement"] < thresholds["top1_agreement_min"]]
    checks = {
        "forced_top1_per_prompt": not failing,
        "forced_kl_mean": cmp_forced["kl_mean"] <= thresholds["forced_kl_mean_max"],
        "forced_kl_p99": cmp_forced["kl_p99"] <= thresholds["forced_kl_p99_max"],
        "decode_kl_mean": cmp["kl_mean"] <= thresholds["decode_kl_mean_max"],
        "decode_kl_p99": cmp["kl_p99"] <= thresholds["decode_kl_p99_max"],
    }
    return {"pass": all(checks.values()), "checks": checks, "prompts_below_top1_budget": failing}


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)
