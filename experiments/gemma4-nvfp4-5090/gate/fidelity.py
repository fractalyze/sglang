"""Fidelity gate: greedy-token agreement and top-k logprob KL against the pinned baseline.

The prompts live only on the host (config.HIDDEN_DIR). The reference outputs
are the pinned baseline's greedy decode with every prompt in one batch.
Thresholds come from `calibrate`: the baseline run again at a different batch
composition (one prompt at a time), which is the nondeterminism a correct
kernel change can legitimately add.
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


def thresholds_from_calibration(calib: Dict) -> Dict:
    return {
        "token_match_min": config.TOKEN_MATCH_MIN,
        "kl_mean_max": max(config.KL_CALIBRATION_FACTOR * calib["kl_mean"], config.KL_MEAN_FLOOR),
        "kl_p99_max": max(config.KL_CALIBRATION_FACTOR * calib["kl_p99"], config.KL_P99_FLOOR),
        "calibration": {k: calib[k] for k in ("kl_mean", "kl_p99", "min_token_match_rate", "n_diverged")},
    }


def verdict(cmp: Dict, thresholds: Dict) -> Dict:
    failing = [p["id"] for p in cmp["per_prompt"] if p["token_match_rate"] < thresholds["token_match_min"]]
    checks = {
        "token_match_per_prompt": not failing,
        "kl_mean": cmp["kl_mean"] <= thresholds["kl_mean_max"],
        "kl_p99": cmp["kl_p99"] <= thresholds["kl_p99_max"],
    }
    return {"pass": all(checks.values()), "checks": checks, "prompts_below_match_budget": failing}


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)
