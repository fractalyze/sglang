"""Multilingual role-play quality guard (`gate rp-quality`, `gate rp-judge`).

Items are first turns of workload sessions (persona card + ~5K tokens of history +
user turn), ITEMS_PER_LANGUAGE per language, so the guard sees the long contexts a
KV-cache change touches. The baseline run fixes the items and its greedy replies.

Two checks, both against the baseline:
- reference consistency (rp-quality): the candidate's teacher-forced NLL per token
  on the baseline's replies may rise at most RP_MAX_NLL_RISE over the baseline's
  own; the candidate's free replies must stay in the persona's language as often.
- pairwise judge (rp-judge): an LLM judge compares baseline and candidate replies
  in both orders; an item scores +1/-1 only when both orders agree. The candidate
  fails when the CI lower bound of its net loss rate exceeds RP_MAX_NET_LOSS_PT.
"""

import asyncio
import math
import random
import re
from typing import Dict, List, Optional, Sequence

import aiohttp

from gate import config
from workload import chat
from workload.schema import Session

ITEMS_PER_LANGUAGE = 10
_TIMEOUT = aiohttp.ClientTimeout(total=1800)


def build_items(sessions: Sequence[Session], tokenizer, seed: str = "rp-v1") -> List[Dict]:
    by_lang: Dict[str, List[Session]] = {}
    for s in sessions:
        by_lang.setdefault(s.language, []).append(s)
    rng = random.Random(seed)
    items = []
    for lang in sorted(by_lang):
        for s in rng.sample(by_lang[lang], min(ITEMS_PER_LANGUAGE, len(by_lang[lang]))):
            msgs = chat.messages_for_turn(s, 0, [t.reply for t in s.turns])
            items.append({"id": f"{s.session_id}/0", "language": lang, "persona_id": s.persona_id,
                          "system": s.system, "user": s.turns[0].user,
                          "input_ids": chat.prompt_ids(tokenizer, msgs)})
    return items


# ---------------------------------------------------------------------------
# Language adherence by script and stopwords.
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "en": {"the", "and", "you", "is", "to", "of", "it", "that", "my", "your", "with", "for"},
    "es": {"el", "la", "que", "de", "y", "es", "en", "los", "por", "con", "una", "mi"},
    "fr": {"le", "la", "les", "et", "est", "de", "que", "un", "une", "je", "tu", "pas"},
    "de": {"der", "die", "das", "und", "ist", "ich", "du", "nicht", "ein", "eine", "mit", "zu"},
}


def _frac(text: str, pattern: str) -> float:
    letters = re.findall(r"\w", text)
    return len(re.findall(pattern, text)) / len(letters) if letters else 0.0


def detect_language(text: str) -> Optional[str]:
    """Best guess among the study languages; None for text with no letters."""
    if not re.search(r"\w", text):
        return None
    hangul = _frac(text, r"[가-힣]")
    kana = _frac(text, r"[぀-ヿ]")
    han = _frac(text, r"[一-鿿]")
    cyr = _frac(text, r"[Ѐ-ӿ]")
    if hangul >= 0.3:
        return "ko"
    if kana >= 0.05:
        return "ja"
    if han >= 0.3:
        return "zh"
    if cyr >= 0.3:
        return "ru"
    words = re.findall(r"[a-zA-ZÀ-ÿ]+", text.lower())
    scores = {lang: sum(1 for w in words if w in sw) for lang, sw in _STOPWORDS.items()}
    return max(scores, key=lambda k: scores[k])


# ---------------------------------------------------------------------------
# Server passes.
# ---------------------------------------------------------------------------


async def _generate(session, url: str, ids: List[int]) -> Dict:
    payload = {"input_ids": ids, "sampling_params": {"temperature": 0.0, "max_new_tokens": config.RP_MAX_NEW_TOKENS}}
    async with session.post(f"{url}/generate", json=payload) as resp:
        resp.raise_for_status()
        body = await resp.json()
    return {"text": body["text"], "output_ids": body.get("output_ids") or []}


async def _forced_nll(session, url: str, prompt: List[int], cont: List[int]) -> float:
    """Mean negative log-likelihood per token of `cont` after `prompt` (teacher forced)."""
    if not cont:
        return float("nan")
    payload = {"input_ids": prompt + cont, "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
               "return_logprob": True, "logprob_start_len": len(prompt) - 1}
    async with session.post(f"{url}/generate", json=payload) as resp:
        resp.raise_for_status()
        body = await resp.json()
    # Row 0 is the last prompt token, which SGLang leaves without a logprob.
    rows = body["meta_info"]["input_token_logprobs"][1:]
    if len(rows) != len(cont) or any(r[0] is None for r in rows):
        raise RuntimeError(f"forced logprobs: {len(rows)} rows for {len(cont)} tokens")
    return -sum(r[0] for r in rows) / len(rows)


async def _gather(fn, args_list, concurrency: int):
    sem = asyncio.Semaphore(concurrency)

    async def one(session, args):
        async with sem:
            return await fn(session, *args)

    async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
        return list(await asyncio.gather(*(one(session, a) for a in args_list)))


def run(url: str, items: List[Dict], reference: Optional[List[Dict]] = None, concurrency: int = 16) -> List[Dict]:
    """Free greedy replies, plus the forced NLL of the reference replies (or of its own at baseline)."""
    outs = asyncio.run(_gather(_generate, [(url, it["input_ids"]) for it in items], concurrency))
    refs = reference or outs
    nll = asyncio.run(_gather(_forced_nll, [(url, it["input_ids"], r["output_ids"]) for it, r in zip(items, refs)],
                              concurrency))
    return [{"id": it["id"], "language": it["language"], "text": o["text"], "output_ids": o["output_ids"],
             "reply_language": detect_language(o["text"]), "ref_nll": n} for it, o, n in zip(items, outs, nll)]


def _mean(xs: Sequence[float]) -> float:
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def consistency_verdict(baseline: List[Dict], candidate: List[Dict]) -> Dict:
    nll_b = _mean([r["ref_nll"] for r in baseline])
    nll_k = _mean([r["ref_nll"] for r in candidate])
    lang_b = sum(r["reply_language"] == r["language"] for r in baseline)
    lang_k = sum(r["reply_language"] == r["language"] for r in candidate)
    checks = {"nll_rise_within_budget": nll_k - nll_b <= config.RP_MAX_NLL_RISE,
              "language_adherence_not_lower": lang_k >= lang_b}
    return {"baseline_nll": nll_b, "candidate_nll": nll_k, "nll_rise": nll_k - nll_b,
            "budget": config.RP_MAX_NLL_RISE, "language_ok": {"baseline": lang_b, "candidate": lang_k,
                                                               "n": len(candidate)},
            "checks": checks, "pass": all(checks.values())}


# ---------------------------------------------------------------------------
# Pairwise LLM judge.
# ---------------------------------------------------------------------------

JUDGE_TEMPLATE = """You are judging two replies from a role-play chat character.

Character card:
{system}

The user's last message:
{user}

Reply A:
{a}

Reply B:
{b}

Which reply is better as the character's next message? Judge staying in character, coherence with the user's message, \
writing in the card's language, and fluency. Length alone is not a reason to prefer a reply. Answer with exactly one \
token: A, B, or TIE."""


def judge_prompt(item: Dict, a: str, b: str) -> str:
    return JUDGE_TEMPLATE.format(system=item["system"], user=item["user"], a=a, b=b)


def parse_judgement(text: str) -> Optional[str]:
    m = re.search(r"\b(TIE|A|B)\b", text.strip().upper())
    return m.group(1) if m else None


def item_score(ab: Optional[str], ba: Optional[str]) -> int:
    """+1 candidate better in both orders, -1 worse in both, else 0. A is baseline in `ab`, candidate in `ba`."""
    if ab == "B" and ba == "A":
        return 1
    if ab == "A" and ba == "B":
        return -1
    return 0


def judge_verdict(scores: Sequence[int], z: float = 1.959964) -> Dict:
    n = len(scores)
    mean = sum(scores) / n
    sd = math.sqrt(sum((s - mean) ** 2 for s in scores) / (n - 1)) if n > 1 else 0.0
    half = z * sd / math.sqrt(n) if n else float("nan")
    net_loss_pt = -100.0 * mean
    ci_low = net_loss_pt - 100.0 * half
    return {"n": n, "wins": sum(1 for s in scores if s > 0), "losses": sum(1 for s in scores if s < 0),
            "net_loss_pt": net_loss_pt, "net_loss_ci95_pt": [ci_low, net_loss_pt + 100.0 * half],
            "budget_pt": config.RP_MAX_NET_LOSS_PT, "pass": ci_low <= config.RP_MAX_NET_LOSS_PT}


async def _judge_one(session, url: str, model: str, prompt: str) -> str:
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0,
               "max_tokens": 8}
    async with session.post(f"{url}/v1/chat/completions", json=payload) as resp:
        resp.raise_for_status()
        body = await resp.json()
    return body["choices"][0]["message"]["content"] or ""


def judge(url: str, model: str, items: List[Dict], baseline: List[Dict], candidate: List[Dict],
          concurrency: int = 16) -> Dict:
    """Both orders per item against an OpenAI-compatible endpoint."""
    by_b = {r["id"]: r for r in baseline}
    by_k = {r["id"]: r for r in candidate}
    prompts = []
    for it in items:
        b, k = by_b[it["id"]]["text"], by_k[it["id"]]["text"]
        prompts += [judge_prompt(it, b, k), judge_prompt(it, k, b)]
    texts = asyncio.run(_gather(_judge_one, [(url, model, p) for p in prompts], concurrency))
    rows = []
    for i, it in enumerate(items):
        ab, ba = parse_judgement(texts[2 * i]), parse_judgement(texts[2 * i + 1])
        rows.append({"id": it["id"], "language": it["language"], "ab": ab, "ba": ba, "score": item_score(ab, ba)})
    return {"items": rows, "verdict": judge_verdict([r["score"] for r in rows]), "judge": {"url": url, "model": model}}
