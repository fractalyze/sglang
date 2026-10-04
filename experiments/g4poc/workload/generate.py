"""Session generator: persona card + same-language history + timed turns.

Each session draws a language, a persona card of that language, and a target
length for its first timed prompt (lognormal around ~4.5K tokens). The history is
filled with whole WildChat conversations of that language, pair by pair in their
original order, until the first prompt reaches the target; the timed turns then
continue with further pairs until the next prompt would pass MAX_INPUT_TOKENS or
the session's turn count is reached. Scripted replies are cut to the output cap
at a sentence end. Everything is drawn from `seed`, so a (pool, config, seed)
triple names the file exactly.
"""

import math
import random
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import msgspec

from workload import chat, personas
from workload.schema import Message, Session, Turn
from workload.sources import Pair

DEFAULT_LANG_WEIGHTS = {"en": 0.25, "ko": 0.20, "ja": 0.15, "zh": 0.15, "es": 0.0625, "fr": 0.0625, "de": 0.0625,
                        "ru": 0.0625}


class GenConfig(msgspec.Struct, frozen=True, kw_only=True):
    n_sessions: int = 2000
    seed: str = "g4poc-v1"
    lang_weights: Dict[str, float] = msgspec.field(default_factory=lambda: dict(DEFAULT_LANG_WEIGHTS))
    first_prompt_median: int = 4500
    first_prompt_sigma: float = 0.35
    first_prompt_min: int = 1500
    first_prompt_max: int = 9000
    max_input_tokens: int = 10240
    turns_mean: float = 6.0
    max_turns: int = 16
    max_new_tokens: int = 300
    min_reply_tokens: int = 16
    think_median_s: float = 15.0
    think_sigma: float = 0.6
    think_min_s: float = 2.0
    think_max_s: float = 120.0
    # Probability that the next history conversation is drawn from role-play conversations.
    rp_prefer: float = 0.7
    source: str = ""


class _ConvPool:
    """One language's pairs grouped by conversation, role-play conversations listed apart."""

    def __init__(self, pairs: Sequence[Pair]):
        by_conv: Dict[str, List[Pair]] = {}
        for p in pairs:
            by_conv.setdefault(p.conv, []).append(p)
        self.rp = [v for v in by_conv.values() if v[0].rp]
        self.other = [v for v in by_conv.values() if not v[0].rp]
        if not self.rp and not self.other:
            raise ValueError("empty pool")

    def draw(self, rng: random.Random, rp_prefer: float) -> List[Pair]:
        use_rp = self.rp and (not self.other or rng.random() < rp_prefer)
        return rng.choice(self.rp if use_rp else self.other)


_SENTENCE_END = re.compile(r"[.!?。！？…\n](?=[^.!?。！？…\n]*$)")


def truncate_tokens(text: str, max_tokens: int, encode: Callable, decode: Callable) -> str:
    """`text` cut to at most `max_tokens`, at the last sentence end when one is past the halfway point."""
    ids = encode(text)
    if len(ids) <= max_tokens:
        return text
    cut = decode(ids[:max_tokens])
    m = _SENTENCE_END.search(cut)
    if m and m.end() >= len(cut) // 2:
        return cut[: m.end()].rstrip()
    return cut.rstrip()


def _lognormal(rng: random.Random, median: float, sigma: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, median * math.exp(rng.gauss(0.0, sigma))))


def _draw_lang(rng: random.Random, weights: Dict[str, float], available: Sequence[str]) -> str:
    langs = [l for l in sorted(weights) if l in available and weights[l] > 0]
    if not langs:
        raise ValueError(f"no language with weight in the pool: weights {weights}, pool {sorted(available)}")
    return rng.choices(langs, weights=[weights[l] for l in langs])[0]


class _Text(msgspec.Struct, frozen=True):
    """A pair with its reply cut to the output cap, and both sides' token counts."""

    user: str
    reply: str
    user_tokens: int
    reply_tokens: int


def _prepare(p: Pair, cfg: GenConfig, encode, decode) -> _Text:
    reply = truncate_tokens(p.assistant, cfg.max_new_tokens, encode, decode)
    return _Text(p.user, reply, len(encode(p.user)), len(encode(reply)))


def _pair_stream(pool: _ConvPool, rng: random.Random, cfg: GenConfig, encode, decode):
    while True:
        for p in pool.draw(rng, cfg.rp_prefer):
            yield _prepare(p, cfg, encode, decode)


class TemplateOverhead(msgspec.Struct, frozen=True):
    """Tokens the chat template adds: per message, and once per prompt (BOS + generation prompt)."""

    per_message: int
    per_prompt: int


def measure_template(tokenizer, encode) -> TemplateOverhead:
    def n(msgs):
        return len(chat.prompt_ids(tokenizer, msgs))

    x, y, z, w = "alpha", "beta", "gamma", "delta"
    two = [{"role": "system", "content": x}, {"role": "user", "content": y}]
    four = two + [{"role": "assistant", "content": z}, {"role": "user", "content": w}]
    per_message = (n(four) - n(two) - len(encode(z)) - len(encode(w))) // 2
    per_prompt = n(two) - len(encode(x)) - len(encode(y)) - 2 * per_message
    return TemplateOverhead(per_message, per_prompt)


def generate_session(i: int, cfg: GenConfig, pools: Dict[str, _ConvPool], encode, decode,
                     overhead: TemplateOverhead) -> Session:
    rng = random.Random(f"{cfg.seed}/{i}")
    lang = _draw_lang(rng, cfg.lang_weights, list(pools))
    persona = rng.choice(personas.PERSONAS[lang])
    target = int(_lognormal(rng, cfg.first_prompt_median, cfg.first_prompt_sigma, cfg.first_prompt_min,
                            cfg.first_prompt_max))
    n_turns = min(cfg.max_turns, 1 + int(rng.expovariate(1.0 / max(cfg.turns_mean - 1.0, 1e-9))))
    stream = _pair_stream(pools[lang], rng, cfg, encode, decode)
    per_msg = overhead.per_message
    length = len(encode(persona.card)) + per_msg + overhead.per_prompt

    history: List[Message] = []
    nxt = next(stream)
    while length + nxt.user_tokens + per_msg < target:
        history += [Message("user", nxt.user), Message("assistant", nxt.reply)]
        length += nxt.user_tokens + nxt.reply_tokens + 2 * per_msg
        nxt = next(stream)

    turns: List[Turn] = []
    while len(turns) < n_turns:
        prompt_len = length + nxt.user_tokens + per_msg
        if turns and prompt_len > cfg.max_input_tokens:
            break
        think = 0.0 if not turns else round(_lognormal(rng, cfg.think_median_s, cfg.think_sigma, cfg.think_min_s,
                                                       cfg.think_max_s), 2)
        max_new = max(cfg.min_reply_tokens, min(cfg.max_new_tokens, nxt.reply_tokens))
        turns.append(Turn(user=nxt.user, reply=nxt.reply, think_s=think, max_new_tokens=max_new))
        length = prompt_len + nxt.reply_tokens + per_msg
        nxt = next(stream)
    return Session(session_id=f"s{i:06d}", language=lang, persona_id=persona.persona_id, system=persona.card,
                   history=history, turns=turns, source=cfg.source)


def generate(cfg: GenConfig, pool: Dict[str, List[Pair]], tokenizer) -> List[Session]:
    def encode(text: str) -> List[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    def decode(ids: List[int]) -> str:
        return tokenizer.decode(ids, skip_special_tokens=True)

    pools = {lang: _ConvPool(pairs) for lang, pairs in pool.items() if pairs and lang in personas.PERSONAS}
    overhead = measure_template(tokenizer, encode)
    return [generate_session(i, cfg, pools, encode, decode, overhead) for i in range(cfg.n_sessions)]


def _quantiles(xs: List[float], qs=(0.1, 0.5, 0.9, 0.99)) -> Dict[str, float]:
    if not xs:
        return {}
    s = sorted(xs)
    out = {f"p{int(q * 100)}": s[min(len(s) - 1, int(q * len(s)))] for q in qs}
    out.update(mean=sum(s) / len(s), max=s[-1], min=s[0], n=len(s))
    return out


def stats(sessions: Sequence[Session], tokenizer, sample: Optional[int] = None) -> Dict:
    """Exact token statistics with the real chat template (scripted replies, no nonce)."""
    rows: List[Tuple[int, int, int]] = []
    langs: Dict[str, int] = {}
    turns_per, thinks = [], []
    for s in sessions[:sample] if sample else sessions:
        langs[s.language] = langs.get(s.language, 0) + 1
        turns_per.append(len(s.turns))
        replies = [t.reply for t in s.turns]
        for k, t in enumerate(s.turns):
            ids = chat.prompt_ids(tokenizer, chat.messages_for_turn(s, k, replies))
            rows.append((k, len(ids), t.max_new_tokens))
            if k:
                thinks.append(t.think_s)
    return {
        "sessions": sum(langs.values()),
        "languages": langs,
        "turns_per_session": _quantiles(turns_per),
        "prompt_tokens_all_turns": _quantiles([r[1] for r in rows]),
        "prompt_tokens_first_turn": _quantiles([r[1] for r in rows if r[0] == 0]),
        "output_tokens": _quantiles([r[2] for r in rows]),
        "think_s": _quantiles(thinks),
    }
