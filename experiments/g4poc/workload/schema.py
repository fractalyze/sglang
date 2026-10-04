"""Session file schema: one multi-turn chat session per JSONL line.

This is the contract a customer's anonymized samples must be converted to; the
load generator reads nothing else. Text, not token ids, so the same file serves
any tokenizer and the server-side chat template sees what a real client sends.

  {"session_id": "s000123", "language": "ko", "persona_id": "ko-barista",
   "system": "<persona card + rules>",
   "history": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}, ...],
   "turns": [{"user": ..., "reply": ..., "think_s": 0.0, "max_new_tokens": 300}, ...],
   "source": "wildchat-1m@7d6490e"}

`history` is the conversation before the first timed turn (it may be empty).
`turns[k].reply` is the scripted assistant reply: in scripted mode it enters the
history after turn k and sets how many tokens turn k decodes; in closed mode the
model's own reply is used instead. `think_s` is the user's delay between the
previous reply arriving and this turn being sent (0 for the first turn).
"""

from typing import Iterable, Iterator, List

import msgspec


class Message(msgspec.Struct, frozen=True):
    role: str
    content: str


class Turn(msgspec.Struct, frozen=True, kw_only=True):
    user: str
    reply: str = ""
    think_s: float = 0.0
    max_new_tokens: int = 300


class Session(msgspec.Struct, frozen=True, kw_only=True):
    session_id: str
    language: str
    system: str
    turns: List[Turn]
    history: List[Message] = []
    persona_id: str = ""
    source: str = ""


ROLES = ("user", "assistant")


def validate(s: Session) -> None:
    """Raises ValueError on a session the load generator cannot replay."""
    if not s.turns:
        raise ValueError(f"{s.session_id}: no turns")
    for i, m in enumerate(s.history):
        if m.role != ROLES[i % 2]:
            raise ValueError(f"{s.session_id}: history[{i}] is {m.role!r}, expected alternating user/assistant")
    if len(s.history) % 2:
        raise ValueError(f"{s.session_id}: history ends on a user message")
    for k, t in enumerate(s.turns):
        if not t.user:
            raise ValueError(f"{s.session_id}: turn {k} has no user text")
        if t.max_new_tokens < 1:
            raise ValueError(f"{s.session_id}: turn {k} max_new_tokens {t.max_new_tokens}")
        if t.think_s < 0:
            raise ValueError(f"{s.session_id}: turn {k} think_s {t.think_s}")


_ENC = msgspec.json.Encoder()
_DEC = msgspec.json.Decoder(Session)


def dumps(s: Session) -> bytes:
    return _ENC.encode(s)


def write(path: str, sessions: Iterable[Session]) -> int:
    n = 0
    with open(path, "wb") as f:
        for s in sessions:
            validate(s)
            f.write(dumps(s) + b"\n")
            n += 1
    return n


def iter_read(path: str) -> Iterator[Session]:
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                s = _DEC.decode(line)
                validate(s)
                yield s


def read(path: str) -> List[Session]:
    return list(iter_read(path))
