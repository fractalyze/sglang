"""A session turn as chat messages and as the token ids the server will see."""

from typing import Dict, List, Sequence

from workload.schema import Session

# Where the per-session nonce goes. "start": before the persona card, so no two
# sessions share a cached prefix (the conservative default). "after_system": after
# the card, so sessions of one persona share its cached prefix as a real fleet would.
NONCE_POSITIONS = ("start", "after_system")


def system_text(session: Session, nonce: str, nonce_at: str = "start") -> str:
    if not nonce:
        return session.system
    if nonce_at == "start":
        return f"[session {nonce}]\n{session.system}"
    if nonce_at == "after_system":
        return f"{session.system}\n[session {nonce}]"
    raise ValueError(f"nonce_at {nonce_at!r} not in {NONCE_POSITIONS}")


def messages_for_turn(session: Session, k: int, replies: Sequence[str], nonce: str = "",
                      nonce_at: str = "start") -> List[Dict[str, str]]:
    """System + history + turns 0..k-1 with `replies[j]` as the assistant's answer + user turn k."""
    if len(replies) < k:
        raise ValueError(f"turn {k} needs {k} replies, got {len(replies)}")
    msgs = [{"role": "system", "content": system_text(session, nonce, nonce_at)}]
    msgs += [{"role": m.role, "content": m.content} for m in session.history]
    for j in range(k):
        msgs.append({"role": "user", "content": session.turns[j].user})
        msgs.append({"role": "assistant", "content": replies[j]})
    msgs.append({"role": "user", "content": session.turns[k].user})
    return msgs


def prompt_ids(tokenizer, messages: List[Dict[str, str]]) -> List[int]:
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=False)
    # transformers 5 returns a BatchEncoding-like mapping for some tokenizers even with return_dict=False.
    return list(ids["input_ids"]) if isinstance(ids, dict) else list(ids)


def reply_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))
