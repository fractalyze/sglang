"""Public conversation text for the history and user turns: WildChat-1M.

WildChat-1M (allenai/WildChat-1M, ODC-BY) holds real users' multi-turn chats with
a language label per conversation; it covers every study language, Korean and
Japanese included, and contains role-play. The pool keeps (user, assistant)
pairs of non-toxic conversations, chat-shaped only (no code blocks, no pasted
documents), and marks pairs from role-play conversations so the generator can
prefer them.

  python -m workload build-pool --parquet s00.parquet ... --out pool.jsonl
"""

import collections
import json
import re
from typing import Dict, Iterable, Iterator, List, Optional

import msgspec

WILDCHAT_ID = "allenai/WildChat-1M"
WILDCHAT_REVISION = "7d6490e462285cf85d91eabea0f9a954fbddcd1f"
WILDCHAT_SHARDS = 14

LANG_CODES = {"English": "en", "Korean": "ko", "Japanese": "ja", "Chinese": "zh", "Spanish": "es",
              "French": "fr", "German": "de", "Russian": "ru"}

ROLEPLAY_RE = re.compile(
    r"(?i)role.?play|act as|pretend|in character|역할|롤플|캐릭터|ロールプレイ|なりきり|キャラ|角色扮演|扮演"
    r"|juego de rol|interpreta|jeu de rôle|incarne|rollenspiel|ролев|отыгр")

# Arbitrary cut-offs, picked by reading samples.
MAX_USER_CHARS = 1200
MIN_ASSISTANT_CHARS = 20
MAX_ASSISTANT_CHARS = 4000


class Pair(msgspec.Struct, frozen=True):
    lang: str
    user: str
    assistant: str
    rp: bool
    conv: str


def shard_url(i: int) -> str:
    return (f"https://huggingface.co/datasets/{WILDCHAT_ID}/resolve/{WILDCHAT_REVISION}/data/"
            f"train-{i:05d}-of-{WILDCHAT_SHARDS:05d}.parquet")


def _chat_shaped(user: str, assistant: str) -> bool:
    if not (1 <= len(user.strip()) <= MAX_USER_CHARS):
        return False
    if not (MIN_ASSISTANT_CHARS <= len(assistant.strip()) <= MAX_ASSISTANT_CHARS):
        return False
    return "```" not in user and "```" not in assistant


def pairs_from_conversation(conv_hash: str, lang: str, conversation: List[Dict]) -> Iterator[Pair]:
    rp = bool(conversation) and bool(ROLEPLAY_RE.search(conversation[0]["content"][:2000]))
    for a, b in zip(conversation, conversation[1:]):
        if a["role"] == "user" and b["role"] == "assistant" and _chat_shaped(a["content"], b["content"]):
            yield Pair(lang, a["content"].strip(), b["content"].strip(), rp, conv_hash)


def pairs_from_rows(rows: Iterable[Dict], cap_per_lang: Optional[int] = None) -> Iterator[Pair]:
    """Rows are WildChat records (conversation_hash, language, toxic, conversation)."""
    kept = collections.Counter()
    for r in rows:
        lang = LANG_CODES.get(r["language"])
        if lang is None or r["toxic"] or (cap_per_lang is not None and kept[lang] >= cap_per_lang):
            continue
        for p in pairs_from_conversation(r["conversation_hash"], lang, r["conversation"]):
            kept[lang] += 1
            yield p


def _parquet_rows(paths: Iterable[str]) -> Iterator[Dict]:
    import pyarrow.parquet as pq

    cols = ["conversation_hash", "language", "toxic", "conversation"]
    for path in paths:
        table = pq.read_table(path, columns=cols)
        for batch in table.to_batches(max_chunksize=4096):
            d = batch.to_pydict()
            for i in range(batch.num_rows):
                conv = [{"role": t["role"], "content": t["content"]} for t in d["conversation"][i]]
                yield {"conversation_hash": d["conversation_hash"][i], "language": d["language"][i],
                       "toxic": d["toxic"][i], "conversation": conv}


def build_pool(parquet_paths: List[str], out_path: str, cap_per_lang: int = 50000) -> Dict[str, int]:
    counts = collections.Counter()
    enc = msgspec.json.Encoder()
    with open(out_path, "wb") as f:
        for p in pairs_from_rows(_parquet_rows(parquet_paths), cap_per_lang):
            f.write(enc.encode(p) + b"\n")
            counts[p.lang] += 1
            counts[f"{p.lang}:rp"] += p.rp
    meta = {"source": WILDCHAT_ID, "revision": WILDCHAT_REVISION, "shards": [str(x) for x in parquet_paths],
            "cap_per_lang": cap_per_lang, "counts": dict(counts)}
    with open(out_path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    return dict(counts)


def load_pool(path: str) -> Dict[str, List[Pair]]:
    dec = msgspec.json.Decoder(Pair)
    by_lang: Dict[str, List[Pair]] = collections.defaultdict(list)
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                p = dec.decode(line)
                by_lang[p.lang].append(p)
    return dict(by_lang)
