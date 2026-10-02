"""Token-exact timing prompts.

Prompts are random 1024-token windows of a fixed text+code corpus (the SGLang
docs and sources at the pinned baseline commit). Every (pair, workload, rep)
draws fresh windows, so no prefix or output cache can hit across timed
requests, and the 8 streams of a W8 batch route to different experts as real
traffic does (identical prompts inflated expert-sharing gains ~3x in Yukon).
Both legs of a pair see the same prompts, so their outputs can be compared.
"""

import glob
import hashlib
import os
import random
from typing import List

import numpy as np

from gate import config

CORPUS_GLOBS = ("docs/**/*.md", "docs/**/*.mdx", "python/sglang/srt/**/*.py")
CORPUS_COMMIT = "a9871012a"
_CORPUS_PATH = os.path.join(config.ROOT, "cache", f"timing_corpus_{CORPUS_COMMIT}.npy")


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(config.MODEL_DIR)


def _corpus_files(src: str) -> List[str]:
    files = []
    for pattern in CORPUS_GLOBS:
        files.extend(glob.glob(os.path.join(src, pattern), recursive=True))
    return sorted(set(files))


def build_corpus(tokenizer) -> np.ndarray:
    """Tokenizes the corpus files of the pinned commit (read with git show, not the work tree)."""
    import subprocess

    src = config.SRC_REPO
    names = subprocess.run(
        ["git", "-C", src, "ls-tree", "-r", "--name-only", CORPUS_COMMIT],
        check=True, capture_output=True, text=True,
    ).stdout.split("\n")
    wanted = set(os.path.relpath(f, src) for f in _corpus_files(src))
    ids: List[int] = []
    for name in sorted(n for n in names if n in wanted):
        text = subprocess.run(
            ["git", "-C", src, "show", f"{CORPUS_COMMIT}:{name}"], check=True, capture_output=True, text=True
        ).stdout
        ids.extend(tokenizer.encode(text, add_special_tokens=False))
    arr = np.asarray(ids, dtype=np.uint32)
    os.makedirs(os.path.dirname(_CORPUS_PATH), exist_ok=True)
    np.save(_CORPUS_PATH, arr)
    return arr


def load_corpus(tokenizer=None) -> np.ndarray:
    if os.path.exists(_CORPUS_PATH):
        return np.load(_CORPUS_PATH)
    return build_corpus(tokenizer or load_tokenizer())


def corpus_digest(corpus: np.ndarray) -> str:
    return hashlib.sha256(corpus.tobytes()).hexdigest()[:16]


def timing_prompts(corpus: np.ndarray, bos_id: int, seed: str, n: int, length: int) -> List[List[int]]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        off = rng.randrange(len(corpus) - length)
        out.append([bos_id] + corpus[off : off + length - 1].tolist())
    return out
