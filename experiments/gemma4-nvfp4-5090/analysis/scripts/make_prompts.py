"""Build diverse fixed-length prompts (token ids) for the Yukon-shaped workload.

Identical prompts make every MoE expert run length B and inflate collision
gains ~3x (Yukon cdb12c49), so each prompt comes from a different document:
Wikipedia prose (wikitext-103 test), Python source, and Markdown docs, in
round-robin order so any prefix of the list mixes domains.

Usage: python make_prompts.py --src-root <sglang checkout> --out prompts.json
"""

import argparse
import glob
import json
import random

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

MODEL = "nvidia/Gemma-4-26B-A4B-NVFP4"


def wiki_docs():
    import pyarrow.parquet as pq

    path = hf_hub_download(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    lines = pq.read_table(path).column("text").to_pylist()
    docs, cur = [], []
    for line in lines:
        if line.startswith(" = ") and not line.startswith(" = = "):
            if cur:
                docs.append("".join(cur))
            cur = [line]
        else:
            cur.append(line)
    if cur:
        docs.append("".join(cur))
    return docs


def file_docs(pattern):
    out = []
    for p in sorted(glob.glob(pattern, recursive=True)):
        try:
            out.append(open(p, encoding="utf-8").read())
        except (UnicodeDecodeError, OSError):
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    rng = random.Random(args.seed)
    pools = {
        "wiki": wiki_docs(),
        "code": file_docs(f"{args.src_root}/python/sglang/srt/**/*.py"),
        "docs": file_docs(f"{args.src_root}/python/**/*.md"),
    }
    for name, docs in pools.items():
        long_docs = [d for d in docs if len(d) > args.len * 6]
        rng.shuffle(long_docs)
        pools[name] = long_docs
        print(name, len(long_docs), "docs long enough")

    prompts = []
    while len(prompts) < args.n and any(pools.values()):
        for name in [n for n, docs in pools.items() if docs]:
            ids = tok(pools[name].pop(), add_special_tokens=False)["input_ids"]
            if len(ids) >= args.len - 1 and len(prompts) < args.n:
                ids = [tok.bos_token_id] + ids[: args.len - 1]
                prompts.append({"domain": name, "input_ids": ids})
    json.dump(prompts, open(args.out, "w"))
    print("wrote", len(prompts), "prompts of", args.len, "tokens")


if __name__ == "__main__":
    main()
