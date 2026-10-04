"""g4poc workload tools.

  workload fetch --shards 0-13 --out-dir DIR          # WildChat-1M parquet shards at the pinned revision
  workload build-pool --parquet DIR/*.parquet --out pool.jsonl [--cap-per-lang N]
  workload generate --pool pool.jsonl --tokenizer DIR --out sessions.jsonl [--n 2000] [--seed S]
  workload stats --sessions sessions.jsonl --tokenizer DIR [--sample N]
"""

import argparse
import json
import os
import subprocess
import sys

import msgspec

from workload import generate, schema, sources


def _shards(spec: str):
    lo, _, hi = spec.partition("-")
    return range(int(lo), int(hi or lo) + 1)


def _tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="workload")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--shards", default="0-13")
    f.add_argument("--out-dir", required=True)
    b = sub.add_parser("build-pool")
    b.add_argument("--parquet", nargs="+", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--cap-per-lang", type=int, default=50000)
    g = sub.add_parser("generate")
    g.add_argument("--pool", required=True)
    g.add_argument("--tokenizer", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--n", type=int, default=2000)
    g.add_argument("--seed", default="g4poc-v1")
    g.add_argument("--config-json", default="", help="GenConfig fields as JSON, overriding the defaults")
    s = sub.add_parser("stats")
    s.add_argument("--sessions", required=True)
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--sample", type=int, default=0)
    args = ap.parse_args(argv)

    if args.cmd == "fetch":
        os.makedirs(args.out_dir, exist_ok=True)
        for i in _shards(args.shards):
            out = os.path.join(args.out_dir, f"wildchat-{i:02d}.parquet")
            if not os.path.exists(out):
                subprocess.run(["curl", "-sfL", "-o", out + ".part", sources.shard_url(i)], check=True)
                os.rename(out + ".part", out)
            print(out)
    elif args.cmd == "build-pool":
        print(json.dumps(sources.build_pool(args.parquet, args.out, args.cap_per_lang), indent=1))
    elif args.cmd == "generate":
        meta_path = args.pool + ".meta.json"
        src = ""
        if os.path.exists(meta_path):
            with open(meta_path) as fh:
                m = json.load(fh)
            src = f"{m['source']}@{m['revision'][:7]}"
        fields = {"n_sessions": args.n, "seed": args.seed, "source": src, **json.loads(args.config_json or "{}")}
        cfg = generate.GenConfig(**fields)
        sessions = generate.generate(cfg, sources.load_pool(args.pool), _tokenizer(args.tokenizer))
        n = schema.write(args.out, sessions)
        with open(args.out + ".config.json", "w") as fh:
            fh.write(json.dumps({"config": json.loads(msgspec.json.encode(cfg)), "pool": args.pool}, indent=1))
        print(f"wrote {n} sessions to {args.out}", file=sys.stderr)
    elif args.cmd == "stats":
        sessions = schema.read(args.sessions)
        print(json.dumps(generate.stats(sessions, _tokenizer(args.tokenizer), args.sample or None), indent=1))


if __name__ == "__main__":
    main()
