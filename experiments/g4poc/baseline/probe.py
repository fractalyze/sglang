"""Concurrency probe: N simultaneous fixed-length requests, then read the scheduler log.

Each request is `input_len` random token ids (no shared prefix, so the radix
cache cannot reuse KV) and decodes exactly `output_len` tokens (ignore_eos).
The scheduler's decode log lines give the running-request count and the
retraction count; the probe reports the peak running count, retractions and
per-request latency.

usage: python probe.py --n 48 --input-len 5000 --output-len 300 --log runs/x/server.log
"""

import argparse
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor

import requests

# Hybrid-SWA format of scheduler_components/pool_stats_observer.get_decode_usage_msg_parts.
DECODE_RE = re.compile(r"Decode batch.*?#running-req: (\d+), #full token: (\d+), full token usage: ([\d.]+), "
                       r"#swa token: (\d+), swa token usage: ([\d.]+).*?#queue-req: (\d+)")
RETRACT_RE = re.compile(r"[Rr]etract")


def one(url: str, input_len: int, output_len: int, seed: int) -> dict:
    rng = random.Random(seed)
    ids = [rng.randrange(1000, 200000) for _ in range(input_len)]
    t0 = time.time()
    r = requests.post(f"{url}/generate", json={
        "input_ids": ids,
        "sampling_params": {"max_new_tokens": output_len, "ignore_eos": True, "temperature": 0},
    }, timeout=3600)
    r.raise_for_status()
    meta = r.json()["meta_info"]
    return {"latency_s": time.time() - t0, "completion_tokens": meta["completion_tokens"],
            "ttft_s": meta.get("first_token_latency") or meta.get("e2e_latency")}


def scan_log(path: str, start_offset: int) -> dict:
    with open(path, errors="replace") as f:
        f.seek(start_offset)
        text = f.read()
    rows = [m.groups() for m in DECODE_RE.finditer(text)]
    peak_running = max((int(r[0]) for r in rows), default=0)
    peak_usage = max((float(r[2]) for r in rows), default=0.0)
    peak_swa = max((float(r[4]) for r in rows), default=0.0)
    return {"decode_lines": len(rows), "peak_running": peak_running, "peak_full_usage": peak_usage,
            "peak_full_tokens": max((int(r[1]) for r in rows), default=0),
            "peak_swa_usage": peak_swa, "peak_swa_tokens": max((int(r[3]) for r in rows), default=0),
            "peak_queue": max((int(r[5]) for r in rows), default=0), "retract_lines": len(RETRACT_RE.findall(text)),
            "retract_samples": [ln for ln in text.splitlines() if RETRACT_RE.search(ln)][:3]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30100")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--input-len", type=int, default=5000)
    ap.add_argument("--output-len", type=int, default=300)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    requests.post(f"{a.url}/flush_cache", timeout=60)
    offset = len(open(a.log, "rb").read())
    t0 = time.time()
    with ThreadPoolExecutor(a.n) as ex:
        res = list(ex.map(lambda i: one(a.url, a.input_len, a.output_len, 1000 * a.n + i), range(a.n)))
    wall = time.time() - t0
    time.sleep(2)
    lat = sorted(r["latency_s"] for r in res)
    out = {"n": a.n, "input_len": a.input_len, "output_len": a.output_len, "wall_s": wall,
           "output_tok_per_s": sum(r["completion_tokens"] for r in res) / wall,
           "latency_p50_s": lat[len(lat) // 2], "latency_max_s": lat[-1], **scan_log(a.log, offset)}
    print(json.dumps(out, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
