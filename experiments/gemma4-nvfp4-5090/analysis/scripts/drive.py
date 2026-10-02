"""Drive a Yukon-shaped workload (B streams x L-token prompt x D decode) against
a running SGLang server and report prefill / decode timing.

Prefill time for a rep is the max TTFT over its B streams (all B prompts arrive
together). Decode time per step is (t_last - t_first) / (D - 1) per stream,
averaged over streams. The radix cache is flushed before every rep so reused
prompts never hit a prefix.

These are single unpaired runs: label any number from here "screen, unpaired".
"""

import argparse
import json
import statistics
import threading
import time

import requests


def one_stream(url, ids, decode_len, out, idx):
    payload = {
        "input_ids": ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": decode_len,
            "ignore_eos": True,
        },
        "stream": True,
    }
    t0 = time.perf_counter()
    t_first = t_last = None
    n_chunks = 0
    text = None
    with requests.post(f"{url}/generate", json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data:"):
                continue
            body = line[5:].strip()
            if body == b"[DONE]":
                break
            now = time.perf_counter()
            if t_first is None:
                t_first = now
            t_last = now
            n_chunks += 1
            text = json.loads(body)
    meta = text["meta_info"] if text else {}
    out[idx] = dict(
        ttft=t_first - t0,
        decode_step=(t_last - t_first) / max(decode_len - 1, 1),
        completion_tokens=meta.get("completion_tokens"),
        output_ids_tail=(text or {}).get("output_ids", [])[-8:],
    )


def run_rep(url, prompts, decode_len):
    requests.post(f"{url}/flush_cache", timeout=60)
    time.sleep(0.5)
    out = [None] * len(prompts)
    threads = [
        threading.Thread(target=one_stream, args=(url, p, decode_len, out, i))
        for i, p in enumerate(prompts)
    ]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    return dict(
        wall=wall,
        prefill=max(o["ttft"] for o in out),
        decode_step=statistics.mean(o["decode_step"] for o in out),
        streams=out,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--decode-len", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--profile-dir", default=None)
    ap.add_argument("--profile-steps", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    prompts = [p["input_ids"] for p in json.load(open(args.prompts))]
    n = len(prompts)

    def batch_for(rep):
        return [prompts[(rep * args.batch + j) % n] for j in range(args.batch)]

    for w in range(args.warmup):
        run_rep(args.url, batch_for(1000 + w), args.decode_len)

    if args.profile_dir:
        requests.post(f"{args.url}/flush_cache", timeout=60)
        requests.post(
            f"{args.url}/start_profile",
            json=dict(
                output_dir=args.profile_dir,
                num_steps=args.profile_steps,
                activities=["CPU", "GPU"],
                profile_by_stage=True,
                record_shapes=True,
                with_stack=False,
            ),
            timeout=60,
        ).raise_for_status()

    reps = [run_rep(args.url, batch_for(r), args.decode_len) for r in range(args.reps)]
    summary = dict(
        label=args.label,
        batch=args.batch,
        prompt_len=len(prompts[0]),
        decode_len=args.decode_len,
        prefill_s=[r["prefill"] for r in reps],
        decode_step_ms=[r["decode_step"] * 1e3 for r in reps],
        wall_s=[r["wall"] for r in reps],
        reps=reps,
    )
    print(
        f"[{args.label}] B={args.batch} prefill_s med={statistics.median(summary['prefill_s']):.4f} "
        f"decode_step_ms med={statistics.median(summary['decode_step_ms']):.3f} "
        f"(all {['%.3f' % x for x in summary['decode_step_ms']]})"
    )
    if args.out:
        json.dump(summary, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
