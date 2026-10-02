"""W32 retraction diagnosis: unpaired W32-only probes of the base ref with extra server flags.

Never a gate number. Each config gets its own server lifetime under the host
lock, the same capped scope and watchdog as a gate leg, one warm-up W32 batch,
then REPS timed W32 batches with the gate's prompt seeds. Retractions are
counted from the server log of the timed window only.

  bin/python trials/diag_w32.py <out_dir> <name>=<json list of extra flags> ...
"""

import asyncio
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gate import client, config, hostwatch, prompts, server  # noqa: E402

REPS = 3
_RETRACT = re.compile(r"Retract requests\. #retracted_reqs: (\d+)")
_POOL = re.compile(r"full_layer_tokens=(\d+), swa_layer_tokens=(\d+)")
_MAX_RUNNING = re.compile(r"max_total_num_tokens=(\d+).*max_running_requests=(\d+).*available_gpu_mem=([\d.]+) GB")


def _summary(rep):
    streams = rep["streams"]
    ttft = sorted(s["ttft_s"] for s in streams)
    tokens = sum(s["output_tokens"] for s in streams)
    return {"wall_s": rep["wall_s"], "tok_s": tokens / rep["wall_s"], "ttft_max_s": ttft[-1],
            "ttft_median_s": ttft[len(ttft) // 2], "e2e_min_s": min(s["e2e_s"] for s in streams)}


def probe(ref, name, extra, out_dir, corpus, bos):
    leg_dir = os.path.join(out_dir, name)
    os.makedirs(leg_dir, exist_ok=True)
    log = os.path.join(leg_dir, "server.log")
    wl = config.W32
    with server.Server(ref, log, extra_args=extra) as srv:
        warm = prompts.timing_prompts(corpus, bos, f"warmup/diag/{wl.name}", wl.concurrency, wl.prompt_tokens)
        asyncio.run(client.run_batch(srv.url, warm, wl.decode_tokens))
        offset = srv.log_offset()
        reps = []
        for rep in range(REPS):
            srv.flush_cache()
            ps = prompts.timing_prompts(corpus, bos, f"diag/{wl.name}/{rep}", wl.concurrency, wl.prompt_tokens)
            reps.append(asyncio.run(client.run_batch(srv.url, ps, wl.decode_tokens)))
    with open(log) as f:
        text = f.read()
    window = text[offset:]
    pool = _POOL.search(text)
    sizing = _MAX_RUNNING.search(text)
    out = {
        "name": name, "extra_args": extra, "commit": srv.commit,
        "full_layer_tokens": int(pool.group(1)), "swa_layer_tokens": int(pool.group(2)),
        "max_running_requests": int(sizing.group(2)), "available_gpu_mem_gb": float(sizing.group(3)),
        "retract_events": len(_RETRACT.findall(window)),
        "retracted_reqs": sum(int(n) for n in _RETRACT.findall(window)),
        "reps": [_summary(r) for r in reps],
        "tok_s_total": sum(sum(s["output_tokens"] for s in r["streams"]) for r in reps) / sum(r["wall_s"] for r in reps),
        "host": srv.host_summary,
    }
    with open(os.path.join(leg_dir, "probe.json"), "w") as f:
        json.dump({**out, "raw": reps}, f, indent=1)
    return out


def main():
    out_dir = sys.argv[1]
    configs = [arg.split("=", 1) for arg in sys.argv[2:]]
    ref = server.load_ref("base")
    tok = prompts.load_tokenizer()
    corpus = prompts.load_corpus(tok)
    results = []
    with hostwatch.host_lock():
        for name, extra in configs:
            res = probe(ref, name, json.loads(extra), out_dir, corpus, tok.bos_token_id)
            print(json.dumps({k: v for k, v in res.items() if k != "host"}), flush=True)
            results.append(res)
    with open(os.path.join(out_dir, "diag.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
