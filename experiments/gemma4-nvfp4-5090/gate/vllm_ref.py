"""`gate vllm-ref`: an ungated vLLM reference number on the gate's workloads and prompts.

Launches the model card's `vllm serve` command (pinned vLLM in its own venv)
under the host-safety protocol and times W8/W1/W32 with the same prompt
generator and per-stream definitions as the gate. It is never a gate leg: it
only shows whether the SGLang baseline is a weak one.
"""

import asyncio
import json
import os
import signal
import subprocess
import time
from typing import Dict, List

import aiohttp
import requests

from gate import config, hostwatch, prompts

VLLM_VERSION = "0.20.0"
VLLM_PYTHON = os.path.join(config.ROOT, "vllm-venv", "bin", "python")
PORT = config.PORT + 1


def command() -> List[str]:
    return [
        VLLM_PYTHON, "-m", "vllm.entrypoints.cli.main", "serve", config.MODEL_DIR,
        "--port", str(PORT),
        "--served-model-name", "gemma4nv",
        # The card's flags; the rest bounds memory to the gate's longest request.
        "--tool-call-parser", "gemma4", "--reasoning-parser", "gemma4", "--enable-auto-tool-choice",
        "--trust-remote-code",
        "--max-model-len", "4096", "--max-num-seqs", "32", "--gpu-memory-utilization", "0.85",
    ]


async def _stream_one(session, input_ids: List[int], max_new: int, t0: float) -> Dict:
    payload = {"model": "gemma4nv", "prompt": input_ids, "max_tokens": max_new, "temperature": 0.0,
               "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    ttft, usage = None, None
    async with session.post(f"http://127.0.0.1:{PORT}/v1/completions", json=payload) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("choices") and chunk["choices"][0].get("text") is not None and ttft is None:
                ttft = time.perf_counter() - t0
            if chunk.get("usage"):
                usage = chunk["usage"]
    return {"ttft_s": ttft, "e2e_s": time.perf_counter() - t0, "output_tokens": usage["completion_tokens"],
            "prompt_tokens": usage["prompt_tokens"]}


async def _batch(ps: List[List[int]], max_new: int) -> Dict:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as s:
        t0 = time.perf_counter()
        streams = await asyncio.gather(*(_stream_one(s, p, max_new, t0) for p in ps))
        return {"streams": list(streams), "wall_s": time.perf_counter() - t0}


def _wait_healthy(proc: subprocess.Popen, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm exited with {proc.returncode}")
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise TimeoutError("vllm not healthy")


def run(out_dir: str, timeout_s: int = 3600) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    tok = prompts.load_tokenizer()
    corpus = prompts.load_corpus(tok)
    log_path = os.path.join(out_dir, "vllm.log")
    res: Dict = {"vllm_version": VLLM_VERSION, "command": command(), "workloads": {}}
    with hostwatch.host_lock():
        res["preflight"] = hostwatch.wait_preflight()
        with open(log_path, "w") as log:
            proc = subprocess.Popen([*hostwatch.memory_cap_prefix(), *command()], stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True, env=dict(os.environ))
            dog = hostwatch.Watchdog(proc.pid, os.path.join(out_dir, "hostmem.csv"), log_path, phases=()).start()
            try:
                dog.set_phase("startup")
                _wait_healthy(proc, timeout_s)
                dog.set_phase("warmup")
                for wl in config.WORKLOADS:
                    ps = prompts.timing_prompts(corpus, tok.bos_token_id, f"vllm-warmup/{wl.name}", wl.concurrency,
                                                wl.prompt_tokens)
                    asyncio.run(_batch(ps, wl.decode_tokens))
                dog.set_phase("timed")
                for wl in config.WORKLOADS:
                    reps = []
                    for rep in range(wl.reps_per_leg):
                        ps = prompts.timing_prompts(corpus, tok.bos_token_id, f"vllm/{wl.name}/{rep}",
                                                    wl.concurrency, wl.prompt_tokens)
                        reps.append(asyncio.run(_batch(ps, wl.decode_tokens)))
                    res["workloads"][wl.name] = reps
            finally:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                res["host"] = dog.stop()
    res["summary"] = summarize(res["workloads"])
    with open(os.path.join(out_dir, "vllm_ref.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def summarize(workloads: Dict) -> Dict:
    w8 = [s for r in workloads["W8"] for s in r["streams"]]
    w1 = [s for r in workloads["W1"] for s in r["streams"]]
    w32 = workloads["W32"]
    return {
        "w8_prefill_s_per_rep": sum(s["ttft_s"] for s in w8) / len(workloads["W8"]),
        "w8_decode_s_per_rep": sum(s["e2e_s"] - s["ttft_s"] for s in w8) / len(workloads["W8"]),
        "w1_tpot_ms": 1e3 * sum(s["e2e_s"] - s["ttft_s"] for s in w1) / sum(s["output_tokens"] - 1 for s in w1),
        "w32_tok_s": sum(s["output_tokens"] for r in w32 for s in r["streams"]) / sum(r["wall_s"] for r in w32),
    }
