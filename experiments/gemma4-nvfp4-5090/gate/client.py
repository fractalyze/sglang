"""HTTP client for the native /generate endpoint (token ids in, token ids out)."""

import asyncio
import json
import time
from typing import Dict, List, Optional

import aiohttp

_TIMEOUT = aiohttp.ClientTimeout(total=1800)


async def _stream_one(session: aiohttp.ClientSession, url: str, input_ids: List[int], max_new: int, t0: float) -> Dict:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new, "ignore_eos": True},
        "stream": True,
    }
    ttft = None
    last: Optional[Dict] = None
    async with session.post(f"{url}/generate", json=payload) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if ttft is None:
                ttft = time.perf_counter() - t0
            last = chunk
    e2e = time.perf_counter() - t0
    meta = last["meta_info"]
    return {
        "ttft_s": ttft,
        "e2e_s": e2e,
        "output_tokens": meta["completion_tokens"],
        "prompt_tokens": meta["prompt_tokens"],
        "cached_tokens": meta.get("cached_tokens", 0),
        "output_ids": last.get("output_ids"),
    }


async def run_batch(url: str, prompts: List[List[int]], max_new: int) -> Dict:
    """Sends all prompts at once; per-stream times are measured from the common start."""
    async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
        t0 = time.perf_counter()
        streams = await asyncio.gather(*(_stream_one(session, url, p, max_new, t0) for p in prompts))
        wall = time.perf_counter() - t0
    return {"streams": list(streams), "wall_s": wall}


async def _greedy_with_logprobs(session, url: str, input_ids: List[int], max_new: int, top_n: int) -> Dict:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new},
        "return_logprob": True,
        "top_logprobs_num": top_n,
        "logprob_start_len": len(input_ids),
    }
    async with session.post(f"{url}/generate", json=payload) as resp:
        resp.raise_for_status()
        body = await resp.json()
    meta = body["meta_info"]
    # output_top_logprobs: per position a list of (logprob, token_id, text|None)
    top = [[(lp, tid) for lp, tid, *_ in pos] for pos in meta["output_top_logprobs"]]
    return {"output_ids": body["output_ids"], "top_logprobs": top, "finish_reason": meta.get("finish_reason")}


async def greedy_batch(url: str, prompts: List[List[int]], max_new: int, top_n: int, concurrency: int) -> List[Dict]:
    """Greedy decode with top-n logprobs; `concurrency` sets the batch composition."""
    sem = asyncio.Semaphore(concurrency)

    async def one(session, p):
        async with sem:
            return await _greedy_with_logprobs(session, url, p, max_new, top_n)

    async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
        return list(await asyncio.gather(*(one(session, p) for p in prompts)))


async def generate_text(url: str, prompts: List[List[int]], max_new: int, concurrency: int) -> List[Dict]:
    sem = asyncio.Semaphore(concurrency)

    async def one(session, p):
        payload = {"input_ids": p, "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new}}
        async with sem:
            async with session.post(f"{url}/generate", json=payload) as resp:
                resp.raise_for_status()
                body = await resp.json()
        return {"text": body["text"], "output_ids": body.get("output_ids")}

    async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
        return list(await asyncio.gather(*(one(session, p) for p in prompts)))
