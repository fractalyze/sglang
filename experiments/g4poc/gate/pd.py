"""Prefill/decode split: per-GPU prefill and decode capacity, and a fleet cost model.

Measurement (`gate pd-measure`), on one colocated server:
- prefill-only-like: uncached prompts of a fixed length with max_new_tokens=1 at a
  fixed number in flight; capacity = uncached prompt tokens / s.
- decode-only-like: a batch of sessions is prefilled first (max_new_tokens=1), then
  the same prompts decode DECODE_TOKENS each with the prefix cached; capacity =
  output tokens / s at that batch, with the per-token latency it costs.
Each sweep's best point (decode: under a per-token latency bound) feeds the model.

Model (`gate pd-model`, fleet_model()): a request has P prompt tokens (hit rate h
cached) and O output tokens. A disaggregated fleet spends P(1-h)/R_p prefill
GPU-seconds and O/R_d decode GPU-seconds per request; the P:D GPU ratio is their
quotient. The decode side receives the request's KV over the P->D link: every full
layer's KV for all P tokens and the sliding layers' last window. The link's cost is
a parameter until it is measured on the bs2<->bs3 link.
"""

import asyncio
import math
import random
import time
from typing import Dict, List, Optional, Sequence

import aiohttp
import msgspec

from gate import config, metrics
from workload import chat
from workload.schema import Session

DECODE_TOKENS = 300
# Gemma-4-26B-A4B, FP8 KV (official config): 25 sliding layers x 8 KV heads x 256 dims x (K + V) x 1 byte,
# and 5 full layers x 2 KV heads x 512 dims, K and V stored separately.
SLIDING_KV_BYTES_PER_TOKEN = 25 * 8 * 256 * 2
FULL_KV_BYTES_PER_TOKEN = 5 * 2 * 512 * 2
SLIDING_WINDOW = 1024
_TIMEOUT = aiohttp.ClientTimeout(total=3600)


def prompts_of_length(sessions: Sequence[Session], tokenizer, length: int, n: int, seed: str) -> List[List[int]]:
    """`n` distinct session prompts cut to exactly `length` tokens (distinct nonces, so nothing is cached)."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        s = rng.choice(sessions)
        k = len(s.turns) - 1
        ids = chat.prompt_ids(tokenizer, chat.messages_for_turn(s, k, [t.reply for t in s.turns],
                                                                nonce=f"{seed}-{len(out)}"))
        if len(ids) >= length:
            out.append(ids[:length])
    return out


async def _one(http: aiohttp.ClientSession, url: str, ids: List[int], max_new: int) -> Dict:
    payload = {"input_ids": ids, "stream": False,
               "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new, "ignore_eos": True}}
    t = time.perf_counter()
    async with http.post(f"{url}/generate", json=payload) as resp:
        resp.raise_for_status()
        body = await resp.json()
    m = body["meta_info"]
    return {"e2e_s": time.perf_counter() - t, "prompt_tokens": m["prompt_tokens"],
            "cached_tokens": m.get("cached_tokens", 0), "output_tokens": m["completion_tokens"]}


async def _closed_loop(url: str, prompts: List[List[int]], in_flight: int, max_new: int) -> Dict:
    """Keeps `in_flight` requests outstanding until every prompt is sent."""
    queue = list(prompts)
    rows: List[Dict] = []

    async def worker(http):
        while queue:
            rows.append(await _one(http, url, queue.pop(), max_new))

    async with aiohttp.ClientSession(timeout=_TIMEOUT, connector=aiohttp.TCPConnector(limit=0)) as http:
        t0 = time.perf_counter()
        await asyncio.gather(*(worker(http) for _ in range(in_flight)))
        wall = time.perf_counter() - t0
    return {"rows": rows, "wall_s": wall}


def prefill_point(url: str, prompts: List[List[int]], in_flight: int) -> Dict:
    r = asyncio.run(_closed_loop(url, prompts, in_flight, max_new=1))
    rows = r["rows"]
    uncached = sum(x["prompt_tokens"] - x["cached_tokens"] for x in rows)
    return {"in_flight": in_flight, "prompt_len": len(prompts[0]), "n": len(rows),
            "prefill_tok_s": uncached / r["wall_s"], "hit_rate": metrics.hit_rate(rows),
            "ttft_p50_s": metrics.percentile([x["e2e_s"] for x in rows], 50),
            "ttft_p90_s": metrics.percentile([x["e2e_s"] for x in rows], 90)}


def decode_point(url: str, prompts: List[List[int]]) -> Dict:
    """Prefills the batch, then decodes DECODE_TOKENS for all of it at once on the cached prefixes."""
    asyncio.run(_closed_loop(url, prompts, len(prompts), max_new=1))
    r = asyncio.run(_closed_loop(url, prompts, len(prompts), max_new=DECODE_TOKENS))
    rows = r["rows"]
    out = sum(x["output_tokens"] for x in rows)
    tpot = [x["e2e_s"] / max(x["output_tokens"], 1) for x in rows]
    return {"batch": len(prompts), "prompt_len": len(prompts[0]), "decode_tok_s": out / r["wall_s"],
            "hit_rate": metrics.hit_rate(rows), "tpot_p50_s": metrics.percentile(tpot, 50),
            "tpot_p90_s": metrics.percentile(tpot, 90), "wall_s": r["wall_s"]}


def best_prefill(points: Sequence[Dict]) -> Dict:
    return max(points, key=lambda p: p["prefill_tok_s"])


def best_decode(points: Sequence[Dict], max_tpot_s: float, min_hit_rate: float = 0.95) -> Optional[Dict]:
    """Highest decode throughput whose p90 per-token latency meets the bound with the prefix still cached."""
    ok = [p for p in points if p["tpot_p90_s"] <= max_tpot_s and p["hit_rate"] >= min_hit_rate]
    return max(ok, key=lambda p: p["decode_tok_s"]) if ok else None


class Workload(msgspec.Struct, frozen=True, kw_only=True):
    mean_prompt_tokens: float
    hit_rate: float
    mean_output_tokens: float


def kv_transfer_bytes(prompt_tokens: float) -> float:
    return prompt_tokens * FULL_KV_BYTES_PER_TOKEN + min(prompt_tokens, SLIDING_WINDOW) * SLIDING_KV_BYTES_PER_TOKEN


def fleet_model(wl: Workload, prefill_tok_s: float, decode_tok_s: float, colocated_out_tok_s: float,
                link_gbps: Optional[float] = None, prices: Sequence[float] = config.GPU_PRICES_USD_PER_HR) -> Dict:
    """Per-request GPU time, P:D ratio and $/1M output tokens, disaggregated vs colocated.

    The KV transfer is reported as bytes, latency and a per-link request ceiling, not charged
    as GPU time; with `link_gbps` None (not yet measured) only the bytes are reported.
    """
    prefill_gpu_s = wl.mean_prompt_tokens * (1.0 - wl.hit_rate) / prefill_tok_s
    decode_gpu_s = wl.mean_output_tokens / decode_tok_s
    disagg_out_tok_s = wl.mean_output_tokens / (prefill_gpu_s + decode_gpu_s)
    xfer_bytes = kv_transfer_bytes(wl.mean_prompt_tokens)
    xfer = {"bytes_per_request": xfer_bytes, "link_gbps": link_gbps}
    if link_gbps:
        t = xfer_bytes * 8 / (link_gbps * 1e9)
        xfer.update(latency_s=t, max_requests_per_s_per_link=1.0 / t)
    return {
        "workload": msgspec.to_builtins(wl),
        "prefill_gpu_s_per_request": prefill_gpu_s,
        "decode_gpu_s_per_request": decode_gpu_s,
        "p_to_d_gpu_ratio": prefill_gpu_s / decode_gpu_s,
        "disagg_out_tok_s_per_gpu": disagg_out_tok_s,
        "colocated_out_tok_s_per_gpu": colocated_out_tok_s,
        "disagg_vs_colocated": disagg_out_tok_s / colocated_out_tok_s,
        "kv_transfer": xfer,
        "usd_per_mtok_output": {"disagg": metrics.cost_table(disagg_out_tok_s, prices),
                                "colocated": metrics.cost_table(colocated_out_tok_s, prices)},
    }


def fleet_size(wl: Workload, sessions: int, think_s: float, e2e_s: float, prefill_tok_s: float,
               decode_tok_s: float, colocated_out_tok_s: float) -> Dict:
    """GPUs to serve `sessions` concurrent sessions, each sending a turn every think_s + e2e_s seconds."""
    req_s = sessions / (think_s + e2e_s)
    out_s = req_s * wl.mean_output_tokens
    n_p = req_s * wl.mean_prompt_tokens * (1.0 - wl.hit_rate) / prefill_tok_s
    n_d = out_s / decode_tok_s
    return {"requests_per_s": req_s, "output_tok_s": out_s,
            "colocated_gpus": math.ceil(out_s / colocated_out_tok_s),
            "disagg_prefill_gpus": math.ceil(n_p), "disagg_decode_gpus": math.ceil(n_d),
            "disagg_gpus": math.ceil(n_p) + math.ceil(n_d)}
