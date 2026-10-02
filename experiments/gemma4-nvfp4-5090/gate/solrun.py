"""`gate sol`: record router choices on the baseline, then build the SOL byte and FLOP tables.

Routing comes from the server's --enable-return-routed-experts (the expert
distribution recorder needs a model hook Gemma4 lacks). The recording server
runs without CUDA graphs and is never a timed leg.
"""

import asyncio
import json
import os
from typing import Dict, List

from gate import client, config, fidelity, prompts, server, sol

_DECODE = 128


async def _routed(url: str, ps: List[List[int]], max_new: int) -> List:
    import aiohttp
    import numpy as np
    import pybase64

    async def one(session, p):
        payload = {"input_ids": p, "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new,
                                                       "ignore_eos": True},
                   "return_routed_experts": True, "routed_experts_start_len": len(p)}
        async with session.post(f"{url}/generate", json=payload) as resp:
            resp.raise_for_status()
            body = await resp.json()
        flat = np.frombuffer(pybase64.b64decode(body["meta_info"]["routed_experts"].encode()), dtype=np.int32)
        return flat.reshape(-1, _N_LAYERS, _TOP_K)

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as session:
        return list(await asyncio.gather(*(one(session, p) for p in ps)))


_N_LAYERS, _TOP_K = 30, 8


def _groups(tok, corpus) -> Dict[str, List[List[List[int]]]]:
    hidden = [p["input_ids"] for p in fidelity.load_prompts()]
    bos = tok.bos_token_id
    return {
        "W8_hidden": [hidden[0:8], hidden[8:16], hidden[len(hidden) - 8:]],
        "W8_corpus": [prompts.timing_prompts(corpus, bos, f"sol/W8/{i}", 8, 1024) for i in range(3)],
        "W1_corpus": [prompts.timing_prompts(corpus, bos, f"sol/W1/{i}", 1, 1024) for i in range(3)],
        "W32_corpus": [prompts.timing_prompts(corpus, bos, "sol/W32/0", 32, 1024)],
    }


def measure_router(ref_name: str, out_dir: str) -> Dict[str, Dict[int, float]]:
    ref = server.load_ref(ref_name)
    tok = prompts.load_tokenizer()
    corpus = prompts.load_corpus(tok)
    # The routed-experts capturer sizes its buffer from num_experts_per_tok, which Gemma4's
    # text config calls top_k_experts; the model itself reads only top_k_experts.
    extra = ["--enable-return-routed-experts", "--disable-cuda-graph",
             "--json-model-override-args", json.dumps({"text_config": {"num_experts_per_tok": _TOP_K}})]
    distinct: Dict[str, Dict[int, float]] = {}
    with server.Server(ref, os.path.join(out_dir, "sol_server.log"), extra_args=extra) as srv:
        for name, groups in _groups(tok, corpus).items():
            per_group = []
            for g in groups:
                routes = asyncio.run(_routed(srv.url, g, _DECODE))
                per_group.append(sol.distinct_experts_from_routes(routes))
            layers = per_group[0].keys()
            distinct[name] = {l: sum(d[l] for d in per_group) / len(per_group) for l in layers}
    return distinct


def tables(distinct: Dict[str, Dict[int, float]], kv_elem_bytes: float, peaks: Dict) -> Dict:
    tensors = sol.checkpoint_tensors()
    weights = sol.weight_table(tensors)
    text_weight_bytes = sum(b for c, b in weights["bytes_by_component"].items() if c != "vision_unused")
    shapes = {"W8": (8, 1024, 128, "W8_hidden"), "W1": (1, 1024, 256, "W1_corpus"), "W32": (32, 1024, 128, "W32_corpus")}
    peak_tflops = dict(sol.SPEC_PEAK_TFLOPS)
    for k, key in (("bf16", "bf16_tflops"), ("fp8", "fp8_tflops"), ("nvfp4", "nvfp4_tflops")):
        if key in peaks.get("gemm", {}):
            peak_tflops[k] = max(peak_tflops[k], peaks["gemm"][key])
    out = {"weights": weights, "kv_elem_bytes": kv_elem_bytes, "peak_tflops_used": peak_tflops,
           "dram_gb_s_datasheet": sol.DRAM_BYTES_PER_S / 1e9,
           "dram_gb_s_measured": peaks.get("dram", {}).get("read_gb_s"), "workloads": {}}
    for wl, (b, p, d, router_key) in shapes.items():
        dec = sol.decode_sol(b, p, d, distinct[router_key], kv_elem_bytes, weights)
        flops = sol.prefill_flops(b, p, tensors)
        pre = sol.prefill_sol(flops, peak_tflops, text_weight_bytes)
        out["workloads"][wl] = {"batch": b, "prompt": p, "decode_tokens": d, "router_source": router_key,
                                "distinct_experts_mean": sum(distinct[router_key].values()) / len(distinct[router_key]),
                                "decode": dec, "prefill_flops": flops, "prefill": pre}
    out["distinct_experts_by_source"] = {k: {"mean": sum(v.values()) / len(v), "by_layer": v} for k, v in distinct.items()}
    return out


def run(ref_name: str, out_dir: str, peaks_path: str) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    distinct = measure_router(ref_name, out_dir)
    peaks = json.load(open(peaks_path)) if os.path.exists(peaks_path) else {}
    # The baseline KV cache is FP8: the checkpoint's kv_cache_quant_algo, which
    # SGLang's kv_cache_dtype=auto adopts (server log: "dtype: torch.float8_e4m3fn").
    kv_elem_bytes = 1.0
    result = tables(distinct, kv_elem_bytes, peaks)
    with open(os.path.join(out_dir, "sol.json"), "w") as f:
        json.dump(result, f, indent=1)
    return result
