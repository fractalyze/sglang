"""`gate sol`: record router choices on the baseline, then build the SOL byte and FLOP tables.

The recording server runs with SGLang's expert distribution recorder in
per_pass mode and without CUDA graphs; it is never a timed leg.
"""

import asyncio
import glob
import json
import os
from typing import Dict, List

import requests

from gate import client, config, fidelity, prompts, server, sol

_DECODE = 128


def _record(srv: server.Server, dump_dir: str, name: str, batch_prompts: List[List[int]]) -> List[Dict]:
    import torch

    before = set(glob.glob(os.path.join(dump_dir, "*.pt")))
    requests.post(f"{srv.url}/start_expert_distribution_record", timeout=60).raise_for_status()
    asyncio.run(client.run_batch(srv.url, batch_prompts, _DECODE))
    requests.post(f"{srv.url}/stop_expert_distribution_record", timeout=60).raise_for_status()
    requests.post(f"{srv.url}/dump_expert_distribution_record", timeout=600).raise_for_status()
    new = sorted(set(glob.glob(os.path.join(dump_dir, "*.pt"))) - before)
    if len(new) != 1:
        raise RuntimeError(f"{name}: expected one new dump in {dump_dir}, got {new}")
    os.rename(new[0], os.path.join(dump_dir, f"{name}.pt"))
    return torch.load(os.path.join(dump_dir, f"{name}.pt"), weights_only=False)["records"]


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
    dump_dir = os.path.join(out_dir, "expert_records")
    ref = dict(ref, env={**ref["env"], "SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR": dump_dir})
    tok = prompts.load_tokenizer()
    corpus = prompts.load_corpus(tok)
    extra = ["--expert-distribution-recorder-mode", "per_pass", "--disable-cuda-graph"]
    distinct: Dict[str, Dict[int, float]] = {}
    with server.Server(ref, os.path.join(out_dir, "sol_server.log"), extra_args=extra) as srv:
        for name, groups in _groups(tok, corpus).items():
            per_group = []
            for i, g in enumerate(groups):
                recs = _record(srv, dump_dir, f"{name}_{i}", g)
                per_group.append(sol.distinct_experts_from_records(recs, batch=len(g)))
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
        out["workloads"][wl] = {"batch": b, "prompt": p, "decode": d, "router_source": router_key,
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
