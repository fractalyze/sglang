"""Speed-of-light model for Gemma-4-26B-A4B-NVFP4 on one RTX 5090.

Decode is bandwidth-bound: one step must read every dense weight once, the
distinct experts the router picked for the batch, the lm_head (tied
embedding) and the KV of every sequence. Prefill is compute-bound: FLOPs over
the tensor-core peak, floored by one read of the weights. Byte counts come
from the checkpoint's safetensors headers, so they are the stored formats
(NVFP4 experts with FP8 block scales, BF16 elsewhere), not a parameter count
times an assumed width.
"""

import glob
import json
import os
import re
import struct
from collections import defaultdict
from typing import Dict, List, Sequence

from gate import config

DRAM_BYTES_PER_S = 1792e9
# RTX 5090 dense tensor peaks (no sparsity), TFLOP/s, from the RTX Blackwell
# GPU architecture whitepaper. `gate peaks` measures the achievable ones.
SPEC_PEAK_TFLOPS = {"bf16": 209.5, "fp8": 419.0, "nvfp4": 838.0}

_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1, "I8": 1}


def checkpoint_tensors(model_dir: str = config.MODEL_DIR) -> Dict[str, Dict]:
    out = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            numel = 1
            for d in meta["shape"]:
                numel *= d
            out[name] = {"dtype": meta["dtype"], "shape": meta["shape"], "bytes": numel * _DTYPE_BYTES[meta["dtype"]]}
    return out


# Refs whose env sets this serve every o_proj as FP8 E4M3 weight-only with an fp32 scale per
# output row (T3b, base4 on): decode reads those bytes, prefill upcasts to bf16 for cuBLAS.
FP8_O_PROJ_ENV = "SGLANG_OPT_USE_TRITON_SMALL_M_FP8_WEIGHT_GEMM"


def served_tensors(tensors: Dict[str, Dict], fp8_o_proj: bool) -> Dict[str, Dict]:
    """The checkpoint tensors in the format the server keeps in memory."""
    if not fp8_o_proj:
        return tensors
    out = dict(tensors)
    for name, t in tensors.items():
        if name.endswith(".self_attn.o_proj.weight") and t["dtype"] == "BF16":
            n_out, n_in = t["shape"]
            out[name] = {"dtype": "F8_E4M3", "shape": t["shape"], "bytes": n_out * n_in + 4 * n_out}
    return out


def component_of(name: str) -> str:
    if "vision" in name or "embed_vision" in name or "audio" in name:
        return "vision_unused"
    if "embed_tokens" in name:
        return "lm_head_tied_embed"
    if ".experts." in name:
        return "experts"
    if ".self_attn." in name:
        return "attention_weights"
    if ".mlp." in name:
        return "dense_mlp"
    if ".router." in name:
        return "router"
    return "norms_misc"


def layer_of(name: str) -> int:
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def weight_table(tensors: Dict[str, Dict]) -> Dict:
    by_comp = defaultdict(int)
    expert_bytes_per_layer = defaultdict(int)
    for name, t in tensors.items():
        comp = component_of(name)
        by_comp[comp] += t["bytes"]
        if comp == "experts":
            expert_bytes_per_layer[layer_of(name)] += t["bytes"]
    text_cfg = _text_config()
    n_exp = text_cfg["num_experts"]
    return {
        "bytes_by_component": dict(by_comp),
        "bytes_per_expert_by_layer": {l: b / n_exp for l, b in sorted(expert_bytes_per_layer.items())},
    }


def _text_config() -> Dict:
    with open(os.path.join(config.MODEL_DIR, "config.json")) as f:
        return json.load(f)["text_config"]


def kv_bytes_per_token(kv_elem_bytes: float) -> Dict[str, float]:
    """KV bytes one token adds per layer type: one K and one V copy per layer.

    attention_k_eq_v shares the projection weight only. The cached K is
    k_norm(x W_k) with RoPE and the cached V is v_norm(x W_k), two different
    tensors (Gemma4Attention.forward), so full-attention layers store two
    copies like sliding layers.
    """
    c = _text_config()
    sliding = c["num_key_value_heads"] * c["head_dim"] * 2 * kv_elem_bytes
    full = c["num_global_key_value_heads"] * c["global_head_dim"] * 2 * kv_elem_bytes
    return {"sliding": sliding, "full": full}


def layer_types() -> List[str]:
    return _text_config()["layer_types"]


def decode_step_bytes(contexts: Sequence[int], distinct_experts_by_layer: Dict[int, float], kv_elem_bytes: float,
                      weights: Dict) -> Dict[str, float]:
    """Bytes one decode step must move for a batch whose sequences hold `contexts` tokens."""
    c = _text_config()
    comp = weights["bytes_by_component"]
    per_expert = weights["bytes_per_expert_by_layer"]
    kv = kv_bytes_per_token(kv_elem_bytes)
    window = c["sliding_window"]
    types = layer_types()
    n_sliding = sum(1 for t in types if t == "sliding_attention")
    n_full = len(types) - n_sliding
    kv_read = sum(n_sliding * min(ctx, window) * kv["sliding"] + n_full * ctx * kv["full"] for ctx in contexts)
    kv_write = len(contexts) * (n_sliding * kv["sliding"] + n_full * kv["full"])
    experts = sum(distinct_experts_by_layer[l] * per_expert[l] for l in per_expert)
    table = {
        "attention_weights": comp["attention_weights"],
        "dense_mlp": comp["dense_mlp"],
        "router": comp["router"],
        "norms_misc": comp["norms_misc"],
        "lm_head_tied_embed": comp["lm_head_tied_embed"],
        "routed_experts": experts,
        "kv_read": kv_read,
        "kv_write": kv_write,
    }
    table["total"] = sum(table.values())
    return table


def decode_sol(batch: int, prompt: int, decode: int, distinct_experts_by_layer: Dict[int, float],
               kv_elem_bytes: float, weights: Dict) -> Dict:
    """SOL time of the decode phase: decode-1 steps after the first token (which prefill produced)."""
    steps = []
    for i in range(1, decode):
        steps.append(decode_step_bytes([prompt + i] * batch, distinct_experts_by_layer, kv_elem_bytes, weights))
    mean_table = {k: sum(s[k] for s in steps) / len(steps) for k in steps[0]}
    total_s = sum(s["total"] for s in steps) / DRAM_BYTES_PER_S
    return {"steps": len(steps), "mean_step_bytes": mean_table, "sol_step_ms": 1e3 * total_s / len(steps),
            "sol_decode_s": total_s}


def prefill_flops(batch: int, prompt: int, tensors: Dict[str, Dict]) -> Dict[str, float]:
    """FLOPs of prefilling `batch` prompts of `prompt` tokens, by precision bucket of the checkpoint."""
    c = _text_config()
    tokens = batch * prompt
    linear_bf16 = 0
    expert_params_per_layer = defaultdict(int)
    for name, t in tensors.items():
        comp = component_of(name)
        # FP8 o_proj counts here too: its prefill runs cuBLAS over the bf16 upcast.
        if comp in ("attention_weights", "dense_mlp", "router") and len(t["shape"]) == 2:
            linear_bf16 += t["shape"][0] * t["shape"][1]
        elif comp == "experts" and name.endswith(".weight"):
            # NVFP4 packs two 4-bit values per uint8 along the input dim.
            expert_params_per_layer[layer_of(name)] += t["shape"][0] * t["shape"][1] * 2
    n_exp, top_k = c["num_experts"], c["top_k_experts"]
    expert_active = sum(p / n_exp for p in expert_params_per_layer.values()) * top_k
    window = c["sliding_window"]
    attn = 0
    for lt in layer_types():
        hd = c["head_dim"] if lt == "sliding_attention" else c["global_head_dim"]
        keys = sum(min(t + 1, window) if lt == "sliding_attention" else t + 1 for t in range(prompt))
        attn += 2 * 2 * c["num_attention_heads"] * hd * keys * batch
    lm_head = 2 * batch * c["vocab_size"] * c["hidden_size"]
    return {
        "linear_bf16": 2 * tokens * linear_bf16,
        "experts_nvfp4": 2 * tokens * expert_active,
        "attention": attn,
        "lm_head_last_token": lm_head,
    }


def prefill_sol(flops: Dict[str, float], peaks_tflops: Dict[str, float], weight_bytes: float) -> Dict:
    """Two SOL views: precision as served, and every GEMM at NVFP4 with attention at FP8."""
    as_served = (
        (flops["linear_bf16"] + flops["lm_head_last_token"]) / peaks_tflops["bf16"]
        + flops["experts_nvfp4"] / peaks_tflops["nvfp4"]
        + flops["attention"] / peaks_tflops["bf16"]
    ) / 1e12
    low_precision = (
        (flops["linear_bf16"] + flops["lm_head_last_token"] + flops["experts_nvfp4"]) / peaks_tflops["nvfp4"]
        + flops["attention"] / peaks_tflops["fp8"]
    ) / 1e12
    bytes_floor = weight_bytes / DRAM_BYTES_PER_S
    return {
        "compute_s_as_served": as_served,
        "compute_s_nvfp4_fp8": low_precision,
        "weight_read_floor_s": bytes_floor,
        "sol_s_as_served": max(as_served, bytes_floor),
        "sol_s_nvfp4_fp8": max(low_precision, bytes_floor),
    }


def distinct_experts_from_routes(routes: List) -> Dict[int, float]:
    """Mean distinct experts per layer per decode step of a batch.

    `routes[r]` is request r's routed experts, int array [steps, layers, top_k]
    from routed_experts_start_len = prompt length: row j is the decode pass that
    consumed generated token j. Step j is the union over the batch; requests of
    unequal prompt length can be a prefill chunk out of step, so this is the
    lockstep approximation of one forward pass.
    """
    import numpy as np

    n_layers = routes[0].shape[1]
    steps = min(r.shape[0] for r in routes)
    sums = np.zeros(n_layers)
    for k in range(steps):
        stacked = np.concatenate([r[k] for r in routes], axis=1)
        sums += [len(np.unique(stacked[layer])) for layer in range(n_layers)]
    return {layer: float(sums[layer] / steps) for layer in range(n_layers)}
