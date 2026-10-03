"""Per-component GPU time per decode step from a SGLang torch-profiler DECODE
trace of Gemma-4-26B-A4B (flashinfer_cutlass MoE, triton attention).

Decode runs inside a CUDA graph, so kernels carry no CPU-op parent; they are
labelled by name plus position. A step ends at the lm_head GEMM (the largest
BF16 GEMM); a layer ends at `_gemma_dual_rmsnorm_residual_kernel`. Within a
layer the BF16 GEMMs come in a fixed order: qkv_proj, o_proj, dense gate_up,
dense down, router (split-K). Layers 5, 11, 17, 23, 29 are full attention.

Usage: python classify_trace.py <DECODE.trace.json.gz> [--json-out f] [--dump-seq N]
"""

import argparse
import collections
import gzip
import json
import re
import statistics

FULL_LAYERS = {5, 11, 17, 23, 29}
LAYER_GEMMS = ["qkv_proj", "o_proj", "dense_mlp", "dense_mlp", "router_gemm"]
ORDER = [
    "qkv_proj", "o_proj", "dense_mlp", "router_gemm", "router_topk", "moe_gemm", "moe_glue",
    "attn_swa", "attn_full", "rope_kvwrite", "norms", "lm_head", "sampling", "step_setup",
    "other", "idle",
]


def load_kernels(path):
    trace = json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path))
    ev = [e for e in trace["traceEvents"]
          if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    ev.sort(key=lambda e: e["ts"])
    return ev


def is_bf16_gemm(n):
    # B>=2: cuBLAS picks the SM80 WMMA fallback on SM120; B=1: cuBLAS gemvx.
    return "wmma_tensorop_bf16" in n or "gemvx" in n or "nvjet" in n


def classify_step(kernels):
    """kernels: one decode step, ending with the lm_head GEMM.

    Returns (us per component, launches per component, layers seen).
    """
    t = collections.Counter()
    n = collections.Counter()
    before = collections.Counter()
    layer, gemm_i, in_layers, after_attn = 0, 0, False, False
    for k in kernels[:-1]:
        before.clear()
        before.update(t)
        name, d = k["name"], k["dur"]
        if not in_layers:
            # Everything before the first layer's input norm is per-step setup.
            if "RMSNormKernel" in name:
                in_layers = True
            else:
                t["step_setup"] += d
                n["step_setup"] += 1
                continue
        if "GroupProblemShape" in name:
            t["moe_gemm"] += d
        elif "tensorrt_llm::kernels::cutlass_kernels" in name:
            t["moe_glue"] += d
        elif "_gemma4_routing_kernel" in name:
            t["router_topk"] += d
        elif "splitKreduce" in name:
            t["router_gemm"] += d
        elif is_bf16_gemm(name):
            t[LAYER_GEMMS[min(gemm_i, len(LAYER_GEMMS) - 1)]] += d
            gemm_i += 1
        elif "stage1" in name or "stage2" in name:
            t["attn_full" if layer in FULL_LAYERS else "attn_swa"] += d
            after_attn = True
        elif "_gemma_dual_rmsnorm_residual_kernel" in name:
            t["norms"] += d
            layer, gemm_i, after_attn = layer + 1, 0, False
        elif "norm" in name.lower():
            t["norms"] += d
        elif "rope" in name or "store_kvcache" in name or (gemm_i == 1 and not after_attn):
            # Between qkv_proj and attention: RoPE, FP8 KV quantize, KV store.
            t["rope_kvwrite"] += d
        elif "act_and_mul" in name:
            t["dense_mlp"] += d
        else:
            t["other"] += d
        for c in t:
            if t[c] != before[c]:
                n[c] += 1
    t["lm_head"] += kernels[-1]["dur"]
    n["lm_head"] += 1
    return t, n, layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--dump-seq", type=int, default=0)
    args = ap.parse_args()

    ev = load_kernels(args.trace)
    heads = [i for i, e in enumerate(ev) if is_bf16_gemm(e["name"]) and e["dur"] > 300]
    if args.dump_seq:
        for e in ev[heads[0] + 1: heads[0] + 1 + args.dump_seq]:
            print(f"{e['dur']:8.1f}  {re.sub(r'<.*', '', e['name'])[:100]}")
        return

    steps = []
    # Skip the partial step before the first lm_head; each later step runs
    # from the kernel after one lm_head to the next lm_head inclusive.
    for a, b in zip(heads, heads[1:]):
        ks = ev[a + 1: b + 1]
        t, launches, n_layers = classify_step(ks)
        # Sampling (softcap, argmax, index ops) runs right after lm_head and is
        # folded into step_setup by position; split it out by name.
        samp = sum(k["dur"] for k in ks[:8] if re.search(r"softcap|reduce_kernel|unrolled_elementwise|Memset", k["name"]))
        t["sampling"] += samp
        t["step_setup"] -= samp
        span = ks[-1]["ts"] + ks[-1]["dur"] - ks[0]["ts"]
        t["idle"] = span - sum(k["dur"] for k in ks)
        steps.append(dict(t=t, launches=launches, span=span, n_layers=n_layers, n_kernels=len(ks)))

    mean = {c: statistics.mean(s["t"].get(c, 0) for s in steps) for c in ORDER}
    span = statistics.mean(s["span"] for s in steps)
    out = dict(
        trace=args.trace,
        steps=len(steps),
        layers_per_step=sorted({s["n_layers"] for s in steps}),
        kernels_per_step=statistics.mean(s["n_kernels"] for s in steps),
        step_span_us=span,
        component_us=mean,
        component_launches={c: statistics.mean(s["launches"].get(c, 0) for s in steps) for c in ORDER},
    )
    print(f"steps={len(steps)} layers={out['layers_per_step']} kernels/step={out['kernels_per_step']:.0f} span={span:.0f} us")
    for c in ORDER:
        print(f"  {c:13s} {mean[c]:8.1f} us  {mean[c] / span:6.1%}")
    if args.json_out:
        json.dump(out, open(args.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
