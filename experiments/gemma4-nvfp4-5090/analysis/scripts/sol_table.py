"""Measured component table: bytes/step, SOL, launch floor, achieved, sol_fraction.

Joins the byte model (sol_model.components, with the measured distinct-expert
counts) to the per-component decode times from classify_trace.py. SOL uses the
best measured DRAM rate on this host (BF16 lm_head GEMV, microbench); the
launch floor is launches x the measured CUDA-graph per-node cost. Achieved
times come from the profiled run (profiler overhead ~1% at B=8, ~8% at B=1);
the e2e row also shows the unprofiled timed step.

Usage: python sol_table.py --batch 8 --components results/components_b8.json \
         --experts results/experts_b8.json --microbench results/microbench_bs2.json \
         --timed-step-ms 9.30
"""

import argparse
import json

from sol_model import components

# Trace components that make up each byte-model component.
JOIN = {
    "qkv_proj": ["qkv_proj"],
    "o_proj": ["o_proj"],
    "dense_mlp": ["dense_mlp"],
    "router": ["router_gemm", "router_topk"],
    "moe_experts": ["moe_gemm", "moe_glue"],
    "attn_swa": ["attn_swa"],
    "attn_full": ["attn_full"],
    "lm_head": ["lm_head", "sampling"],
    "norms_glue": ["norms", "rope_kvwrite"],
    "overhead": ["step_setup", "other", "idle"],
}
LABEL = {
    "qkv_proj": "qkv_proj (BF16)",
    "o_proj": "o_proj (BF16)",
    "dense_mlp": "dense MLP (BF16 GEMMs + GeGLU)",
    "router": "router GEMM + top-k",
    "moe_experts": "routed experts (NVFP4 grouped GEMM + glue)",
    "attn_swa": "attention, sliding (FP8 KV)",
    "attn_full": "attention, full (FP8 KV)",
    "lm_head": "lm_head (BF16) + sampling",
    "norms_glue": "norms, RoPE, KV write",
    "overhead": "step setup + idle gaps",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--components", required=True)
    ap.add_argument("--experts", required=True)
    ap.add_argument("--microbench", required=True)
    ap.add_argument("--timed-step-ms", type=float, required=True)
    ap.add_argument("--fp4-tflops", type=float, default=838.0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    comp = json.load(open(args.components))
    us, launches = comp["component_us"], comp["component_launches"]
    mb = json.load(open(args.microbench))
    bw = max(mb["read_GBs"], mb["lm_head_M8_GBs"]) * 1e9
    floor_us = mb["graph_launch_floor_us"]
    e = json.load(open(args.experts))["distinct_experts_mean"]
    model = components(args.batch, ctx=1024 + 64, distinct_experts=e, kv_bytes=1)
    peaks = {"bf16": mb["bf16_tflops"] * 1e12, "fp4": args.fp4_tflops * 1e12}

    rows = []
    for key, parts in JOIN.items():
        achieved = sum(us.get(p, 0) for p in parts)
        n = sum(launches.get(p, 0) for p in parts)
        byt, flops = model.get(key, (0, 0))
        peak = peaks["fp4"] if key == "moe_experts" else peaks["bf16"]
        sol = max(byt / bw, flops / peak) * 1e6
        rows.append(dict(component=key, label=LABEL[key], bytes=byt, sol_us=sol,
                         launches=n, floor_us=n * floor_us, achieved_us=achieved))
    span = sum(r["achieved_us"] for r in rows)
    for r in rows:
        r["share"] = r["achieved_us"] / span
        r["sol_fraction"] = r["sol_us"] / r["achieved_us"] if r["achieved_us"] else None
        r["gap_us"] = r["achieved_us"] - max(r["sol_us"], r["floor_us"])
        r["rank_key"] = r["share"] * (1 - (r["sol_fraction"] or 0))
    tot_sol = sum(r["sol_us"] for r in rows)
    out = dict(batch=args.batch, distinct_experts=e, bw_GBs=bw / 1e9, launch_floor_us=floor_us,
               traced_step_us=span, timed_step_us=args.timed_step_ms * 1e3, total_sol_us=tot_sol,
               e2e_sol_fraction_timed=tot_sol / (args.timed_step_ms * 1e3), rows=rows)

    print(f"| Component | Bytes/step | SOL µs | Launches (floor µs) | Achieved µs | Share | sol_fraction | Gap µs | share x (1-sol_frac) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda r: -r["rank_key"]):
        sf = f"{r['sol_fraction']:.2f}" if r["sol_fraction"] is not None else "—"
        print(f"| {r['label']} | {r['bytes'] / 1e6:,.0f} MB | {r['sol_us']:,.0f} | {r['launches']:.0f} ({r['floor_us']:,.0f}) "
              f"| {r['achieved_us']:,.0f} | {r['share']:.1%} | {sf} | {r['gap_us']:,.0f} | {r['rank_key']:.3f} |")
    print(f"| **step** | **{sum(r['bytes'] for r in rows) / 1e9:.2f} GB** | **{tot_sol:,.0f}** | "
          f"**{sum(r['launches'] for r in rows):.0f} ({sum(r['floor_us'] for r in rows):,.0f})** | **{span:,.0f}** (timed {args.timed_step_ms * 1e3:,.0f}) "
          f"| | **{tot_sol / (args.timed_step_ms * 1e3):.2f}** (timed) | | |")
    if args.json_out:
        json.dump(out, open(args.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
