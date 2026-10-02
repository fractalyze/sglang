"""Byte / FLOP model and speed-of-light (SOL) time per component for
nvidia/Gemma-4-26B-A4B-NVFP4 on one RTX 5090.

SOL per component = max(FLOPs / peak, bytes / BW) at the component's shape
(SOL-ExecBench SOLAR style). sol_fraction = SOL / achieved (1.0 = at bound).
BW and peaks default to measured microbench ceilings (see PROFILE.md), not
datasheet numbers.

Shapes come from the checkpoint config.json / hf_quant_config.json:
- experts NVFP4 (4-bit e2m1 + one fp8 scale per 16 -> 0.5625 B/param);
- attention projections, dense MLP, router and the tied lm_head are BF16;
- full-attention layers (k_eq_v) still load a V shard in qkv_proj because the
  loader copies K weights into it, so V columns are real bytes and FLOPs.
"""

import argparse
import json

H = 2816
LAYERS = 30
FULL_LAYERS = 5
SWA_LAYERS = LAYERS - FULL_LAYERS
WINDOW = 1024
NQ = 16
SWA_KV, SWA_HD = 8, 256
FULL_KV, FULL_HD = 2, 512
DENSE_I = 2112
MOE_I = 704
TOPK = 8
VOCAB = 262144
BF16 = 2
FP4_B_PER_PARAM = 0.5 + 1 / 16


def components(B, ctx, distinct_experts, kv_bytes, prefill_tokens=None):
    """Return {name: (bytes, flops)} for one forward.

    Decode: B tokens, each sequence has `ctx` cached tokens.
    Prefill: `prefill_tokens` new tokens per sequence (B sequences, no cache).
    """
    T = B if prefill_tokens is None else B * prefill_tokens
    c = {}

    def gemm(name, k, n, wbytes_per_param, count):
        w = k * n * wbytes_per_param * count
        a = T * (k + n) * BF16 * count
        f = 2 * T * k * n * count
        prev = c.get(name, (0, 0))
        c[name] = (prev[0] + w + a, prev[1] + f)

    gemm("qkv_proj", H, NQ * SWA_HD + 2 * SWA_KV * SWA_HD, BF16, SWA_LAYERS)
    gemm("qkv_proj", H, NQ * FULL_HD + 2 * FULL_KV * FULL_HD, BF16, FULL_LAYERS)
    gemm("o_proj", NQ * SWA_HD, H, BF16, SWA_LAYERS)
    gemm("o_proj", NQ * FULL_HD, H, BF16, FULL_LAYERS)
    gemm("dense_mlp", H, 2 * DENSE_I, BF16, LAYERS)
    gemm("dense_mlp", DENSE_I, H, BF16, LAYERS)
    gemm("router", H, 128, BF16, LAYERS)

    expert_params = H * 2 * MOE_I + MOE_I * H
    if prefill_tokens is None:
        e_bytes = distinct_experts * expert_params * FP4_B_PER_PARAM * LAYERS
    else:
        e_bytes = 128 * expert_params * FP4_B_PER_PARAM * LAYERS
    e_flops = 2 * T * TOPK * expert_params * LAYERS
    # SOL counts only the unavoidable activation traffic: read x, write y.
    e_act = T * 2 * H * BF16 * LAYERS
    c["moe_experts"] = (e_bytes + e_act, e_flops)

    def attn(nkv, hd, nlayers, attended_per_seq, kv_tokens_read_per_seq):
        kv = B * kv_tokens_read_per_seq * nkv * hd * 2 * kv_bytes * nlayers
        q_o = T * NQ * hd * 2 * BF16 * nlayers
        f = 4 * B * attended_per_seq * NQ * hd * nlayers
        return kv + q_o, f

    if prefill_tokens is None:
        swa_len = min(ctx, WINDOW)
        c["attn_swa"] = attn(SWA_KV, SWA_HD, SWA_LAYERS, swa_len, swa_len)
        c["attn_full"] = attn(FULL_KV, FULL_HD, FULL_LAYERS, ctx, ctx)
    else:
        P = prefill_tokens
        swa_pairs = sum(min(i + 1, WINDOW) for i in range(P))
        full_pairs = P * (P + 1) // 2
        c["attn_swa"] = attn(SWA_KV, SWA_HD, SWA_LAYERS, swa_pairs, P)
        c["attn_full"] = attn(FULL_KV, FULL_HD, FULL_LAYERS, full_pairs, P)

    # Prefill only computes logits for the last token of each sequence.
    lm_rows = B
    c["lm_head"] = (
        VOCAB * H * BF16 + lm_rows * VOCAB * 4 * 3,
        2 * lm_rows * H * VOCAB,
    )
    # Norm/residual/rope glue: ~8 activation-sized read+writes per layer.
    c["norms_glue"] = (T * H * BF16 * 2 * 8 * LAYERS, 0)
    return c


def sol_table(comp, bw, peaks):
    rows = {}
    for name, (byt, flops) in comp.items():
        peak = peaks["fp4"] if name == "moe_experts" else peaks["bf16"]
        t_mem = byt / bw
        t_cmp = flops / peak
        rows[name] = dict(
            bytes=byt,
            flops=flops,
            sol_us=max(t_mem, t_cmp) * 1e6,
            bound="mem" if t_mem >= t_cmp else "compute",
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bw-gbs", type=float, default=1792)
    ap.add_argument("--bf16-tflops", type=float, default=209.5)
    ap.add_argument("--fp4-tflops", type=float, default=838)
    ap.add_argument("--experts-json", default=None,
                    help='{"1": e1, "8": e8, "32": e32} distinct experts/layer')
    ap.add_argument("--kv-bytes", type=float, default=2)
    args = ap.parse_args()
    bw = args.bw_gbs * 1e9
    peaks = dict(bf16=args.bf16_tflops * 1e12, fp4=args.fp4_tflops * 1e12)
    # Uniform-routing expectation 128 * (1 - (1 - 8/128)^B); real routing is
    # skewed, so measured values (--experts-json) should be lower.
    experts = {str(b): 128 * (1 - (1 - TOPK / 128) ** b) for b in (1, 8, 32)}
    if args.experts_json:
        experts.update(json.load(open(args.experts_json)))

    out = {}
    for B in (1, 8, 32):
        comp = components(B, ctx=1024 + 64, distinct_experts=experts[str(B)],
                          kv_bytes=args.kv_bytes)
        out[f"decode_B{B}"] = sol_table(comp, bw, peaks)
    comp = components(8, ctx=0, distinct_experts=128, kv_bytes=args.kv_bytes,
                      prefill_tokens=1024)
    out["prefill_B8x1024"] = sol_table(comp, bw, peaks)

    for key, rows in out.items():
        tot = sum(r["sol_us"] for r in rows.values())
        byt = sum(r["bytes"] for r in rows.values())
        print(f"\n== {key}: SOL {tot:.0f} us, {byt / 1e9:.2f} GB")
        for name, r in rows.items():
            print(f"  {name:12s} {r['bytes'] / 1e6:9.1f} MB {r['flops'] / 1e9:10.1f} GF "
                  f"SOL {r['sol_us']:8.1f} us ({r['bound']})")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
