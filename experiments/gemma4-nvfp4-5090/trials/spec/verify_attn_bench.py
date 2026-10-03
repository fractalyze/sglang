"""T-SPEC2 microbench: Triton verify attention on Gemma-4 shapes (kernel level).

Times extend_attention_fwd (today's TARGET_VERIFY path) against the split-KV
verify kernel in its CUDA mode (occupancy splits, tiled stage 2, sliding window)
per layer type, and checks they agree. Each timing is the median of
triton.testing.do_bench; the KV pool is large enough that every call streams its
KV slice from DRAM rather than L2.

Shapes: 16 query heads; sliding layers head_dim 256 / 8 KV heads / window 1024
(KV slice = last 1024 prefix tokens); full layers head_dim 512 / 2 KV heads;
FP8 E4M3 KV with per-layer scales; 1+k = 6 (k=5) and 4 (k=3) verify rows.

  python verify_attn_bench.py [--ctx 1100] [--out results.json]
"""

import argparse
import json

import torch
import triton

from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
from sglang.kernels.ops.attention.verify_splitkv import verify_splitkv_fwd

H_Q = 16
LAYERS = {"sliding": dict(head_dim=256, h_kv=8, window=1024), "full": dict(head_dim=512, h_kv=2, window=-1)}
POOL_TOKENS = 1 << 20


def _inputs(bs, l_ext, ctx, head_dim, h_kv, window, device="cuda"):
    prefix = min(ctx, window) if window > 0 else ctx
    k_pool = torch.randn(POOL_TOKENS, h_kv, head_dim, device=device).to(torch.float8_e4m3fn)
    v_pool = torch.randn(POOL_TOKENS, h_kv, head_dim, device=device).to(torch.float8_e4m3fn)
    # Each sequence's KV sits at a random place in the pool, as after real traffic.
    kv_indices = torch.cat(
        [torch.randperm(POOL_TOKENS, device=device)[:prefix] for _ in range(bs)]
    ).to(torch.int64)
    kv_indptr = torch.arange(0, bs * prefix + 1, prefix, dtype=torch.int32, device=device)
    qo_indptr = torch.arange(0, bs * l_ext + 1, l_ext, dtype=torch.int32, device=device)
    q = torch.randn(bs * l_ext, H_Q, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(bs * l_ext, h_kv, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(bs * l_ext, h_kv, head_dim, dtype=torch.bfloat16, device=device)
    return q, k, v, k_pool, v_pool, qo_indptr, kv_indptr, kv_indices


def _bench_one(bs, l_ext, ctx, layer, sm_count):
    cfg = LAYERS[layer]
    q, k, v, kp, vp, qo, kvp, kvi = _inputs(bs, l_ext, ctx, cfg["head_dim"], cfg["h_kv"], cfg["window"])
    sm_scale = 1.0
    o_ref = torch.empty(q.shape[0], H_Q, cfg["head_dim"], dtype=q.dtype, device=q.device)
    o_new = torch.empty_like(o_ref)
    common = (q, k, v)

    def extend():
        extend_attention_fwd(*common, o_ref, kp, vp, qo, kvp, kvi, None, True, None, l_ext, 0.5, 0.25,
                             sm_scale=sm_scale, sliding_window_size=cfg["window"])

    def splitkv():
        ran = verify_splitkv_fwd(*common, o_new, kp, vp, qo, kvp, kvi, None, True, None, l_ext, 0.5, 0.25,
                                 sm_scale=sm_scale, sliding_window_size=cfg["window"],
                                 allow_sliding_window=True, sm_count=sm_count, max_bs=64)
        assert ran

    extend()
    splitkv()
    err = (o_new.float() - o_ref.float()).abs().max().item()
    t_ext = triton.testing.do_bench(extend, warmup=50, rep=300)
    t_new = triton.testing.do_bench(splitkv, warmup=50, rep=300)
    return {"bs": bs, "rows": l_ext, "layer": layer, "extend_us": round(1e3 * t_ext, 1),
            "splitkv_us": round(1e3 * t_new, 1), "speedup": round(t_ext / t_new, 2), "max_abs_err": err}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=1100)
    ap.add_argument("--out")
    args = ap.parse_args()
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    rows = []
    for bs in (1, 8, 32):
        for l_ext in (4, 6):
            for layer in LAYERS:
                r = _bench_one(bs, l_ext, args.ctx, layer, sm_count)
                print(json.dumps(r), flush=True)
                rows.append(r)
    # Per verify forward: 25 sliding + 5 full layers.
    summary = {}
    for bs in (1, 8, 32):
        for l_ext in (4, 6):
            pick = {r["layer"]: r for r in rows if r["bs"] == bs and r["rows"] == l_ext}
            ext = 25 * pick["sliding"]["extend_us"] + 5 * pick["full"]["extend_us"]
            new = 25 * pick["sliding"]["splitkv_us"] + 5 * pick["full"]["splitkv_us"]
            summary[f"bs{bs}_rows{l_ext}"] = {"extend_ms": round(ext / 1e3, 3), "splitkv_ms": round(new / 1e3, 3)}
    print(json.dumps(summary, indent=1))
    if args.out:
        json.dump({"ctx": args.ctx, "sm_count": sm_count, "rows": rows, "per_forward": summary}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
