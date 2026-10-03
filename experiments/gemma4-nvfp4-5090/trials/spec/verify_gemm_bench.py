"""W12 T-SPEC6 step 0: the target's qkv_proj and lm_head at speculative verify widths.

Under MTP with k drafts, a target forward at batch B runs M = (1 + k) * B rows: M=6 at B=1, k=5
and up to M=48 at B=8. W10's profile left these layers on cuBLAS's SM80 WMMA fallback (1.96 ms
of a B=1 round). T3c measured the same layers at plain decode M (1-32) and retired at -0.92% W1,
so this bench asks whether the verify widths are a different shape class.

Same method as t3/gemm_microbench_w7.py: CUDA graph of 64 calls, best of 7 replays, weights
rotated over >= 256 MB. Candidates per (shape, M):

  cublas_default   F.linear (the current path)
  bf16_*           T3's single-pass Triton kernel, BF16 weight, over a tile sweep
  fp8w_*           the same kernel reading an E4M3 weight with a per-output-channel scale
                   (T3b's reviewed weight-only path; changes target numerics)

  python verify_gemm_bench.py --out w12-verify.json [--ms 4,6,8,12,24,36,48]
  python verify_gemm_bench.py --large-m --out w12-verify-large.json

--large-m covers the W32 verify widths (M = 6 * 32 = 192): qkv only, the FP8 kernel with BLOCK_M
capped at 32 or 64 (one program per M block, so the E4M3 weight is re-read from L2) against cuBLAS
BF16 and the current M > max_m route, an exact bf16 upcast of the E4M3 weight then cuBLAS.
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
import triton

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "t3"))

from gemm_microbench import L2_ROTATE_BYTES, time_graph  # noqa: E402
from gemm_microbench_w7 import quantize_weight_per_channel, single_pass  # noqa: E402

SHAPES = [
    ("qkv_sliding", 8192, 2816),
    ("qkv_full", 10240, 2816),
    ("lm_head", 262144, 2816),
]
# The configs that won any T3/T3b/T3c/T-SPEC4 cell, plus their neighbours; BLOCK_M grows with M,
# so the shared-memory budget caps BLOCK_N x BLOCK_K at M=48 (BLOCK_M=64).
CONFIGS = [(bn, bk, ns, 4) for bn in (16, 32, 64, 128) for bk in (64, 128, 256) for ns in (3, 4)
           if bn * bk <= 16384]


def candidates(x, w, w8, s):
    c = {"cublas_default": lambda: F.linear(x, w)}
    for cfg in CONFIGS:
        c["bf16_bn%d_bk%d_s%d_w%d" % cfg] = single_pass(x, w, None, *cfg)
        c["fp8w_bn%d_bk%d_s%d_w%d" % cfg] = single_pass(x, w8, s, *cfg)
    return c


LARGE_MS = (48, 64, 96, 128, 192, 256)
LARGE_CONFIGS = [(64, 128, 4), (32, 128, 4), (64, 128, 3), (64, 64, 4)]


def capped_fp8(x, w8, s, block_n, block_k, num_stages, block_m):
    from gemm_microbench_w7 import _single_pass_kernel

    m, k = x.shape
    n = w8.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    grid = (triton.cdiv(n, block_n), triton.cdiv(m, block_m))

    def run():
        _single_pass_kernel[grid](x, w8, s, out, m, n, k, BLOCK_M=block_m, BLOCK_N=block_n,
                                  BLOCK_K=block_k, HAS_SCALE=True, num_warps=4, num_stages=num_stages)
        return out

    return run


def large_candidates(x, w, w8, s):
    c = {"cublas_default": lambda: F.linear(x, w),
         "fp8_upcast_cublas": lambda: (F.linear(x, w8.to(torch.bfloat16)) * s).to(torch.bfloat16)}
    for bm in (32, 64):
        for cfg in LARGE_CONFIGS:
            c["fp8w_bn%d_bk%d_s%d_bm%d" % (*cfg, bm)] = capped_fp8(x, w8, s, *cfg, bm)
    return c


def run_large(bw_gbs):
    torch.manual_seed(0)
    rows = []
    for name, n, k in SHAPES[:2]:
        n_copies = max(2, math.ceil(L2_ROTATE_BYTES / (n * k * 2)))
        ws = [torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02 for _ in range(n_copies)]
        q = [quantize_weight_per_channel(w) for w in ws]
        for m in LARGE_MS:
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            ref32 = x.float() @ ws[0].float().t()
            for cname in large_candidates(x, ws[0], q[0][0], q[0][1]):
                rec = {"shape": name, "N": n, "K": k, "M": m, "cand": cname}
                try:
                    per_copy = [large_candidates(x, ws[i], q[i][0], q[i][1])[cname] for i in range(n_copies)]
                    y = per_copy[0]()
                    torch.cuda.synchronize()
                    rec["rel_l2_vs_fp32"] = ((y.float() - ref32).norm() / ref32.norm()).item()
                    rec["us"] = time_graph(lambda i: per_copy[i], n_copies)
                except Exception as e:  # noqa: BLE001 - an unsupported tile is a recorded result
                    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                torch.cuda.synchronize()
                rows.append(rec)
                print(json.dumps(rec), flush=True)
        del ws, q
        torch.cuda.empty_cache()
    return rows


def run(ms, bw_gbs, shapes):
    torch.manual_seed(0)
    rows = []
    for name, n, k in SHAPES:
        if shapes and name not in shapes:
            continue
        wbytes = n * k * 2
        n_copies = max(2, math.ceil(L2_ROTATE_BYTES / wbytes))
        ws = [torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02 for _ in range(n_copies)]
        q = [quantize_weight_per_channel(w) for w in ws]
        for m in ms:
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            ref32 = x.float() @ ws[0].float().t()
            for cname in candidates(x, ws[0], q[0][0], q[0][1]):
                fp8 = cname.startswith("fp8")
                act = (m * k + m * n) * 2
                rec = {"shape": name, "N": n, "K": k, "M": m, "cand": cname,
                       "sol_us": ((wbytes // 2 if fp8 else wbytes) + act) / (bw_gbs * 1e9) * 1e6}
                try:
                    per_copy = [candidates(x, ws[i], q[i][0], q[i][1])[cname] for i in range(n_copies)]
                    y = per_copy[0]()
                    torch.cuda.synchronize()
                    rec["rel_l2_vs_fp32"] = ((y.float() - ref32).norm() / ref32.norm()).item()
                    t0 = time.time()
                    rec["us"] = time_graph(lambda i: per_copy[i], n_copies)
                    rec["sol_fraction"] = rec["sol_us"] / rec["us"]
                    rec["wall_s"] = time.time() - t0
                except Exception as e:  # noqa: BLE001 - an unsupported tile is a recorded result
                    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                torch.cuda.synchronize()
                rows.append(rec)
                print(json.dumps(rec), flush=True)
            del x
        del ws, q
        torch.cuda.empty_cache()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--bw-gbs", type=float, default=1650.0)
    p.add_argument("--ms", default="4,6,8,12,24,36,48")
    p.add_argument("--shapes", default="", help="comma-separated subset of " + ",".join(s[0] for s in SHAPES))
    p.add_argument("--large-m", action="store_true")
    a = p.parse_args()
    rows = run_large(a.bw_gbs) if a.large_m else run([int(v) for v in a.ms.split(",")], a.bw_gbs, set(filter(None, a.shapes.split(","))))
    with open(a.out, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(), "torch": torch.__version__,
                   "triton": triton.__version__, "bw_gbs": a.bw_gbs, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
