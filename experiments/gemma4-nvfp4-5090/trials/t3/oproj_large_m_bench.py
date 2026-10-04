"""W13 T3d step 0: o_proj FP8 weight-only above the T3b kernel's max_m (32).

base4's o_proj stores E4M3 weights (T3b). For M <= 32 it runs the small-M Triton kernel; above
that it upcasts the whole weight to bf16 and calls cuBLAS, which W12 measured at 2.1-2.5x cuBLAS
BF16 on qkv shapes at M=48-256. M=48 is every B=8 verify (k=5), M=192 every W32 verify, and
M >= 512 the prefill chunks. This bench picks, per M, the fastest route that keeps one E4M3 copy
of the weight and the same numerics tier (exact bf16 upcast per tile, fp32 accumulation):

  cublas_bf16       F.linear on a bf16 weight: the "dequant once" route (keeps a second copy,
                    +0.41 GB); listed as the bar, not as a candidate unless nothing else gets close
  upcast_cublas     the current M > 32 route
  fp8w_bm*_bn*_...  the T3b kernel with BLOCK_M capped (one program per M block, so the E4M3
                    weight is re-read from L2 per M block) over a tile sweep
  a8_rowwise        torch._scaled_mm with per-token activation quantization: informational only,
                    it is a different numerics tier (activation rounding)

Method as t3/gemm_microbench.py: CUDA graph of 64 calls, best of 7 replays, weights rotated over
>= 256 MB.

  python oproj_large_m_bench.py --out w13-oproj-large.json
"""

import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F
import triton

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gemm_microbench import L2_ROTATE_BYTES, time_graph  # noqa: E402
from gemm_microbench_w7 import _single_pass_kernel, quantize_weight_per_channel  # noqa: E402

SHAPES = [("o_proj_sliding", 2816, 4096), ("o_proj_full", 2816, 8192)]
MS = (32, 48, 64, 96, 128, 192, 256, 384, 512, 1024, 2048, 4096, 8192)
# (block_m, block_n, block_k, num_stages, num_warps); shared memory per stage is
# (BM + BN) * BK * 2 bytes for the bf16 tiles plus BN * BK for the fp8 load, under ~99 KB.
TILES = [
    (bm, bn, bk, ns, nw)
    for bm in (16, 32, 64, 128)
    for bn in (32, 64, 128)
    for bk in (64, 128, 256)
    for ns in (3, 4)
    for nw in (4, 8)
    if ns * ((bm + bn) * bk * 2 + bn * bk) <= 99 * 1024 and not (bm >= 64 and nw == 4 and bn == 128)
]


def capped_fp8(x, w8, s, bm, bn, bk, ns, nw):
    m, k = x.shape
    n = w8.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    grid = (triton.cdiv(n, bn), triton.cdiv(m, bm))

    def run():
        _single_pass_kernel[grid](x, w8, s, out, m, n, k, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                                  HAS_SCALE=True, num_warps=nw, num_stages=ns)
        return out

    return run


def a8_rowwise(x, w8, s):
    w8t = w8.t()
    s_row = s.view(1, -1)

    def run():
        xs = x.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 448.0
        xq = (x.float() / xs).to(torch.float8_e4m3fn)
        return torch._scaled_mm(xq, w8t, scale_a=xs, scale_b=s_row, out_dtype=torch.bfloat16)

    return run


def candidates(x, w, w8, s, m):
    c = {
        "cublas_bf16": lambda: F.linear(x, w),
        "upcast_cublas": lambda: (F.linear(x, w8.to(torch.bfloat16)) * s).to(torch.bfloat16),
        "a8_rowwise": a8_rowwise(x, w8, s),
    }
    for t in TILES:
        bm = t[0]
        # A BLOCK_M much wider than M is all padding; one narrower tile already covers it.
        if bm > max(16, triton.next_power_of_2(m)):
            continue
        c["fp8w_bm%d_bn%d_bk%d_s%d_w%d" % t] = capped_fp8(x, w8, s, *t)
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--ms", default=",".join(map(str, MS)))
    a = p.parse_args()
    ms = [int(v) for v in a.ms.split(",")]
    torch.manual_seed(0)
    rows = []
    for name, n, k in SHAPES:
        n_copies = max(2, math.ceil(L2_ROTATE_BYTES / (n * k * 2)))
        ws = [torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02 for _ in range(n_copies)]
        q = [quantize_weight_per_channel(w) for w in ws]
        for m in ms:
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            ref32 = x.float() @ ws[0].float().t()
            names = list(candidates(x, ws[0], q[0][0], q[0][1], m))
            for cname in names:
                rec = {"shape": name, "N": n, "K": k, "M": m, "cand": cname}
                try:
                    per_copy = [candidates(x, ws[i], q[i][0], q[i][1], m)[cname] for i in range(n_copies)]
                    y = per_copy[0]()
                    torch.cuda.synchronize()
                    rec["rel_l2_vs_fp32"] = ((y.float() - ref32).norm() / ref32.norm()).item()
                    rec["us"] = time_graph(lambda i: per_copy[i], n_copies)
                except Exception as e:  # noqa: BLE001 - an unsupported tile is a recorded result
                    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                torch.cuda.synchronize()
                rows.append(rec)
            best = min((r for r in rows if r["shape"] == name and r["M"] == m and "us" in r
                        and r["cand"].startswith("fp8w")), key=lambda r: r["us"], default=None)
            ref = {r["cand"]: r.get("us") for r in rows if r["shape"] == name and r["M"] == m}
            print(name, m, "cublas_bf16", ref.get("cublas_bf16"), "upcast", ref.get("upcast_cublas"),
                  "a8", ref.get("a8_rowwise"), "best_fp8w", best and (best["cand"], best["us"]), flush=True)
            del x
        del ws, q
        torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(), "torch": torch.__version__,
                   "triton": triton.__version__, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
