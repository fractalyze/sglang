"""W7 step 0: the T3 follow-ups on the exact Gemma-4-26B-A4B shapes.

Two benches, same timing method as gemm_microbench.py (CUDA graph of 64 calls, best of 7
replays, weights rotated over >= 256 MB so every call streams its weight from DRAM):

  t3c  the BF16 small-M linears T3 left on cuBLAS: qkv_proj (sliding / full), the MoE router
       (N=128) and lm_head (N=262144). Candidates: cuBLAS, cuBLASLt, FlashInfer tinygemm, the
       T3 single-pass Triton kernel over a (BLOCK_N, BLOCK_K, stages, warps) sweep and split-K.
  t3b  FP8 o_proj. Control is the T3 BF16 kernel at its tuned config. Candidates: FP8 E4M3
       weight-only (per-output-channel scale) single-pass Triton, W8A8 torch._scaled_mm
       (per-tensor and rowwise, dynamic activation quantization inside the timed call), and
       sgl-kernel fp8_scaled_mm with sgl_per_token_quant_fp8. At M=1024 (prefill) it also times
       dequantize-to-BF16 + cuBLAS, the prefill path a weight-only trial needs.

  python gemm_microbench_w7.py --bench t3c --out t3c.json
  python gemm_microbench_w7.py --bench t3b --out t3b.json
"""

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from gemm_microbench import L2_ROTATE_BYTES, time_graph, triton_splitk

T3C_SHAPES = [
    ("qkv_sliding", 8192, 2816),   # q 16x256 + k 8x256 + v 8x256
    ("qkv_full", 10240, 2816),     # q 16x512 + k 2x512 + v 2x512 (v loads a copy of k)
    ("router", 128, 2816),         # 128 experts
    ("lm_head", 262144, 2816),     # tied embedding, vocab 262144
]
T3B_SHAPES = [
    ("o_proj_sliding", 2816, 4096),
    ("o_proj_full", 2816, 8192),
]
# T3's tuned (BLOCK_N, BLOCK_K, stages) for o_proj, the control of t3b.
T3_OPROJ_CFG = {(2816, 4096): (32, 256, 3), (2816, 8192): (32, 256, 3)}
FP8_MAX = 448.0


@triton.jit
def _single_pass_kernel(
    x_ptr, w_ptr, s_ptr, out_ptr, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, HAS_SCALE: tl.constexpr,
):
    """out = x @ w.T (* s[n] when HAS_SCALE); w is bf16 or fp8 (upcast to bf16 per tile)."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :],
                    mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)
        acc = tl.dot(x, tl.trans(w.to(tl.bfloat16)), acc)
    if HAS_SCALE:
        acc = acc * tl.load(s_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def single_pass(x, w, scale, block_n, block_k, num_stages, num_warps):
    m, k = x.shape
    n = w.shape[0]
    block_m = max(16, triton.next_power_of_2(m))
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    grid = (triton.cdiv(n, block_n), triton.cdiv(m, block_m))

    def run():
        _single_pass_kernel[grid](x, w, scale if scale is not None else w, out, m, n, k,
                                  BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
                                  HAS_SCALE=scale is not None, num_warps=num_warps,
                                  num_stages=num_stages)
        return out

    return run


SP_CONFIGS = [(bn, bk, ns, nw) for bn in (16, 32, 64, 128) for bk in (64, 128, 256)
              for ns in (3, 4) for nw in (4, 8) if not (bn == 128 and bk == 256)]
SPLITK_CONFIGS = [(bn, bk, sk, 4, 3) for bn in (16, 32) for bk in (128, 256) for sk in (2, 4, 8)]
# Router (N=128): finer split-K so each split is 1-3 K tiles.
NARROW_SPLITK_CONFIGS = [(bn, bk, sk, 4, ns) for bn in (16, 32, 64) for bk in (64, 128)
                         for sk in (11, 16, 22) for ns in (2, 3)]


def quantize_weight_per_channel(w):
    s = w.float().abs().amax(dim=1).clamp(min=1e-12) / FP8_MAX
    return (w.float() / s[:, None]).to(torch.float8_e4m3fn), s


def t3c_candidates(x, w, m):
    import flashinfer.gemm as fg

    n = w.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    c = {"cublas_default": lambda: F.linear(x, w),
         "torch_matmul_wT": lambda: torch.matmul(x, w.T)}

    def lt():
        torch.backends.cuda.preferred_blas_library("cublaslt")
        try:
            return F.linear(x, w)
        finally:
            torch.backends.cuda.preferred_blas_library("cublas")

    c["torch_cublaslt"] = lt
    c["fi_tinygemm_direct"] = lambda: (fg.tinygemm_bf16(x, w, out), out)[1]
    for cfg in SP_CONFIGS:
        c["sp_bn%d_bk%d_s%d_w%d" % cfg] = single_pass(x, w, None, *cfg)
    if n <= 16384:  # split-K only matters where the grid is too small to fill 170 SMs
        for cfg in SPLITK_CONFIGS + (NARROW_SPLITK_CONFIGS if n <= 256 else []):
            c["splitk_bn%d_bk%d_sk%d_w%d_s%d" % cfg] = triton_splitk(x, w, *cfg)
    return c


def t3b_candidates(x, w, w8, s, m):
    from sgl_kernel import fp8_scaled_mm, sgl_per_token_quant_fp8

    n, k = w.shape
    c = {"cublas_bf16": lambda: F.linear(x, w)}
    if m <= 32:
        bn, bk, ns = T3_OPROJ_CFG[(n, k)]
        c["t3_bf16_tuned"] = single_pass(x, w, None, bn, bk, ns, 4)
        for cfg in SP_CONFIGS:
            c["fp8w_bn%d_bk%d_s%d_w%d" % cfg] = single_pass(x, w8, s, *cfg)
    w8_t = w8.t()
    s_row = s.view(1, n).contiguous()
    s_tensor = s.max().view(())
    w8_pt = (w.float() / s_tensor).to(torch.float8_e4m3fn).t()

    def a8_rowwise():
        sa = x.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / FP8_MAX
        x8 = (x.float() / sa).to(torch.float8_e4m3fn)
        return torch._scaled_mm(x8, w8_t, scale_a=sa, scale_b=s_row, out_dtype=torch.bfloat16)

    def a8_tensor():
        sa = (x.float().abs().amax().clamp(min=1e-12) / FP8_MAX)
        x8 = (x.float() / sa).to(torch.float8_e4m3fn)
        return torch._scaled_mm(x8, w8_pt, scale_a=sa, scale_b=s_tensor, out_dtype=torch.bfloat16)

    x8_buf = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    sa_buf = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    s_col = s.view(n, 1).contiguous()

    def sgl_w8a8():
        sgl_per_token_quant_fp8(x, x8_buf, sa_buf)
        return fp8_scaled_mm(x8_buf, w8_t, sa_buf, s_col, torch.bfloat16, None)

    c["w8a8_scaled_mm_rowwise"] = a8_rowwise
    c["w8a8_scaled_mm_tensor"] = a8_tensor
    c["w8a8_sgl_fp8_scaled_mm"] = sgl_w8a8
    wd = torch.empty_like(w)

    def dequant_cublas():
        torch.mul(w8, s[:, None], out=wd)
        return F.linear(x, wd)

    c["fp8w_dequant_cublas"] = dequant_cublas
    return c


def run_bench(bench, ms, bw_gbs, only):
    torch.manual_seed(0)
    dev = "cuda"
    rows = []
    shapes = T3C_SHAPES if bench == "t3c" else T3B_SHAPES
    for name, n, k in shapes:
        wbytes = n * k * 2
        n_copies = max(2, math.ceil(L2_ROTATE_BYTES / wbytes))
        ws = [torch.randn(n, k, dtype=torch.bfloat16, device=dev) * 0.02 for _ in range(n_copies)]
        q = [quantize_weight_per_channel(w) for w in ws] if bench == "t3b" else None
        for m in ms:
            if name == "lm_head" and m > 32:
                continue
            x = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
            ref32 = x.float() @ ws[0].float().t()
            ref_cublas = F.linear(x, ws[0])

            def cands(i):
                if bench == "t3c":
                    return t3c_candidates(x, ws[i], m)
                return t3b_candidates(x, ws[i], q[i][0], q[i][1], m)

            names = list(cands(0).keys())
            for cname in names:
                if only is not None and cname not in only:
                    continue
                fp8 = cname.startswith(("fp8", "w8a8"))
                act_bytes = (m * k + m * n) * 2
                sol_us = ((wbytes // 2 if fp8 else wbytes) + act_bytes) / (bw_gbs * 1e9) * 1e6
                rec = {"bench": bench, "shape": name, "N": n, "K": k, "M": m, "cand": cname,
                       "sol_us": sol_us, "sol_bf16_us": (wbytes + act_bytes) / (bw_gbs * 1e9) * 1e6}
                try:
                    per_copy = [cands(i)[cname] for i in range(n_copies)]
                    y = per_copy[0]()
                    torch.cuda.synchronize()
                    rel = (y.float() - ref32).norm() / ref32.norm()
                    rec["rel_l2_vs_fp32"] = rel.item()
                    rec["max_abs_vs_cublas"] = (y.float() - ref_cublas.float()).abs().max().item()
                    t0 = time.time()
                    rec["us"] = time_graph(lambda i: per_copy[i], n_copies)
                    rec["sol_fraction"] = sol_us / rec["us"]
                    rec["wall_s"] = time.time() - t0
                except Exception as e:  # noqa: BLE001 - recording unsupported candidates is the point
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
    p.add_argument("--bench", choices=("t3c", "t3b"), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bw-gbs", type=float, default=1650.0)
    p.add_argument("--ms", default="1,8,16,32,1024")
    p.add_argument("--only", default=None, help="comma-separated candidate names")
    a = p.parse_args()
    only = set(a.only.split(",")) if a.only else None
    rows = run_bench(a.bench, [int(v) for v in a.ms.split(",")], a.bw_gbs, only)
    with open(a.out, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(), "torch": torch.__version__,
                   "triton": triton.__version__, "bw_gbs": a.bw_gbs, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
