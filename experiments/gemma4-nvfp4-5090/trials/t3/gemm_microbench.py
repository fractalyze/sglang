"""T3 step 0: small-M BF16 GEMM candidates on the exact Gemma-4-26B-A4B shapes.

Times every candidate inside a CUDA graph (decode runs under graphs) and
rotates over enough weight copies to exceed the 5090's L2, so each call
streams its weight from DRAM as the 30-layer model does. sol_us is the
weight + activation bytes at the measured DRAM read bandwidth.

  python gemm_microbench.py --out results.json [--bw-gbs 1650]
"""

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# (name, N, K) of the BF16 linears excluded from NVFP4 (y = x @ W.T, W is N x K).
SHAPES = [
    ("o_proj_sliding", 2816, 4096),
    ("o_proj_full", 2816, 8192),
    ("dense_gate_up", 4224, 2816),
    ("dense_down", 2816, 2112),
    ("qkv_sliding", 8192, 2816),
]
MS = [1, 8, 16, 32, 1024]
L2_ROTATE_BYTES = 256 * 1024**2
GRAPH_CALLS = 64


@triton.jit
def _splitk_gemm_kernel(
    x_ptr, w_ptr, part_ptr, M, N, K, K_PER_SPLIT,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """part[s, m, n] = sum over this split's K range of x[m, k] * w[n, k] (fp32)."""
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k0 = pid_s * K_PER_SPLIT
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, K_PER_SPLIT, BLOCK_K):
        offs_k = k0 + kk + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :],
                    mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
    tl.store(part_ptr + (pid_s * M + offs_m[:, None]) * N + offs_n[None, :], acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _splitk_reduce_kernel(part_ptr, out_ptr, MN, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        acc += tl.load(part_ptr + s * MN + offs, mask=offs < MN, other=0.0)
    tl.store(out_ptr + offs, acc.to(tl.bfloat16), mask=offs < MN)


def triton_splitk(x, w, block_n, block_k, split_k, num_warps, num_stages):
    m, k = x.shape
    n = w.shape[0]
    block_m = max(16, triton.next_power_of_2(min(m, 64)))
    k_per = triton.cdiv(triton.cdiv(k, split_k), block_k) * block_k
    part = torch.empty((split_k, m, n), dtype=torch.float32, device=x.device)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)

    def run():
        grid = (triton.cdiv(n, block_n), split_k, triton.cdiv(m, block_m))
        _splitk_gemm_kernel[grid](x, w, part, m, n, k, k_per, BLOCK_M=block_m, BLOCK_N=block_n,
                                  BLOCK_K=block_k, num_warps=num_warps, num_stages=num_stages)
        _splitk_reduce_kernel[(triton.cdiv(m * n, 1024),)](part, out, m * n, SPLIT_K=split_k, BLOCK=1024)
        return out

    return run


TRITON_CONFIGS = [
    (bn, bk, sk, nw, ns)
    for bn in (16, 32, 64)
    for bk in (128, 256)
    for sk in (1, 2, 4)
    for nw in (4,)
    for ns in (3, 4)
]


def candidates(x, w):
    """name -> zero-arg callable computing x @ w.T in BF16 (None if unsupported)."""
    import flashinfer.gemm as fg

    m, n = x.shape[0], w.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    c = {"cublas_default": lambda: F.linear(x, w)}

    def lt():
        torch.backends.cuda.preferred_blas_library("cublaslt")
        try:
            return F.linear(x, w)
        finally:
            torch.backends.cuda.preferred_blas_library("cublas")

    c["torch_cublaslt"] = lt
    for be in ("cublaslt", "cudnn", "cutlass", "tinygemm", "tgv", "cute-dsl"):
        c[f"fi_{be}"] = (lambda be=be: fg.mm_bf16(x, w.t(), out=out, backend=be))
    c["fi_tinygemm_direct"] = lambda: (fg.tinygemm_bf16(x, w, out), out)[1]
    # T3b reference only (changes numerics): FP8 E4M3 weight with a per-tensor scale,
    # activation quantized per tensor inside the timed call.
    w8 = w.to(torch.float8_e4m3fn)
    one = torch.ones((), dtype=torch.float32, device=x.device)

    def fp8():
        x8 = x.to(torch.float8_e4m3fn)
        return torch._scaled_mm(x8, w8.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)

    c["fp8_scaled_mm_wa"] = fp8
    for cfg in TRITON_CONFIGS:
        c["triton_bn%d_bk%d_sk%d_w%d_s%d" % cfg] = triton_splitk(x, w, *cfg)
    return c


def time_graph(fn_for_copy, n_copies):
    """Mean us per call of a graph that cycles over n_copies distinct weights."""
    fns = [fn_for_copy(i) for i in range(n_copies)]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns:
            f()
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(GRAPH_CALLS):
            fns[i % n_copies]()
    g.replay()
    torch.cuda.synchronize()
    best = math.inf
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(7):
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1e3 / GRAPH_CALLS)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--bw-gbs", type=float, default=1650.0)
    p.add_argument("--ms", default=",".join(map(str, MS)))
    p.add_argument("--only", default=None, help="comma-separated candidate names")
    a = p.parse_args()
    only = set(a.only.split(",")) if a.only else None
    torch.manual_seed(0)
    dev = "cuda"
    rows = []
    for name, n, k in SHAPES:
        wbytes = n * k * 2
        n_copies = max(2, math.ceil(L2_ROTATE_BYTES / wbytes))
        ws = [torch.randn(n, k, dtype=torch.bfloat16, device=dev) * 0.02 for _ in range(n_copies)]
        for m in map(int, a.ms.split(",")):
            x = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
            ref32 = (x.float() @ ws[0].float().t())
            ref_cublas = F.linear(x, ws[0])
            sol_us = (wbytes + (m * k + m * n) * 2) / (a.bw_gbs * 1e9) * 1e6
            names = list(candidates(x, ws[0]).keys())
            for cname in names:
                if cname.startswith("triton") and m > 64:
                    continue
                if only is not None and cname not in only:
                    continue
                rec = {"shape": name, "N": n, "K": k, "M": m, "cand": cname, "sol_us": sol_us}
                try:
                    per_copy = [candidates(x, w)[cname] for w in ws]
                    y = per_copy[0]()
                    torch.cuda.synchronize()
                    rec["max_abs_vs_cublas"] = (y.float() - ref_cublas.float()).abs().max().item()
                    rec["max_abs_vs_fp32"] = (y.float() - ref32).abs().max().item()
                    rec["cublas_max_abs_vs_fp32"] = (ref_cublas.float() - ref32).abs().max().item()
                    t0 = time.time()
                    rec["us"] = time_graph(lambda i: per_copy[i], n_copies)
                    rec["sol_fraction"] = sol_us / rec["us"]
                    rec["wall_s"] = time.time() - t0
                except Exception as e:  # noqa: BLE001 - recording unsupported backends is the point
                    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                torch.cuda.synchronize()
                rows.append(rec)
                print(json.dumps(rec), flush=True)
            del x
        del ws
        torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(), "torch": torch.__version__,
                   "bw_gbs": a.bw_gbs, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
