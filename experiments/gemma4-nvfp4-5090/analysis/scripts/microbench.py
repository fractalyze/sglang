"""Measured ceilings for the SOL model on this GPU: DRAM copy bandwidth and
dense BF16 / FP8 tensor throughput at large shapes (CUDA-event timed)."""

import json

import torch


def timeit(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters / 1e3


out = {}
n = 2 * 1024**3 // 2
a = torch.empty(n, dtype=torch.bfloat16, device="cuda")
b = torch.empty_like(a)
t = timeit(lambda: b.copy_(a))
out["copy_GBs"] = 2 * a.numel() * 2 / t / 1e9
x = torch.randn(n // 4, dtype=torch.bfloat16, device="cuda")
t = timeit(lambda: x.sum())
out["read_GBs"] = x.numel() * 2 / t / 1e9
del a, b, x
M = N = K = 8192
A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
t = timeit(lambda: A @ B)
out["bf16_tflops"] = 2 * M * N * K / t / 1e12
A8 = A.to(torch.float8_e4m3fn)
B8 = B.t().contiguous().to(torch.float8_e4m3fn).t()
one = torch.ones((), device="cuda")
try:
    t = timeit(lambda: torch._scaled_mm(A8, B8, one, one, out_dtype=torch.bfloat16))
    out["fp8_tflops"] = 2 * M * N * K / t / 1e12
except Exception as ex:
    out["fp8_error"] = repr(ex)[:200]
# Decode-shaped BF16 GEMV: lm_head 262144 x 2816 at M=1 and M=8.
W = torch.randn(262144, 2816, dtype=torch.bfloat16, device="cuda")
for m in (1, 8):
    X = torch.randn(m, 2816, dtype=torch.bfloat16, device="cuda")
    t = timeit(lambda: X @ W.t())
    out[f"lm_head_M{m}_us"] = t * 1e6
    out[f"lm_head_M{m}_GBs"] = W.numel() * 2 / t / 1e9
out["device"] = torch.cuda.get_device_name()
out["clocks_note"] = "unlocked clocks on shared host; see clocks.csv"
print(json.dumps(out, indent=1))
