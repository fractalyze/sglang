"""`gate peaks`: measured DRAM bandwidth, tensor-core peaks and the per-launch floor on this GPU.

Runs in the venv python (needs torch); the SOL tables quote both the datasheet
bandwidth and the measured one, and a bound whose implied bandwidth exceeds the
measured peak is flagged.
"""

import json
import sys

import torch


def _time_ms(fn, iters: int = 20) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def dram_bandwidth() -> dict:
    n = 2 * 1024**3 // 2
    a = torch.empty(n, dtype=torch.bfloat16, device="cuda").normal_()
    b = torch.empty_like(a)
    copy_ms = _time_ms(lambda: b.copy_(a))
    read_ms = _time_ms(lambda: a.sum(dtype=torch.float32))
    nbytes = a.numel() * a.element_size()
    return {"copy_gb_s": 2 * nbytes / copy_ms / 1e6, "read_gb_s": nbytes / read_ms / 1e6, "bytes": nbytes}


def gemm_peaks(m: int = 8192) -> dict:
    out = {}
    a = torch.randn(m, m, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(m, m, dtype=torch.bfloat16, device="cuda")
    out["bf16_tflops"] = 2 * m**3 / _time_ms(lambda: a @ b) / 1e9
    try:
        a8, b8 = a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn).t()
        one = torch.ones((), device="cuda")
        out["fp8_tflops"] = 2 * m**3 / _time_ms(lambda: torch._scaled_mm(a8, b8, one, one, out_dtype=torch.bfloat16)) / 1e9
    except Exception as e:  # recorded, not fatal: the SOL falls back to the datasheet value
        out["fp8_error"] = repr(e)[:300]
    try:
        from sgl_kernel import cutlass_scaled_fp4_mm, scaled_fp4_quant

        gs = torch.tensor(1.0, device="cuda")
        a4, sa = scaled_fp4_quant(a, gs)
        b4, sb = scaled_fp4_quant(b, gs)
        alpha = torch.tensor(1.0, device="cuda")
        out["nvfp4_tflops"] = 2 * m**3 / _time_ms(
            lambda: cutlass_scaled_fp4_mm(a4, b4, sa, sb, alpha, torch.bfloat16)) / 1e9
    except Exception as e:  # recorded, not fatal: the SOL falls back to the datasheet value
        out["nvfp4_error"] = repr(e)[:300]
    return out


def launch_floor_us() -> dict:
    x = torch.zeros(1, device="cuda")
    eager_ms = _time_ms(lambda: x.add_(1), iters=2000)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        x.add_(1)
        with torch.cuda.graph(g):
            for _ in range(1000):
                x.add_(1)
    graph_ms = _time_ms(g.replay, iters=10) / 1000
    return {"eager_launch_us": 1e3 * eager_ms, "graph_node_us": 1e3 * graph_ms}


def main() -> None:
    props = torch.cuda.get_device_properties(0)
    res = {"device": props.name, "sm_count": props.multi_processor_count,
           "dram": dram_bandwidth(), "gemm": gemm_peaks(), "launch": launch_floor_us()}
    json.dump(res, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
