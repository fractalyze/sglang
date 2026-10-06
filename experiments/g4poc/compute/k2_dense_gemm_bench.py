"""K2: decode-M dense FP8 GEMM microbench on the served shapes (no server; existing kernels only).

At 8 in flight the decode step spends ~38% of its time in the CUTLASS FP8 W8A8 dense GEMMs (qkv, o,
gate_up, down; grids of 22-64 CTAs on 170 SMs). This times, per shape and decode M, inside a CUDA graph
of 30 calls that rotate over 30 weight copies (so every call streams its weight from DRAM, as one
decode step does):

- `w8a8`: the served path, `apply_fp8_linear` (per-token FP8 activation quant + CUTLASS scaled mm);
- `w8a8_triton`: the same W8A8 math on `triton_scaled_mm` over a tile grid (the tree's tuned-config route,
  `SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE`, which has no RTX 5090 config for these shapes), activation quant included;
- `wonly`: the tree's small-M Triton GEMM reading the same E4M3 weight with its per-channel scale
  (weight-only: bf16 activations, no activation quant), over a few tile configs;
- `bf16`: cuBLAS on a bf16 copy, for reference.

Run on a g4poc host with the ref's SGLang tree first on PYTHONPATH:
  PYTHONPATH=<tree>/python python compute/k2_dense_gemm_bench.py --out <json>
"""

import argparse
import json
import sys
from typing import Callable, Dict, List

import torch

# Gemma-4-26B-A4B dense linears (N, K): qkv sliding / full (k_eq_v loads K into the V shard), o sliding /
# full, dense MLP gate_up and down.
SHAPES = {
    "qkv_sliding": (8192, 2816),
    "qkv_full": (10240, 2816),
    "o_sliding": (2816, 4096),
    "o_full": (2816, 8192),
    "gate_up": (4224, 2816),
    "down": (2816, 2112),
}
# Per decode step: 25 sliding and 5 full layers.
CALLS_PER_STEP = {"qkv_sliding": 25, "qkv_full": 5, "o_sliding": 25, "o_full": 5, "gate_up": 30, "down": 30}
MS = (1, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48)
WONLY_TILES = ((32, 256, 4), (32, 128, 4), (64, 128, 3), (16, 256, 4), (64, 256, 3))
# triton_scaled_mm (W8A8, the tree's tuned-config route) tiles: (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages).
W8A8_TRITON_TILES = tuple((16, bn, bk, 4, st) for bn in (16, 32, 64) for bk in (128, 256, 512) for st in (2, 4))
N_COPIES = 30
REPLAYS = 20


def graph_time_us(fn: Callable[[int], None], n_calls: int) -> float:
    """Mean per-call time of fn(0..n_calls-1) captured in one CUDA graph."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(n_calls):  # warm-up (Triton JIT, CUTLASS init) outside the capture
            fn(i)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(n_calls):
            fn(i)
    g.replay()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(REPLAYS):
        g.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1e3 / (REPLAYS * n_calls)


def bench_shape(name: str, n: int, k: int, ms: List[int]) -> List[Dict]:
    from sglang.kernels.ops.gemm import triton_small_m_bf16_gemm as sm
    from sglang.kernels.ops.quantization.fp8_kernel import triton_scaled_mm
    from sglang.srt.layers.quantization import fp8_utils
    from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

    torch.manual_seed(0)
    w_bf16 = [(torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16) for _ in range(N_COPIES)]
    q = [sm.quantize_fp8_weight_per_channel(w) for w in w_bf16]
    w_fp8 = [wq for wq, _ in q]  # [N, K] row-major E4M3
    scale = [s for _, s in q]  # [N] fp32
    w_served = [w.t() for w in w_fp8]  # the served layout: a [K, N] view of the [N, K] weight
    scale_served = [s.view(-1, 1) for s in scale]
    rows = []
    for m in ms:
        x = (torch.randn(m, k, device="cuda")).to(torch.bfloat16)
        ref = (x.float() @ (w_fp8[0].float() * scale[0][:, None]).t())
        row = {"shape": name, "n": n, "k": k, "m": m, "weight_mb": n * k / 1e6}

        def w8a8(i):
            apply_fp8_linear(input=x, weight=w_served[i], weight_scale=scale_served[i], input_scale=None,
                             bias=None, use_per_token_if_dynamic=True, compressed_tensor_quant=True)

        row["w8a8_us"] = graph_time_us(w8a8, N_COPIES)
        out = apply_fp8_linear(input=x, weight=w_served[0], weight_scale=scale_served[0], input_scale=None,
                               bias=None, use_per_token_if_dynamic=True, compressed_tensor_quant=True)
        row["w8a8_rel_err"] = ((out.float() - ref).norm() / ref.norm()).item()
        best = None
        for bn, bk, st in WONLY_TILES:
            cfg = sm._TileConfig(bn, bk, st, max_m=max(ms))

            def wonly(i, cfg=cfg):
                sm._launch(x, w_fp8[i], scale[i], cfg)

            try:
                us = graph_time_us(wonly, N_COPIES)
            except Exception as e:  # a tile that does not fit shared memory at this M
                row[f"wonly_{bn}x{bk}x{st}_error"] = type(e).__name__
                continue
            row[f"wonly_{bn}x{bk}x{st}_us"] = us
            if best is None or us < best[0]:
                best = (us, f"{bn}x{bk}x{st}")
        out = sm._launch(x, w_fp8[0], scale[0], sm._TileConfig(32, 256, 4, max_m=max(ms)))
        row["wonly_rel_err"] = ((out.float() - ref).norm() / ref.norm()).item()
        row["wonly_best_us"], row["wonly_best_tile"] = best

        best = None
        for bm, bn, bk, nw, st in W8A8_TRITON_TILES:
            def w8a8_tri(i, bm=bm, bn=bn, bk=bk, nw=nw, st=st):
                qx, xs = fp8_utils.sglang_per_token_quant_fp8(x)
                triton_scaled_mm(qx, w_served[i], xs, scale_served[i], torch.bfloat16, None, block_size_m=bm,
                                 block_size_n=bn, block_size_k=bk, use_heuristic=False, num_warps=nw,
                                 num_stages=st)

            try:
                us = graph_time_us(w8a8_tri, N_COPIES)
            except Exception as e:
                row[f"w8a8_triton_{bm}x{bn}x{bk}x{nw}x{st}_error"] = type(e).__name__
                continue
            if best is None or us < best[0]:
                best = (us, {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "num_warps": nw,
                             "num_stages": st})
        row["w8a8_triton_best_us"], row["w8a8_triton_best_config"] = best

        def bf16(i):
            torch.nn.functional.linear(x, w_bf16[i])

        row["bf16_us"] = graph_time_us(bf16, N_COPIES)
        row["wonly_gbps"] = n * k / row["wonly_best_us"] / 1e3
        row["w8a8_gbps"] = n * k / row["w8a8_us"] / 1e3
        rows.append(row)
        print(json.dumps({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in row.items()
                          if kk in ("shape", "m", "w8a8_us", "w8a8_triton_best_us", "wonly_best_us", "wonly_best_tile",
                                    "bf16_us", "w8a8_gbps", "wonly_gbps")}), flush=True)
    del w_bf16, q, w_fp8, scale, w_served, scale_served
    torch.cuda.empty_cache()
    return rows


def per_step(rows: List[Dict]) -> Dict:
    """Per decode step at each M: summed time of the 120 dense GEMMs, served vs weight-only."""
    out = {}
    for m in sorted({r["m"] for r in rows}):
        rs = {r["shape"]: r for r in rows if r["m"] == m}
        if set(rs) != set(SHAPES):
            continue
        out[m] = {k: sum(CALLS_PER_STEP[s] * rs[s][k] for s in SHAPES)
                  for k in ("w8a8_us", "w8a8_triton_best_us", "wonly_best_us", "bf16_us")}
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--ms", default=",".join(map(str, MS)))
    p.add_argument("--shapes", default=",".join(SHAPES))
    args = p.parse_args()
    ms = [int(x) for x in args.ms.split(",")]
    rows = []
    for name in args.shapes.split(","):
        n, k = SHAPES[name]
        rows += bench_shape(name, n, k, ms)
    res = {"device": torch.cuda.get_device_name(), "rows": rows, "per_step_us": per_step(rows)}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["per_step_us"], indent=1))
    print(f"result: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
