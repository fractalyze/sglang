"""K4: the Triton grouped decode attention at the served shapes against its KV-bytes floor (no new kernel).

Gemma-4-26B-A4B decode attention on the ship stack, per layer type, FP8 E4M3 KV, page size 1, scattered slots:
- sliding (25 layers): 16 q heads over 8 KV heads, head_dim 256, a 1,024-token window per request;
- full (5 layers): 16 q heads over 2 KV heads, head_dim 512, the request's whole context (3-9K, mean 6K).
`decode_attention_fwd_grouped` (stage 1 + the split reduce) is timed in a CUDA graph over N_COPIES layers' KV, so each
call streams its KV from DRAM, at batch 8 / 12 / 28. The served launch (BLOCK_N 32, 4 warps, 2 stages, 8 splits) is
compared with a grid of stage-1 tile constants and split counts, applied by wrapping the stage-1 kernel object (the
kernel algorithm is unchanged). The floor is the KV bytes (K and V, one byte per element) over the measured DRAM read
rate.

  PYTHONPATH=<tree>/python python compute/k4_decode_attn_bench.py --out <json>
"""

import argparse
import itertools
import json
import sys
from typing import Dict, List

import torch

Q_HEADS = 16
LAYERS = {"sliding": dict(kv_heads=8, dim=256, window=1024), "full": dict(kv_heads=2, dim=512, window=None)}
BATCHES = (8, 12, 28)
N_COPIES = 4
REPLAYS = 20
SERVED = dict(BLOCK_N=32, num_warps=4, num_stages=2, splits=8)
GRID = [dict(BLOCK_N=bn, num_warps=nw, num_stages=st, splits=sp)
        for bn, nw, st, sp in itertools.product((16, 32, 64), (2, 4, 8), (1, 2, 3), (8, 16, 32))]


class _Stage1Override:
    """Stands in for the stage-1 JIT kernel: forwards every launch with the tile constants under test."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.consts: Dict = {}

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            kwargs.update(self.consts)
            return launch(*args, **kwargs)

        return run


def graph_time_us(fn, n: int) -> float:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(n):
            fn(i)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(n):
            fn(i)
    g.replay()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(REPLAYS):
        g.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1e3 / (REPLAYS * n)


def read_bandwidth_gbps() -> float:
    x = torch.empty(2 * 1024 ** 3 // 2, dtype=torch.bfloat16, device="cuda").uniform_()
    us = graph_time_us(lambda i: x.sum(), 4)
    del x
    return 2 * 1024 ** 3 / us / 1e3


def main() -> None:
    from sglang.kernels.ops.attention import decode_attention as da

    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()

    override = _Stage1Override(da._fwd_grouped_kernel_stage1)
    da._fwd_grouped_kernel_stage1 = override
    bw = read_bandwidth_gbps()
    print(json.dumps({"read_gbps": bw}), flush=True)
    gen = torch.Generator().manual_seed(0)
    rows: List[Dict] = []
    for (name, lt), bs in itertools.product(LAYERS.items(), BATCHES):
        h, d = lt["kv_heads"], lt["dim"]
        lens = (torch.full((bs,), lt["window"]) if lt["window"]
                else torch.randint(3000, 9001, (bs,), generator=gen))
        total = int(lens.sum())
        pool = 2 * total
        kbuf = [torch.randn(pool, h, d, device="cuda").to(torch.float8_e4m3fn) for _ in range(N_COPIES)]
        vbuf = [torch.randn(pool, h, d, device="cuda").to(torch.float8_e4m3fn) for _ in range(N_COPIES)]
        indptr = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
        indptr[1:] = torch.cumsum(lens.cuda(), 0)
        indices = torch.randperm(pool, generator=gen)[:total].to(torch.int64).cuda()
        q = torch.randn(bs, Q_HEADS, d, device="cuda").to(torch.bfloat16)
        o = torch.empty(bs, Q_HEADS, d, device="cuda", dtype=torch.bfloat16)
        kv_bytes = total * h * d * 2
        floor = kv_bytes / bw / 1e3

        def timed(cfg):
            sp = cfg["splits"]
            logits = torch.empty(bs, Q_HEADS, sp, d, device="cuda", dtype=torch.float32)
            lse = torch.empty(bs, Q_HEADS, sp, device="cuda", dtype=torch.float32)
            nsplit = torch.full((bs,), sp, dtype=torch.int32, device="cuda")
            override.consts = {k: cfg[k] for k in ("BLOCK_N", "num_warps", "num_stages")}

            def call(i):
                da.decode_attention_fwd_grouped(q, kbuf[i], vbuf[i], o, indptr, indices, logits, lse, nsplit, sp,
                                                d ** -0.5, 1.0)

            return graph_time_us(call, N_COPIES)

        row = {"layer": name, "bs": bs, "kv_mb": kv_bytes / 1e6, "floor_us": floor, "served_us": timed(SERVED)}
        best = None
        for cfg in GRID:
            try:
                us = timed(cfg)
            except Exception:  # a tile past SM120 shared memory or registers
                continue
            if best is None or us < best[0]:
                best = (us, cfg)
        row["best_us"], row["best_config"] = best
        override.consts = {}
        rows.append(row)
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        del kbuf, vbuf
        torch.cuda.empty_cache()
    with open(args.out, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(), "read_gbps": bw, "rows": rows}, f, indent=1)
    print(f"result: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
