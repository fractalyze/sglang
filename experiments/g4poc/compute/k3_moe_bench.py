"""K3: the served MoE layer at decode M against its weight-bandwidth floor (no server; existing kernels only).

One MoE layer of Gemma-4-26B-A4B as served (`fused_experts_impl`: moe_align, per-token FP8 quant, the gate_up fused_moe
kernel, gelu-and-mul, quant, the down kernel, the top-8 sum; FP8 W8A8 per-channel, E=128, hidden 2816, intermediate
704) is timed in a CUDA graph over L_COPIES layers of weights (so every layer streams its experts from DRAM), at each
decode M and a range of D, the distinct experts the step's routing touches. At decode M a layer reads D x 5.95 MB of
expert weights, so D sets the floor; real routing touches fewer experts than uniform top-8 draws (the served kernel
time at 8 running is below the uniform-routing floor), and D is read back by matching the served time to this curve.

Routing models per (M, D): `even` gives each of the D experts about the same load; `skew` draws the D experts with
Zipf(1) popularity, which loads a few experts past one BLOCK_SIZE_M block. Configs: the served C1 file
(SGLANG_MOE_CONFIG_DIR) and a grid through `override_config`.

  PYTHONPATH=<tree>/python SGLANG_MOE_CONFIG_DIR=<c1 dir> G4POC_MODEL_DIR=<model> python compute/k3_moe_bench.py --out <json>
"""

import argparse
import itertools
import json
import os
import sys
from typing import Dict, List, Optional

import torch

E, HIDDEN, INTER, TOP_K = 128, 2816, 704, 8
EXPERT_BYTES = (2 * INTER * HIDDEN + HIDDEN * INTER)  # FP8: one byte per weight
L_COPIES = 6
REPLAYS = 20
MS = (8, 12, 16, 28, 48)
DS = (16, 24, 32, 40, 48, 64, 80, 96, 128)
GRID = [dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=gm, num_warps=nw, num_stages=st)
        for bm, bn, bk, gm, nw, st in itertools.product((16, 32), (32, 64, 128), (128, 256), (1, 16), (4,), (3, 4))]


def routing(m: int, d: int, mode: str, gen: torch.Generator) -> torch.Tensor:
    """topk_ids [m, TOP_K] over a random set of d experts; every token gets TOP_K distinct experts."""
    experts = torch.randperm(E, generator=gen)[:d]
    if mode == "even":
        w = torch.ones(d)
    else:
        w = 1.0 / torch.arange(1, d + 1, dtype=torch.float)
    ids = torch.stack([experts[torch.multinomial(w, TOP_K, replacement=False, generator=gen)] for _ in range(m)])
    return ids.to(torch.int32)


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
    """DRAM read rate of a plain reduction over 2 GB (the floor's denominator)."""
    x = torch.empty(2 * 1024 ** 3 // 2, dtype=torch.bfloat16, device="cuda").uniform_()
    us = graph_time_us(lambda i: x.sum(), 4)
    del x
    return 2 * 1024 ** 3 / us / 1e3


def main() -> None:
    from sglang.srt.layers.moe.moe_runner.triton_utils import override_config
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts_impl
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    # The MoE config lookup reads the runtime context (as SGLang's own MoE benchmarks set it up).
    set_global_server_args_for_scheduler(ServerArgs(model_path=os.environ["G4POC_MODEL_DIR"]))
    from sglang.srt.distributed.parallel_state import init_distributed_environment, initialize_model_parallel

    # The kernel sequence reads the TP group (a single rank here), as in SGLang's MoE benchmark.
    init_distributed_environment(world_size=1, rank=0, distributed_init_method="tcp://127.0.0.1:23457",
                                 local_rank=0, backend="nccl")
    initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)

    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--ms", default=",".join(map(str, MS)))
    p.add_argument("--ds", default=",".join(map(str, DS)))
    p.add_argument("--grid-ms", default="8,12,28", help="Ms that also sweep the config grid")
    args = p.parse_args()
    ms = [int(x) for x in args.ms.split(",")]
    ds = [int(x) for x in args.ds.split(",")]
    grid_ms = {int(x) for x in args.grid_ms.split(",")}

    bw = read_bandwidth_gbps()
    print(json.dumps({"read_gbps": bw}), flush=True)
    torch.manual_seed(0)
    w1 = [(torch.randn(E, 2 * INTER, HIDDEN, device="cuda") * 0.02).to(torch.float8_e4m3fn) for _ in range(L_COPIES)]
    w2 = [(torch.randn(E, HIDDEN, INTER, device="cuda") * 0.02).to(torch.float8_e4m3fn) for _ in range(L_COPIES)]
    s1 = [torch.full((E, 2 * INTER, 1), 1e-2, device="cuda") for _ in range(L_COPIES)]
    s2 = [torch.full((E, HIDDEN, 1), 1e-2, device="cuda") for _ in range(L_COPIES)]
    gen = torch.Generator().manual_seed(1)
    rows: List[Dict] = []
    for m in ms:
        x = torch.randn(m, HIDDEN, device="cuda").to(torch.bfloat16)
        for d, mode in itertools.product([d for d in ds if TOP_K <= d <= min(E, TOP_K * m)], ("even", "skew")):
            ids = [routing(m, d, mode, gen).cuda() for _ in range(L_COPIES)]
            wts = [torch.full((m, TOP_K), 1.0 / TOP_K, device="cuda") for _ in range(L_COPIES)]
            distinct = sum(len(torch.unique(t)) for t in ids) / L_COPIES
            loads = [torch.bincount(t.flatten().long(), minlength=E) for t in ids]
            max_load = sum(int(l.max()) for l in loads) / L_COPIES

            def layer(i, ids=ids, wts=wts):
                fused_experts_impl(x, w1[i], w2[i], wts[i], ids[i], activation="gelu", use_fp8_w8a8=True,
                                   per_channel_quant=True, w1_scale=s1[i], w2_scale=s2[i])

            row = {"m": m, "d_target": d, "mode": mode, "distinct": distinct, "max_load": max_load,
                   "floor_us": distinct * EXPERT_BYTES / bw / 1e3, "c1_us": graph_time_us(layer, L_COPIES)}
            if m in grid_ms:
                best: Optional[tuple] = None
                for cfg in GRID:
                    try:
                        with override_config(cfg):
                            us = graph_time_us(layer, L_COPIES)
                    except Exception:  # a tile that does not fit shared memory
                        continue
                    if best is None or us < best[0]:
                        best = (us, cfg)
                row["best_us"], row["best_config"] = best
            rows.append(row)
            print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    with open(args.out, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(), "read_gbps": bw, "expert_bytes": EXPERT_BYTES,
                   "rows": rows}, f, indent=1)
    print(f"result: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
