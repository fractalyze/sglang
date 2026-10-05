"""C4 microbench: Triton decode attention time against the kv-split cap (`--triton-attention-num-kv-splits`).

The Triton backend picks each sequence's number of KV splits with `get_num_kv_splits_triton`
(more splits for longer sequences and smaller batches, up to the cap; the default cap is 8;
`--triton-attention-split-tile-size` only raises the cap to ceil(context / tile) outside
deterministic mode). This runs the backend's own heuristic and `decode_attention_fwd` on
Gemma-4-26B-A4B's two layer kinds at the served lengths, FP8 KV, for several batch sizes
and caps, and reports one decode step's attention time (5 full + 25 sliding layers).

  PYTHONPATH=<tree>/python python compute/decode_split_bench.py --out decode_split.json
"""

import argparse
import json
from typing import Dict, List

import torch
import triton

from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd
from sglang.kernels.ops.attention.metadata import get_num_kv_splits_triton

FP8 = torch.float8_e4m3fn
# The backend sizes splits with model_config.get_num_kv_heads(): the served config's num_key_value_heads = 2.
NUM_HEAD, BACKEND_KV_HEADS = 16, 2
LAYERS = {"full": (512, 2, 5), "sliding": (256, 8, 25)}  # kind -> (head_dim, kv heads, layers in the model)
WINDOW = 1024
BATCHES = (12, 20, 32)
CAPS = (4, 8, 16, 32, 64)


def seq_lens(batch: int, seed: int = 0) -> torch.Tensor:
    """Served-like context lengths: ~5.7K mean (prompt p50 5.6K plus the reply so far), 2K..10.5K."""
    g = torch.Generator().manual_seed(seed)
    lens = (torch.randn(batch, generator=g) * 1500 + 5700).clamp(2000, 10500).to(torch.int32)
    return lens.cuda()


def splits_for(lens: torch.Tensor, cap: int, sm_count: int) -> torch.Tensor:
    out = torch.empty_like(lens)
    get_num_kv_splits_triton[(1,)](out, lens, lens.numel(), 1, NUM_HEAD, BACKEND_KV_HEADS, cap, sm_count,
                                   MAX_NUM_SEQ=256)
    return out


def bench(kind: str, lens: torch.Tensor, cap: int, sm_count: int) -> Dict:
    head_dim, kvh, _ = LAYERS[kind]
    # One split count per sequence from the full lengths serves every layer (forward_decode passes
    # forward_metadata.num_kv_splits to the sliding layers too, with the window's kv indices).
    splits = splits_for(lens, cap, sm_count)
    if kind == "sliding":
        lens = lens.clamp(max=WINDOW)  # the backend passes window-trimmed kv indices
    batch = lens.numel()
    total = int(lens.sum())
    g = torch.Generator(device="cuda").manual_seed(1)
    k_buf = torch.randn(total + 64, kvh, head_dim, generator=g, device="cuda").to(FP8)
    v_buf = torch.randn(total + 64, kvh, head_dim, generator=g, device="cuda").to(FP8)
    q = torch.randn(batch, NUM_HEAD, head_dim, generator=g, device="cuda", dtype=torch.bfloat16) / head_dim ** 0.5
    o = torch.empty_like(q)
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
    kv_indptr[1:] = torch.cumsum(lens, 0)
    kv_indices = torch.randperm(total + 64, generator=g, device="cuda")[:total].to(torch.int64)
    attn_logits = torch.empty(batch, NUM_HEAD, cap, head_dim, dtype=torch.float32, device="cuda")
    attn_lse = torch.empty(batch, NUM_HEAD, cap, dtype=torch.float32, device="cuda")

    def run():
        decode_attention_fwd(q, k_buf, v_buf, o, kv_indptr, kv_indices, attn_logits, attn_lse, splits, cap,
                             1.0, 1.0, 1.0, enable_lean=False)

    run()
    out = o.float().clone()
    us = triton.testing.do_bench(run, warmup=20, rep=100) * 1e3
    return {"us": us, "splits_mean": splits.float().mean().item(), "splits_max": int(splits.max()), "out": out}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    rows: List[Dict] = []
    for batch in BATCHES:
        lens = seq_lens(batch)
        ref = {}
        for cap in CAPS:
            row = {"batch": batch, "cap": cap}
            step_us = 0.0
            for kind, (_, _, n_layers) in LAYERS.items():
                r = bench(kind, lens, cap, sm_count)
                if cap == 8:
                    ref[kind] = r["out"]
                row[kind] = {k: v for k, v in r.items() if k != "out"}
                step_us += n_layers * r["us"]
                row[kind]["out"] = r["out"]
            row["step_us"] = step_us
            rows.append(row)
        for row in (r for r in rows if r["batch"] == batch):
            for kind in LAYERS:
                row[kind]["max_abs_diff_vs_cap8"] = (row[kind].pop("out") - ref[kind]).abs().max().item()
            base = next(r for r in rows if r["batch"] == batch and r["cap"] == 8)["step_us"]
            row["step_vs_cap8"] = row["step_us"] / base
            print(f"B={batch:2d} cap={row['cap']:2d}: step {row['step_us']:7.1f} us ({row['step_vs_cap8']:.3f}x cap 8)"
                  f"  full {row['full']['us']:6.1f} us splits {row['full']['splits_mean']:.1f}"
                  f"  sliding {row['sliding']['us']:6.1f} us splits {row['sliding']['splits_mean']:.1f}"
                  f"  maxdiff {row['full']['max_abs_diff_vs_cap8']:.2e}/{row['sliding']['max_abs_diff_vs_cap8']:.2e}",
                  flush=True)
    with open(args.out, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(), "sm_count": sm_count, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
