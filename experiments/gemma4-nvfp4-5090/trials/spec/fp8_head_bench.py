"""T-SPEC4 microbench: the MTP assistant's 262144 x 1024 head, BF16 cuBLAS vs FP8 small-M tiles.

Run inside a tree that has SGLANG_OPT_MTP_FP8_LM_HEAD (commit 4ebe6175af or later):
  python fp8_head_bench.py --out <json>
"""

import argparse
import json

import torch
import torch.nn.functional as F
import triton

from sglang.kernels.ops.gemm import triton_small_m_bf16_gemm as k

SHAPE = (262144, 1024)
TILES = [(32, 128, 3), (32, 256, 4), (64, 128, 4), (64, 256, 4), (128, 128, 3), (128, 256, 3), (256, 128, 3)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(0)
    w = torch.randn(*SHAPE, dtype=torch.bfloat16, device="cuda") * 0.02
    w8, scale = k.quantize_fp8_weight_per_channel(w)
    rows = []
    for m in (1, 8, 32):
        x = torch.randn(m, SHAPE[1], dtype=torch.bfloat16, device="cuda")
        bf16_ms = triton.testing.do_bench(lambda: F.linear(x, w), warmup=50, rep=300)
        row = {"m": m, "bf16_cublas_us": 1e3 * bf16_ms, "bf16_gbps": w.numel() * 2 / bf16_ms / 1e6, "fp8": {}}
        for tile in TILES:
            k._FP8_HEAD_TUNED_SHAPES[SHAPE] = k._TileConfig(*tile)
            ms = triton.testing.do_bench(lambda: k.triton_small_m_fp8_vocab_head(x, w8, scale), warmup=50, rep=300)
            row["fp8"]["x".join(map(str, tile))] = {"us": 1e3 * ms, "gbps": w8.numel() / ms / 1e6}
        best = min(row["fp8"].items(), key=lambda kv: kv[1]["us"])
        row["best_fp8"] = best[0]
        print(json.dumps({"m": m, "bf16_us": round(row["bf16_cublas_us"], 1), "best": best[0],
                          "best_us": round(best[1]["us"], 1)}))
        rows.append(row)
    with open(args.out, "w") as f:
        json.dump({"shape": SHAPE, "gpu": torch.cuda.get_device_name(), "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
