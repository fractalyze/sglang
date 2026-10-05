"""C2 microbench: Triton extend attention tiles on sm120 for Gemma-4's two layer kinds.

Times `extend_attention_fwd` on prefill shapes taken from the base's in-flight C12 window
(server log: 97% of prefill batches hold one request; median 347 new tokens over a 3,769-token
cached prefix; the top decile are 4,096-token chunks of uncached prompts), with an FP8 KV
cache, for the full layers (head_dim 512, 16 q / 2 kv heads) and the sliding layers
(head_dim 256, 16 q / 8 kv heads, window 1024: the backend passes only the in-window prefix).
Every tile config in the sweep is compared against the sm120 default (env knob off): time and
max |out - default|. Run on the GPU host with the candidate SGLang tree first on PYTHONPATH:

  python compute/extend_attn_bench.py --out bench.json [--quick]
"""

import argparse
import itertools
import json
from typing import Dict, List, Optional, Tuple

import torch
import triton

from sglang.kernels.ops.attention import extend_attention as ea
from sglang.srt.environ import envs

# (name, new tokens, cached prefix tokens, weight in the C12 prefill mix)
SHAPES = [("turn", 347, 3769, 0.85), ("cold-chunk", 4096, 0, 0.10), ("second-chunk", 4096, 4096, 0.05)]
LAYERS = {  # head_dim -> (q heads, kv heads, sliding window or -1, layers in the model)
    512: (16, 2, -1, 5),
    256: (16, 8, 1024, 25),
}
FP8 = torch.float8_e4m3fn


def make_inputs(head_dim: int, new: int, prefix: int, seed: int = 0) -> Dict:
    qh, kvh, window, _ = LAYERS[head_dim]
    g = torch.Generator(device="cuda").manual_seed(seed)
    if window > 0:
        prefix = min(prefix, window)  # the backend passes window-trimmed kv indices
    dev = "cuda"
    rnd = lambda *s: torch.randn(*s, generator=g, device=dev, dtype=torch.bfloat16)
    pool = prefix + 64
    perm = torch.randperm(pool, generator=g, device=dev)[:prefix].to(torch.int64)
    return {
        "q": rnd(new, qh, head_dim), "k": rnd(new, kvh, head_dim), "v": rnd(new, kvh, head_dim),
        "k_buf": (rnd(pool, kvh, head_dim) * 0.5).to(FP8), "v_buf": (rnd(pool, kvh, head_dim) * 0.5).to(FP8),
        "qo_indptr": torch.tensor([0, new], dtype=torch.int32, device=dev),
        "kv_indptr": torch.tensor([0, prefix], dtype=torch.int32, device=dev),
        "kv_indices": perm, "new": new, "window": window,
        "window_kv_offsets": torch.zeros(1, dtype=torch.int32, device=dev) if window > 0 else None,
    }


def run(inp: Dict) -> torch.Tensor:
    o = torch.empty_like(inp["q"])
    ea.extend_attention_fwd(
        inp["q"], inp["k"], inp["v"], o, inp["k_buf"], inp["v_buf"], inp["qo_indptr"], inp["kv_indptr"],
        inp["kv_indices"], None, True, None, inp["new"], 1.0, 1.0, sm_scale=1.0,
        sliding_window_size=inp["window"], window_kv_offsets=inp["window_kv_offsets"],
    )
    return o


def time_us(inp: Dict) -> float:
    return triton.testing.do_bench(lambda: run(inp), warmup=10, rep=50) * 1e3


def configs(head_dim: int, quick: bool) -> List[Tuple[int, int, int, int, int]]:
    if head_dim == 512:
        space = ([16, 32, 64], [16, 32, 64], [32, 64, 128], [4, 8], [1, 2, 3])
    else:
        space = ([32, 64, 128], [32, 64], [32, 64, 128], [4, 8], [1, 2, 3])
    if quick:
        space = tuple(s[:2] for s in space)
    return list(itertools.product(*space))


def sweep(head_dim: int, quick: bool) -> Dict:
    inputs = {name: make_inputs(head_dim, new, prefix) for name, new, prefix, _ in SHAPES}
    with envs.SGLANG_OPT_TRITON_EXTEND_SM120_FP8_KV_TILES.override(False):
        ref = {n: run(i).clone() for n, i in inputs.items()}
        base = {n: time_us(i) for n, i in inputs.items()}
    rows = []
    saved = ea._SM120_FP8_KV_EXTEND_TILES.get(head_dim)
    try:
        with envs.SGLANG_OPT_TRITON_EXTEND_SM120_FP8_KV_TILES.override(True):
            for cfg in configs(head_dim, quick):
                ea._SM120_FP8_KV_EXTEND_TILES[head_dim] = cfg
                row: Dict = {"cfg": cfg}
                try:
                    row["max_abs_diff"] = max(
                        (run(i) - ref[n]).abs().max().item() for n, i in inputs.items())
                    row["us"] = {n: time_us(i) for n, i in inputs.items()}
                except Exception as e:  # out of shared memory / registers for this tile
                    row["error"] = type(e).__name__
                rows.append(row)
    finally:
        ea._SM120_FP8_KV_EXTEND_TILES[head_dim] = saved
    weight = {name: w for name, _, _, w in SHAPES}
    mix = lambda us: sum(weight[n] * t for n, t in us.items())
    for r in rows:
        if "us" in r:
            r["mix_speedup"] = mix(base) / mix(r["us"])
    ok = sorted((r for r in rows if "us" in r), key=lambda r: -r["mix_speedup"])
    return {"head_dim": head_dim, "default_us": base, "default_mix_us": mix(base), "best": ok[:10],
            "n_configs": len(rows), "n_failed": sum("error" in r for r in rows)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    res = {"device": torch.cuda.get_device_name(), "shapes": SHAPES,
           "results": [sweep(hd, args.quick) for hd in (512, 256)]}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    for r in res["results"]:
        print(r["head_dim"], "default", {k: round(v) for k, v in r["default_us"].items()},
              "failed", r["n_failed"], "/", r["n_configs"])
        for b in r["best"][:5]:
            print("  ", b["cfg"], round(b["mix_speedup"], 3), {k: round(v) for k, v in b["us"].items()},
                  "maxdiff", round(b["max_abs_diff"], 4))


if __name__ == "__main__":
    main()
