"""C2-A numerics: error of the sm120 extend tiles against an fp32 reference, next to the default tiles'.

The Triton extend kernel's FP8-prefix path casts q to FP8 and quantizes the softmax weights to
FP8 against the running row max of each KV tile, so the tile width decides which values are
rounded and how. A tile change therefore moves outputs at the reorder level; this measures
whether it also moves them further from exact attention. Every config runs the same inputs
(FP8 prefix K/V, bf16 extend q/k/v, the backend's window-trimmed prefix on sliding layers) and
is compared with fp32 attention under the kernel's own masks (prefix: q_pos <= kv_pos + window;
extend: causal and the same window).

  PYTHONPATH=<tree>/python python compute/extend_tiles_accuracy.py --out acc.json
"""

import argparse
import json
from typing import Dict, Optional, Tuple

import torch

from sglang.kernels.ops.attention import extend_attention as ea
from sglang.srt.environ import envs

FP8 = torch.float8_e4m3fn
LAYERS = {512: (16, 2, -1), 256: (16, 8, 1024)}  # head_dim -> (q heads, kv heads, sliding window)
# (name, new tokens, cached prefix): a teacher-forced reply over a long prompt, the C12 median turn,
# a mean-sized turn, and the two 4,096-token chunk kinds.
SHAPES = [("forced-reply", 184, 4251), ("turn", 347, 3769), ("mean-turn", 1700, 4000),
          ("cold-chunk", 4096, 0), ("second-chunk", 4096, 4096)]
# Score scale: q is drawn so that q.k has about this standard deviation.
SCORE_STD = (1.0, 4.0)
VARIANTS = {
    "default": None,
    "new": "table",
    "exact": {512: (16, 32, 32, 4, 1), 256: (32, 64, 64, 8, 1)},
}


def make_inputs(head_dim: int, new: int, prefix: int, score_std: float, seed: int = 0) -> Dict:
    qh, kvh, window = LAYERS[head_dim]
    if window > 0:
        prefix = min(prefix, window)  # the backend passes window-trimmed kv indices
    g = torch.Generator(device="cuda").manual_seed(seed)
    rnd = lambda *s: torch.randn(*s, generator=g, device="cuda", dtype=torch.float32)
    pool = prefix + 64
    # k ~ N(0, 1) per element, so q.k has std |q|; scale q to the requested score std.
    q = rnd(new, qh, head_dim) * (score_std / head_dim ** 0.5)
    return {
        "q": q.bfloat16(), "k": rnd(new, kvh, head_dim).bfloat16(), "v": rnd(new, kvh, head_dim).bfloat16(),
        "k_buf": rnd(pool, kvh, head_dim).to(FP8), "v_buf": rnd(pool, kvh, head_dim).to(FP8),
        "kv_indices": torch.randperm(pool, generator=g, device="cuda")[:prefix].to(torch.int64),
        "qo_indptr": torch.tensor([0, new], dtype=torch.int32, device="cuda"),
        "kv_indptr": torch.tensor([0, prefix], dtype=torch.int32, device="cuda"),
        "new": new, "prefix": prefix, "window": window,
        "window_kv_offsets": torch.zeros(1, dtype=torch.int32, device="cuda") if window > 0 else None,
    }


def run_kernel(inp: Dict) -> torch.Tensor:
    o = torch.empty_like(inp["q"])
    ea.extend_attention_fwd(
        inp["q"], inp["k"], inp["v"], o, inp["k_buf"], inp["v_buf"], inp["qo_indptr"], inp["kv_indptr"],
        inp["kv_indices"], None, True, None, inp["new"], 1.0, 1.0, sm_scale=1.0,
        sliding_window_size=inp["window"], window_kv_offsets=inp["window_kv_offsets"],
    )
    return o.float()


def reference(inp: Dict) -> torch.Tensor:
    """fp32 attention with the kernel's masks; K/V from the same FP8 prefix and bf16 extend tensors."""
    new, prefix, window = inp["new"], inp["prefix"], inp["window"]
    q = inp["q"].float()
    k = torch.cat([inp["k_buf"][inp["kv_indices"]].float(), inp["k"].float()])
    v = torch.cat([inp["v_buf"][inp["kv_indices"]].float(), inp["v"].float()])
    m = torch.arange(new, device="cuda")[:, None]
    n = torch.arange(prefix + new, device="cuda")[None, :]
    is_prefix = n < prefix
    q_pos = prefix + m
    allowed = torch.where(is_prefix, torch.ones_like(n, dtype=torch.bool), (n - prefix) <= m)
    if window > 0:
        allowed &= torch.where(is_prefix, q_pos <= n + window, m <= (n - prefix) + window)
    qh, kvh = q.shape[1], k.shape[1]
    out = torch.empty_like(q)
    for h in range(qh):
        s = q[:, h] @ k[:, h // (qh // kvh)].T
        s = s.masked_fill(~allowed, float("-inf"))
        out[:, h] = torch.softmax(s, dim=-1) @ v[:, h // (qh // kvh)]
    return out


def errors(out: torch.Tensor, ref: torch.Tensor) -> Dict[str, float]:
    d = out - ref
    return {"rel_l2": (d.norm() / ref.norm()).item(), "max_abs": d.abs().max().item(),
            "p999_abs": d.abs().flatten().float().quantile(0.999).item() if d.numel() < 2 ** 24
            else d.abs().flatten()[:2 ** 24].float().quantile(0.999).item()}


def with_variant(variant, fn):
    if variant is None:
        with envs.SGLANG_OPT_TRITON_EXTEND_SM120_FP8_KV_TILES.override(False):
            return fn()
    saved = dict(ea._SM120_FP8_KV_EXTEND_TILES)
    try:
        if isinstance(variant, dict):
            ea._SM120_FP8_KV_EXTEND_TILES.update(variant)
        with envs.SGLANG_OPT_TRITON_EXTEND_SM120_FP8_KV_TILES.override(True):
            return fn()
    finally:
        ea._SM120_FP8_KV_EXTEND_TILES.clear()
        ea._SM120_FP8_KV_EXTEND_TILES.update(saved)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rows = []
    for head_dim in LAYERS:
        for name, new, prefix in SHAPES:
            for std in SCORE_STD:
                inp = make_inputs(head_dim, new, prefix, std)
                ref = reference(inp)
                row = {"head_dim": head_dim, "shape": name, "new": new, "prefix": inp["prefix"], "score_std": std}
                for label, variant in VARIANTS.items():
                    row[label] = errors(with_variant(variant, lambda: run_kernel(inp)), ref)
                row["new_over_default_rel_l2"] = row["new"]["rel_l2"] / row["default"]["rel_l2"]
                row["exact_over_default_rel_l2"] = row["exact"]["rel_l2"] / row["default"]["rel_l2"]
                rows.append(row)
                print(f"hd{head_dim} {name:12s} std {std}: rel_l2 default {row['default']['rel_l2']:.2e} "
                      f"new {row['new']['rel_l2']:.2e} ({row['new_over_default_rel_l2']:.2f}x) "
                      f"exact {row['exact']['rel_l2']:.2e}  max_abs default {row['default']['max_abs']:.3f} "
                      f"new {row['new']['max_abs']:.3f}", flush=True)
                del inp, ref
                torch.cuda.empty_cache()
    with open(args.out, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(), "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
