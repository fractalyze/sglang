"""T4's byte-level KV test at speculative verify widths (gemma4nv W11, T-SPEC5).

Under MTP k=5 the target's TARGET_VERIFY forward runs T4's fused q/k/v norm +
RoPE + FP8 KV store over M = (1 + k) * B rows. The committed test
(`test_gemma4_fused_qkv_rope_kv.py`) covers M in {1, 8, 22, 32, 300}; this
driver calls the same test function at the verify widths for B in {1, 2, 8, 32}
without editing the tree under test.

Run on a gemma4nv host, from the deployed experiments dir, under the host-safety
path (host.lock + gpu.lock + the 24G scope):
  python -m trials.spec.verify_width_kv --tree /data/jooman/gemma4nv/trees/<sha12> --out FILE
"""

import argparse
import importlib.util
import itertools
import json
import os
import sys

K_DRAFTS = 5
BATCHES = (1, 2, 8, 32)
TEST = "test/registered/kernels/ops/layernorm/test_gemma4_fused_qkv_rope_kv.py"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(args.tree, "python"))
    spec = importlib.util.spec_from_file_location("kv_test", os.path.join(args.tree, TEST))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    widths = [(1 + K_DRAFTS) * b for b in BATCHES]
    results = []
    for shape, m, scale in itertools.product(mod._SHAPES, widths, (None, 1.0, 0.37)):
        case = {"shape": shape[0], "M": m, "scale": scale}
        try:
            mod.test_bit_exact_against_unfused_path(shape, m, scale)
            case["bit_exact"] = True
        except AssertionError as e:
            case.update(bit_exact=False, error=str(e)[:500])
        results.append(case)
        print(json.dumps(case), flush=True)
    out = {"tree": args.tree, "widths": widths, "cases": results,
           "all_bit_exact": all(c["bit_exact"] for c in results)}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"all_bit_exact={out['all_bit_exact']} ({len(results)} cases)")


if __name__ == "__main__":
    main()
