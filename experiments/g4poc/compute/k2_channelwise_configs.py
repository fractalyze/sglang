"""K2-c1: tuned `triton_scaled_mm` configs for the served dense FP8 shapes, from the decode-M microbench.

SGLang's per-token x per-channel W8A8 FP8 linear (`apply_fp8_linear`) consults
`kernels/ops/quantization/configs/N=..,K=..,device_name=..,dtype=fp8_w8a8_channelwise.json` before CUTLASS and
runs `triton_scaled_mm` with the tile of the nearest tuned M (a null entry keeps CUTLASS). This writes one file
per Gemma-4-26B-A4B dense shape: every decode M the bench timed gets its fastest tile; M from 64 up is null, so
prefill chunks keep CUTLASS (not measured here).

  python compute/k2_channelwise_configs.py --bench <bench.json> --out <dir>
"""

import argparse
import json
import os
from typing import Dict

PREFILL_MS = (64, 128, 256, 512, 1024, 2048, 4096)


def configs_from_bench(bench: Dict) -> Dict[str, Dict[str, object]]:
    """File name -> {M: tile config or None}, one file per (N, K)."""
    device = bench["device"].replace(" ", "_")
    out: Dict[str, Dict[str, object]] = {}
    for row in bench["rows"]:
        name = f"N={row['n']},K={row['k']},device_name={device},dtype=fp8_w8a8_channelwise.json"
        out.setdefault(name, {})[str(row["m"])] = row["w8a8_triton_best_config"]
    for table in out.values():
        for m in PREFILL_MS:
            table[str(m)] = None
    return {name: dict(sorted(table.items(), key=lambda kv: int(kv[0]))) for name, table in out.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bench", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    with open(args.bench) as f:
        bench = json.load(f)
    os.makedirs(args.out, exist_ok=True)
    for name, table in configs_from_bench(bench).items():
        with open(os.path.join(args.out, name), "w") as f:
            json.dump(table, f, indent=4)
            f.write("\n")
        print(name, len(table))


if __name__ == "__main__":
    main()
