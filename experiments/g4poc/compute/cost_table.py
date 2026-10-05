"""Cost table for the report: per stack and E2E p90 SLO, the most in-flight requests per GPU that meet the
SLO, and the cheapest point that meets it (most output goodput), with $/1M output and total tokens at
the study's GPU prices. Reads `gate sweep` outputs.

  python compute/cost_table.py base=<sweep.json> final=<sweep.json> ... [--slo 6,10,15]
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gate import config, metrics  # noqa: E402


def _concurrency(p: Dict) -> int:
    load = p["load"]
    return load["concurrency"] if isinstance(load, dict) else int(str(load).rsplit("C", 1)[1])


def rows_for(sweep: Dict, slos: Sequence[float]) -> List[Dict]:
    out = []
    for slo in slos:
        ok = [p for p in sweep["points"] if metrics.meets_slo(p["summary"], slo)]
        if not ok:
            out.append({"slo_s": slo, "max_inflight": None, "cheapest": None})
            continue
        most = max(ok, key=_concurrency)
        cheap = max(ok, key=lambda p: p["summary"]["output_tok_s_per_gpu"])
        s = cheap["summary"]
        out.append({
            "slo_s": slo, "max_inflight": _concurrency(most), "max_inflight_p90_s": most["summary"]["e2e_p90_s"],
            "cheapest": {"inflight": _concurrency(cheap), "out_tok_s": s["output_tok_s_per_gpu"],
                         "total_tok_s": s["total_tok_s_per_gpu"], "e2e_p90_s": s["e2e_p90_s"],
                         "hit": s["prefix_cache_hit_rate"],
                         "usd_per_mtok_output": metrics.cost_table(s["output_tok_s_per_gpu"]),
                         "usd_per_mtok_total": metrics.cost_table(s["total_tok_s_per_gpu"])}})
    return out


def markdown(tables: Dict[str, List[Dict]]) -> str:
    prices = " / ".join(f"{p:.2f}" for p in config.GPU_PRICES_USD_PER_HR)
    lines = [f"| stack | E2E p90 SLO | max in flight | cheapest point | out / total tok/s | p90 | hit | "
             f"$/1M output ({prices}) | $/1M total |", "|" + "---|" * 9]
    for name, rows in tables.items():
        for r in rows:
            c = r["cheapest"]
            if c is None:
                lines.append(f"| {name} | {r['slo_s']:g} s | none | | | | | | |")
                continue
            fmt = lambda d, n: " / ".join(f"{v:.{n}f}" for v in d.values())
            lines.append(f"| {name} | {r['slo_s']:g} s | {r['max_inflight']} | {c['inflight']} | "
                         f"{c['out_tok_s']:,.0f} / {c['total_tok_s']:,.0f} | {c['e2e_p90_s']:.2f} s | {c['hit']:.2f} | "
                         f"{fmt(c['usd_per_mtok_output'], 3)} | {fmt(c['usd_per_mtok_total'], 4)} |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("sweeps", nargs="+", metavar="NAME=SWEEP_JSON")
    p.add_argument("--slo", default=",".join(f"{s:g}" for s in config.SLOS_E2E_P90_S))
    p.add_argument("--out")
    args = p.parse_args()
    slos = [float(x) for x in args.slo.split(",")]
    tables = {}
    for spec in args.sweeps:
        name, path = spec.split("=", 1)
        with open(path) as f:
            tables[name] = rows_for(json.load(f), slos)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(tables, f, indent=1)
    print(markdown(tables))


if __name__ == "__main__":
    main()
