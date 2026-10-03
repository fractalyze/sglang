"""Verify-cost curve for T-SPEC1 from spec_probe outputs (offline, no GPU).

Inputs: a `sweep` probe.json (decode step ms with exactly M running rows) and
an `experts` run dir (routing_{timing,hidden}.npz, [tokens, layers, top_k]).

Distinct experts per layer:
  verify(B, k) -- union over B streams of the 1+k consecutive tokens j..j+k,
                  which is what one verify forward routes (accepted rows are
                  the greedy tokens; rejected rows are approximated by them);
  decode(M)    -- union over M different streams at one step, which is what
                  the sweep's decode step at M rows routes.

Cost model fitted on the sweep: step(M) = a + b*M + c*E_decode(M). The verify
estimate is a + b*B*(1+k) + c*E_verify(B, k). The b*rows term includes the
per-row attention read, which a verify does once per stream instead of once
per row, so the estimate is an upper bound by that share.

  python verify_cost.py --sweep SWEEP/probe.json --experts EXPERTS_DIR [--out table.json]
"""

import argparse
import json
from typing import List

import numpy as np

N_EXPERTS = 128


def _load(path: str) -> List[np.ndarray]:
    z = np.load(path)
    return [z[k] for k in sorted(z.files, key=lambda s: int(s.split("_")[1]))]


def _distinct(token_sets: List[np.ndarray]) -> float:
    """token_sets: arrays [n_tokens, layers, top_k]; mean over layers of the union size."""
    stacked = np.concatenate(token_sets, axis=0)  # [rows, layers, top_k]
    layers = stacked.shape[1]
    sizes = [np.unique(stacked[:, l, :]).size for l in range(layers)]
    return float(np.mean(sizes))


def e_verify(streams: List[np.ndarray], batch: int, k: int, rng: np.random.Generator, samples: int = 200) -> float:
    vals = []
    min_len = min(s.shape[0] for s in streams)
    for _ in range(samples):
        picks = rng.choice(len(streams), size=batch, replace=False)
        j = int(rng.integers(0, min_len - k))
        vals.append(_distinct([streams[p][j : j + k + 1] for p in picks]))
    return float(np.mean(vals))


def e_decode(streams: List[np.ndarray], rows: int, rng: np.random.Generator, samples: int = 200) -> float:
    # More rows than streams: other steps of a stream, 32+ tokens apart, stand in for more streams.
    min_len = min(s.shape[0] for s in streams)
    vals = []
    for _ in range(samples):
        sets = []
        for r in range(rows):
            s = streams[int(rng.integers(len(streams)))]
            j = int(rng.integers(0, min_len))
            sets.append(s[j : j + 1])
        vals.append(_distinct(sets))
    return float(np.mean(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--experts", required=True)
    ap.add_argument("--out")
    ap.add_argument("--ks", default="0,1,2,3,4,5,7")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    sweep = {int(m): v["step_ms_median"] for m, v in json.load(open(args.sweep))["sweep"].items() if v["step_ms_median"]}
    timing = _load(f"{args.experts}/routing_timing.npz")
    hidden = _load(f"{args.experts}/routing_hidden.npz")

    rows = sorted(sweep)
    e_dec = {m: e_decode(timing, m, rng) for m in rows}
    x = np.array([[1.0, m, e_dec[m]] for m in rows])
    y = np.array([sweep[m] for m in rows])
    (a, b, c), *_ = np.linalg.lstsq(x, y, rcond=None)
    fit_err = float(np.max(np.abs(x @ np.array([a, b, c]) - y) / y))

    ks = [int(v) for v in args.ks.split(",")]
    table = []
    for batch in (1, 8):
        base = a + b * batch + c * e_verify(timing, batch, 0, rng)
        for k in ks:
            row = {"B": batch, "k": k, "rows": batch * (1 + k)}
            for name, streams in (("timing", timing), ("hidden", hidden)):
                if batch > len(streams):
                    continue
                row[f"E_verify_{name}"] = round(e_verify(streams, batch, k, rng), 2)
            row["E_decode_same_rows"] = round(e_decode(timing, batch * (1 + k), rng), 2)
            est = a + b * batch * (1 + k) + c * row["E_verify_timing"]
            row["verify_ms_est"] = round(est, 3)
            row["cost_ratio_est"] = round(est / base, 3)
            if batch * (1 + k) in sweep:
                row["decode_ms_same_rows_measured"] = sweep[batch * (1 + k)]
                row["decode_ratio_measured"] = round(sweep[batch * (1 + k)] / sweep[batch], 3)
            table.append(row)
    out = {"fit": {"a_ms": a, "b_ms_per_row": b, "c_ms_per_expert": c, "max_rel_err": fit_err},
           "sweep_ms": sweep, "E_decode": e_dec, "table": table}
    print(json.dumps(out, indent=1))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
