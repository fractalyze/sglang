"""K2 headroom model: predicted end-to-end gain of each decode lever at each operating point.

A decode-only lever that saves s ms of an S ms decode step changes the GPU wall time by d * s / S, where
d is the decode share of wall time (the rest is prefill passes, which the lever does not touch). Under
requests always in flight, throughput scales as 1 / (1 - g) and E2E latency by about (1 - g), where g is
the wall-time fraction saved; under poisson chat sessions the same g raises the sessions a GPU holds at
the SLO by about 1 / (1 - g). Levers stack in order: each later lever sees the step and the decode
share left by the earlier ones.

Inputs (`compute/k2/inputs.json`): per point the untraced decode step (ms), the decode share range and
the baseline metrics; per lever and point the saving per decode step (ms, low and high) and an optional
prefill-side gain as a fraction of prefill time (low and high).

  python compute/k2_headroom.py compute/k2/inputs.json
"""

import json
import sys
from typing import Dict, List, Sequence, Tuple


def wall_gain(step_ms: float, decode_share: float, save_ms: float, prefill_gain: float = 0.0) -> float:
    """Fraction of GPU wall time saved: the decode part times its step saving plus the prefill part's gain."""
    return decode_share * save_ms / step_ms + (1.0 - decode_share) * prefill_gain


def after(step_ms: float, decode_share: float, save_ms: float, prefill_gain: float = 0.0) -> Tuple[float, float]:
    """The decode step and decode share once a lever is in."""
    new_step = step_ms - save_ms
    dec = decode_share * new_step / step_ms
    pre = (1.0 - decode_share) * (1.0 - prefill_gain)
    return new_step, dec / (dec + pre)


def stack(point: Dict, levers: Sequence[Dict], bound: str) -> List[Dict]:
    """Each lever's gain at one bound ("lo" or "hi"), applied in order on the point's decode step."""
    step, share = point["step_ms"], point["decode_share"][bound]
    out = []
    for lv in levers:
        save = lv["save_ms"][point["name"]][bound]
        pg = lv.get("prefill_gain", {"lo": 0.0, "hi": 0.0})[bound]
        g = wall_gain(step, share, save, pg)
        out.append({"lever": lv["name"], "gain": g, "step_ms": step, "decode_share": share, "save_ms": save})
        step, share = after(step, share, save, pg)
    return out


def predicted(base: Dict, g: float, closed_loop: bool) -> Dict:
    """Baseline metrics moved by a wall-time gain g."""
    out = {"usd_per_mtok": base["usd_per_mtok"] * (1.0 - g)}
    if closed_loop:
        out["tok_s"] = base["tok_s"] / (1.0 - g)
        out["p90_s"] = base["p90_s"] * (1.0 - g)
    else:
        out["sessions"] = base["sessions"] / (1.0 - g)
    return out


def main() -> None:
    spec = json.load(open(sys.argv[1]))
    for point in spec["points"]:
        lo, hi = stack(point, spec["levers"], "lo"), stack(point, spec["levers"], "hi")
        print(f"\n{point['name']}: decode step {point['step_ms']:.2f} ms, decode share "
              f"{point['decode_share']['lo']:.2f}-{point['decode_share']['hi']:.2f}")
        g_lo = g_hi = 0.0
        for a, b in zip(lo, hi):
            g_lo, g_hi = 1 - (1 - g_lo) * (1 - a["gain"]), 1 - (1 - g_hi) * (1 - b["gain"])
            p_lo = predicted(point["base"], a["gain"], point["closed_loop"])
            p_hi = predicted(point["base"], b["gain"], point["closed_loop"])
            print(f"  {a['lever']:28s} save {a['save_ms']:.2f}-{b['save_ms']:.2f} ms of {a['step_ms']:.2f}-"
                  f"{b['step_ms']:.2f} at share {a['decode_share']:.2f}-{b['decode_share']:.2f}: "
                  f"E2E {100 * a['gain']:.1f}-{100 * b['gain']:.1f}%  "
                  + "  ".join(f"{k} {p_lo[k]:.3g}-{p_hi[k]:.3g}" for k in p_lo)
                  + f"  | cumulative {100 * g_lo:.1f}-{100 * g_hi:.1f}%")


if __name__ == "__main__":
    main()
