"""Per-point prefill passes and per-step decode rates from a `gate sweep` server log (timed windows only).

Each sweep point starts with a cache flush, so the log's last len(names) flush segments are the points in order;
each point's timed window is 240-720 s after its first timestamped line (the think-time loads' warm-up and window).
Per-step decode rate = the decode line's gen throughput / its running count, grouped by running count: two hosts
running the same config at the same batch should match, which separates a slower host from extra prefill work.

  python compute/server_log_compare.py <server.log> C48,C64,C80[,C96]
"""

import re
import statistics
import sys
from datetime import datetime
from typing import Dict, List

FLUSH = "Cache flushed successfully"
TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]")
DEC = re.compile(r"Decode batch, #running-req: (\d+),.*gen throughput \(token/s\): ([\d.]+)")
PRE = re.compile(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)")
WARMUP_S, WINDOW_S = 240.0, 480.0
MIN_STEPS = 20


def point_stats(segment: str, warmup_s: float = WARMUP_S, window_s: float = WINDOW_S) -> Dict:
    t0 = None
    steps: Dict[int, List[float]] = {}
    passes = new_tokens = 0
    for line in segment.splitlines():
        m = TS.match(line)
        if not m:
            continue
        t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
        t0 = t if t0 is None else t0
        if not t0 + warmup_s <= t < t0 + warmup_s + window_s:
            continue
        d = DEC.search(line)
        if d:
            steps.setdefault(int(d.group(1)), []).append(float(d.group(2)) / int(d.group(1)))
            continue
        p = PRE.search(line)
        if p:
            passes += 1
            new_tokens += int(p.group(2))
    return {"prefill_passes": passes, "new_tokens": new_tokens,
            "new_tokens_per_pass": new_tokens / passes if passes else 0.0,
            "decode_steps_per_s": {n: statistics.median(v) for n, v in sorted(steps.items()) if len(v) >= MIN_STEPS}}


def points(log_text: str, names: List[str]) -> Dict[str, Dict]:
    segments = log_text.split(FLUSH)[-len(names):]
    return {name: point_stats(seg) for name, seg in zip(names, segments)}


def main() -> None:
    with open(sys.argv[1]) as f:
        res = points(f.read(), sys.argv[2].split(","))
    for name, s in res.items():
        print(f"{name}: prefill passes {s['prefill_passes']}, new tokens {s['new_tokens']}, "
              f"per pass {s['new_tokens_per_pass']:.0f}")
        print("   decode steps/s by batch: " + " ".join(
            f"b{n}:{r:.1f}" for n, r in s["decode_steps_per_s"].items() if 4 <= n <= 16))


if __name__ == "__main__":
    main()
