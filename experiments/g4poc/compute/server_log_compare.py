"""Per-point prefill passes and per-step decode rates from a `gate sweep` server log (timed windows only).

Each sweep point starts with a cache flush, so the log's last len(names) flush segments are the points in order;
each point's timed window is 240-720 s after its flush (the think-time loads' warm-up and window). Per-step decode
rate = the decode line's gen throughput / its running count, grouped by running count: two hosts running the same
config at the same batch should match, which separates a slower host from extra prefill work.

  python compute/server_log_compare.py <server.log> C48,C64,C80[,C96]
"""
import re, sys, statistics as st
from datetime import datetime
TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]")
DEC = re.compile(r"Decode batch, #running-req: (\d+),.*gen throughput \(token/s\): ([\d.]+)")
PRE = re.compile(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)")
text = open(sys.argv[1]).read().split("Cache flushed successfully")
names = sys.argv[2].split(",")
segs = text[-len(names):]
for name, seg in zip(names, segs):
    lines = seg.splitlines()
    t0 = None; dec = {}; npre = 0; newtok = 0; seqs = 0
    for l in lines:
        m = TS.match(l)
        if not m: continue
        t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
        t0 = t if t0 is None else t0
        if not (t0 + 240 <= t < t0 + 720): continue
        d = DEC.search(l)
        if d:
            dec.setdefault(int(d.group(1)), []).append(float(d.group(2)) / int(d.group(1)))
            continue
        p = PRE.search(l)
        if p:
            npre += 1; seqs += int(p.group(1)); newtok += int(p.group(2))
    steps = " ".join(f"b{n}:{st.median(v):.1f}({len(v)})" for n, v in sorted(dec.items()) if 4 <= n <= 16 and len(v) >= 20)
    print(f"{name}: prefill batches {npre}, new tok {newtok}, new tok/batch {newtok/max(npre,1):.0f}, prefill batches/s {npre/480:.2f}")
    print(f"   decode steps/s by batch: {steps}")
