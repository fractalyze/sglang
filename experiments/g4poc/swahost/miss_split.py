"""Per-turn prefix-miss split of a HiCache think-time run, from the server's PFXDBG admission log.

Needs a server started with G4POC_DEBUG_PREFIX_SPLIT=1 on a tree that logs `sess=` and `full_kv=`
(jumanzii/g4poc-swa-host-dbg, c6f392a28b). Each admission line names the session instance (`sess`: a
hash of the first 32 prompt ids, which hold the per-session nonce), the prompt length (`fill`), the
device hit, the host hit the match promised, what the load-back restored (`loaded`), and `full_kv`:
the Full-layer prefix present on device or host before the SWA validator trims the match.

A session's turn k >= 1 can reuse at most its previous prompt minus the empty-thought generation
suffix: Gemma-4's template renders a past model turn as `<|turn>model\\n<reply>`, so the match ends at
the previous prompt's `<|turn>model\\n` (WORKLOAD.md). `expected` is that length. Classes:

  first      turn 0 of a session instance (always a miss)
  hit        device_hit + loaded >= expected - tol
  declined   the match promised the prefix (device_hit + host_hit) but the load-back restored less
  swa_gone   the Full-layer prefix is still present (full_kv >= expected - tol) but the match stopped
             short: the sliding window behind the match end had no device or host SWA copy
  full_gone  the Full-layer prefix itself is gone from device and host (full_kv < expected - tol)

Token view: every prompt token is either reused, or recomputed for one of: first turn, the per-turn
suffix past `expected` (previous reply + new user message: never reusable), or a returning turn's
shortfall attributed to its class.

`window_trimmed` drops the window's last `--tail-s` seconds: harness fd08bf8fbf's `slots` generator starts a
fresh session (an uncached turn 0) in every slot whose next turn falls past the window end, which inflates
first turns and load at the end of each point (PC2; fixed in dff92efc2b, not deployed for these runs).

usage: python miss_split.py <server.log> [--warmup-s 240] [--window-s 480] [--tail-s 60] [--suffix 4] [--tol 64]
"""

import argparse
import collections
import datetime
import json
import re

LINE = re.compile(r"^\[(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[^\]]*\].*PFXDBG (?P<kv>rid=.*)$")
INT_FIELDS = (
    "fill", "device_hit", "host_hit", "swa_host_hit", "loaded", "full_kv", "hfree_kv", "hfree_swa",
    "dfree_full", "dfree_swa", "bk_fail_full", "bk_fail_write", "bk_fail_tok", "dropped",
)
CLASSES = ("first", "hit", "declined", "swa_gone", "full_gone")


def parse(path):
    rows, seen = [], set()
    with open(path, errors="replace") as f:
        for raw in f:
            m = LINE.match(raw.rstrip("\n"))
            if not m:
                continue
            kv = dict(tok.split("=", 1) for tok in m.group("kv").split() if "=" in tok)
            if "sess" not in kv or kv["rid"] in seen:  # first admission of a rid only (retractions re-admit)
                continue
            seen.add(kv["rid"])
            row = {k: int(kv[k]) for k in INT_FIELDS if k in kv and kv[k].lstrip("-").isdigit()}
            row["rid"], row["sess"] = kv["rid"], kv["sess"]
            row["t"] = datetime.datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S").timestamp()
            rows.append(row)
    return rows


def classify(rows, suffix, tol):
    last_fill = {}
    for r in rows:  # log order is admission order
        prev = last_fill.get(r["sess"])
        last_fill[r["sess"]] = r["fill"]
        if prev is None:
            r["cls"], r["expected"] = "first", 0
            continue
        expected = max(0, prev - suffix)
        usable = r["device_hit"] + r["loaded"]
        r["expected"] = expected
        if usable >= expected - tol:
            r["cls"] = "hit"
        elif r["device_hit"] + r["host_hit"] >= expected - tol:
            r["cls"] = "declined"
        elif r.get("full_kv", -1) >= expected - tol:
            r["cls"] = "swa_gone"
        else:
            r["cls"] = "full_gone"
    return rows


def summarize(rows, t0, lo, hi):
    sel = [r for r in rows if lo <= r["t"] - t0 < hi]
    n = collections.Counter(r["cls"] for r in sel)
    tok = collections.Counter()
    for r in sel:
        usable = min(r["fill"], r["device_hit"] + r["loaded"])
        tok["prompt"] += r["fill"]
        tok["reused"] += usable
        if r["cls"] == "first":
            tok["first"] += r["fill"] - usable
            continue
        tok["suffix"] += max(0, r["fill"] - max(r["expected"], usable))
        tok[r["cls"]] += max(0, r["expected"] - usable)
    returning = sum(n[c] for c in CLASSES if c != "first")
    missed = returning - n["hit"]
    out = {
        "admissions": len(sel),
        "sessions": len({r["sess"] for r in sel}),
        "turns": {c: n[c] for c in CLASSES},
        "missed_returning_share": {c: round(n[c] / missed, 3) if missed else 0.0
                                   for c in ("declined", "swa_gone", "full_gone")},
        "prompt_tokens": tok["prompt"],
        "token_share": {k: round(tok[k] / tok["prompt"], 4) if tok["prompt"] else 0.0
                        for k in ("reused", "first", "suffix", "declined", "swa_gone", "full_gone")},
    }
    gone = [r for r in sel if r["cls"] in ("swa_gone", "full_gone")]
    if gone:
        # How much of the expected prefix the Full layers still had, and how much the match kept.
        out["full_kv_over_expected_p50"] = sorted(
            r.get("full_kv", 0) / max(1, r["expected"]) for r in gone)[len(gone) // 2]
        out["usable_over_expected_p50"] = sorted(
            (r["device_hit"] + r["loaded"]) / max(1, r["expected"]) for r in gone)[len(gone) // 2]
    if sel:
        last = sel[-1]
        out["counters_at_end"] = {k: last.get(k) for k in ("bk_fail_full", "bk_fail_write", "bk_fail_tok",
                                                             "dropped")}
        for k in ("hfree_kv", "hfree_swa", "dfree_full", "dfree_swa"):
            vals = sorted(r[k] for r in sel if k in r)
            if vals:
                out[k + "_p10_p50"] = [vals[len(vals) // 10], vals[len(vals) // 2]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--warmup-s", type=float, default=240.0)
    ap.add_argument("--window-s", type=float, default=480.0)
    ap.add_argument("--tail-s", type=float, default=60.0,
                    help="seconds cut from the window's end for window_trimmed (end-of-point generator bias)")
    ap.add_argument("--suffix", type=int, default=4,
                    help="tokens of the empty-thought generation suffix past `<|turn>model\\n`")
    ap.add_argument("--tol", type=int, default=64)
    ap.add_argument("--point-gap-s", type=float, default=60.0,
                    help="an admission gap this long starts a new sweep point")
    args = ap.parse_args()
    rows = parse(args.log)
    if not rows:
        raise SystemExit("no PFXDBG lines with sess= in " + args.log)
    # A sweep runs its points back to back on one server; split them at idle gaps (drain + next start).
    points, cur = [], [rows[0]]
    for prev, r in zip(rows, rows[1:]):
        if r["t"] - prev["t"] > args.point_gap_s:
            points.append(cur)
            cur = []
        cur.append(r)
    points.append(cur)
    result = []
    for i, pts in enumerate(points):
        classify(pts, args.suffix, args.tol)
        t0 = pts[0]["t"]
        result.append({
            "point": i,
            "start": datetime.datetime.fromtimestamp(t0).strftime("%H:%M:%S"),
            "window": summarize(pts, t0, args.warmup_s, args.warmup_s + args.window_s),
            "window_trimmed": summarize(pts, t0, args.warmup_s, args.warmup_s + args.window_s - args.tail_s),
            "all": summarize(pts, t0, 0, float("inf")),
        })
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
