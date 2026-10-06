"""K2: launch, gap and op-class profile of the decode steps of a served ref (megakernel headroom).

A ref is served under the gate's host lock. For each in-flight point the cache is flushed, the gate's
in-flight replay runs, and torch-profiler windows (CPU + GPU activities, no stacks, no shapes) are taken
inside the replay's timed window. SGLang wraps every `ModelRunner.forward` in a `step[<MODE> bs=..]`
range, which the trace carries on the GPU timeline as a `gpu_user_annotation`; kernels launched by
a `cudaGraphLaunch` share its correlation id. From those:

- **Cycles.** Forward k's cycle runs from its first GPU op to forward k+1's first GPU op, so the
  cycles tile the window and a decode cycle holds its graph replay, its sampling and the next
  forward's input preparation. A decode cycle followed by a decode forward is "clean"; per-step
  numbers are over clean cycles.
- **Per clean decode cycle:** kernel launches (graph nodes and eager launches), GPU busy time on the
  compute stream (union of op intervals), gaps (cycle minus busy) split into gaps between two graph
  nodes and the rest, launches under SMALL_US and their time, and busy time by op class.
- **Window shares:** wall time in decode cycles, extend cycles and idle (no forward in flight), the
  dilution of any decode-only gain.

Ops on other streams (HiCache transfer kernels) overlap the compute stream and are reported apart.

  python compute/k2_decode_profile.py serve --ref final-hc-cp2048-lpm --concurrency 8,12,28
  python compute/k2_decode_profile.py analyse --trace <trace.json.gz> [--dump-graph <out.tsv>]
"""

import argparse
import bisect
import collections
import glob
import gzip
import json
import os
import re
import sys
import threading
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SMALL_US = 5.0
STEP_RE = re.compile(r"^step\[(?P<mode>[A-Z_]+) bs=(?P<bs>\d+)(?: toks=(?P<toks>\d+))?")
GPU_OP_CATS = ("kernel", "gpu_memcpy", "gpu_memset")

# Ordered: the first match names the class. Built from the kernel names of the final config's decode
# graph (Triton attention, Triton fused_moe FP8, CUTLASS FP8 dense, FP8 KV); unknown names fall to
# "other" and are listed in the report so the table can be extended.
OP_CLASSES: Tuple[Tuple[str, str], ...] = (
    ("moe_gemm", r"fused_moe_kernel"),
    ("moe_route", r"moe_align|moe_sum|topk|top_k_softmax|gating|routing|count_and_sort|"
                  r"_moe_|expert"),
    ("norm_rope_kv", r"rmsnorm|RMSNorm|_norm|Norm|rope|rotary|store_kvcache|kvcache|set_kv"),
    ("attention", r"_fwd_kernel|stage1|stage2|decode_att|flash|attn|attention"),
    ("dense_gemm", r"cutlass|gemm(?!a)|Gemm|nvjet|cublas|sm120_|sm90_|_mm_|matmul|wgmma"),
    ("act_quant", r"per_token_quant|quant_fp8|scaled_fp8|fp8_quant|_quant|quantize"),
    ("sampling", r"argmax|sampl|multinomial|top_p|softmax|log_softmax|logit"),
    ("elementwise_copy", r"elementwise|copy|Copy|cat|index|fill|gather|scatter|reduce|Memcpy|Memset|"
                         r"act_and_mul|gelu|silu|tanh|clamp|where|arange|cumsum"),
)
_CLASS_RES = tuple((c, re.compile(p)) for c, p in OP_CLASSES)


def op_class(name: str) -> str:
    for cls, rx in _CLASS_RES:
        if rx.search(name):
            return cls
    return "other"


def _open(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def load_trace(path: str) -> Dict:
    with _open(path) as f:
        return json.load(f)


def extract(trace: Dict) -> Dict:
    """GPU ops, graph-launch correlation ids and GPU-side step spans of a chrome trace."""
    ops, steps, graph_corr = [], [], set()
    for e in trace["traceEvents"]:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        if cat in GPU_OP_CATS:
            a = e.get("args", {})
            ops.append({"name": e["name"], "ts": float(e["ts"]), "dur": float(e["dur"]), "cat": cat,
                        "stream": a.get("stream", e.get("tid")), "corr": a.get("correlation"),
                        "grid": a.get("grid")})
        elif cat == "gpu_user_annotation":
            m = STEP_RE.match(e["name"])
            if m:
                steps.append({"mode": m["mode"], "bs": int(m["bs"]), "toks": int(m["toks"] or 0),
                              "ts": float(e["ts"]), "dur": float(e["dur"])})
        elif cat in ("cuda_runtime", "cuda_driver") and "GraphLaunch" in e["name"]:
            corr = e.get("args", {}).get("correlation")
            if corr is not None:
                graph_corr.add(corr)
    ops.sort(key=lambda o: o["ts"])
    steps.sort(key=lambda s: s["ts"])
    for o in ops:
        o["graph"] = o["corr"] in graph_corr
    return {"ops": ops, "steps": steps, "n_graph_launches": len(graph_corr)}


def compute_stream(ops: Sequence[Dict]) -> object:
    """The stream that runs the decode graphs (else the busiest stream)."""
    by = collections.Counter()
    for o in ops:
        by[o["stream"]] += o["dur"] * (1000 if o["graph"] else 1)
    return by.most_common(1)[0][0]


def union_busy(intervals: Iterable[Tuple[float, float]]) -> float:
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def _clip(o: Dict, t0: float, t1: float) -> Tuple[float, float]:
    return max(o["ts"], t0), min(o["ts"] + o["dur"], t1)


def cycles(extracted: Dict) -> Dict:
    """Per-forward cycles on the compute stream, with launches, busy, gaps and class times."""
    ops, steps = extracted["ops"], extracted["steps"]
    stream = compute_stream(ops)
    main = [o for o in ops if o["stream"] == stream]
    side = [o for o in ops if o["stream"] != stream]
    starts = [o["ts"] for o in main]
    out = []
    for i, st in enumerate(steps[:-1]):
        t0, t1 = st["ts"], steps[i + 1]["ts"]
        lo, hi = bisect.bisect_left(starts, t0), bisect.bisect_left(starts, t1)
        cyc = main[lo:hi]
        if not cyc:
            continue
        busy = union_busy(_clip(o, t0, t1) for o in cyc)
        graph_ops = [o for o in cyc if o["graph"]]
        g0 = graph_ops[0]["ts"] if graph_ops else t1
        g1 = max(o["ts"] + o["dur"] for o in graph_ops) if graph_ops else t1
        # Exclusive time: the part of an op not overlapped by an earlier op. A kernel launched with
        # programmatic dependent launch starts while its predecessor drains and waits inside, so its
        # traced span overstates its cost; only the exclusive part is charged to its class.
        gaps = {"pre": 0.0, "graph": 0.0, "post": 0.0}
        cls_us: Dict[str, float] = collections.Counter()
        names: Dict[str, List[float]] = collections.defaultdict(lambda: [0, 0.0])
        small_n, small_us = 0, 0.0
        prev_end = t0
        for o in cyc:
            # The wait before the first graph node (its launch) belongs to the pre-graph phase.
            phase = "pre" if o["ts"] <= g0 else ("graph" if o["ts"] < g1 else "post")
            gaps[phase] += max(0.0, o["ts"] - prev_end)
            end = min(o["ts"] + o["dur"], t1)
            excl = max(0.0, end - max(o["ts"], prev_end))
            prev_end = max(prev_end, end)
            c = op_class(o["name"]) if o["cat"] == "kernel" else o["cat"]
            cls_us[c] += excl
            names[o["name"]][0] += 1
            names[o["name"]][1] += excl
            if o["cat"] == "kernel" and excl < SMALL_US:
                small_n += 1
                small_us += excl
        gaps["post"] += max(0.0, t1 - prev_end)
        out.append({
            "mode": st["mode"], "bs": st["bs"], "toks": st["toks"], "next_mode": steps[i + 1]["mode"],
            "t0": t0, "wall_us": t1 - t0, "fwd_span_us": st["dur"],
            "pre_us": g0 - t0, "graph_us": g1 - g0, "post_us": t1 - g1,
            "n_ops": len(cyc), "n_kernels": sum(o["cat"] == "kernel" for o in cyc),
            "n_graph_nodes": len(graph_ops),
            "busy_us": busy, "gap_us": (t1 - t0) - busy, "graph_gap_us": gaps["graph"],
            "pre_gap_us": gaps["pre"], "post_gap_us": gaps["post"],
            "small_n": small_n, "small_us": small_us, "class_us": dict(cls_us), "names": dict(names),
        })
    side_us = sum(o["dur"] for o in side)
    span = (steps[-1]["ts"] - steps[0]["ts"]) if len(steps) > 1 else 0.0
    return {"stream": stream, "cycles": out, "side_stream_us": side_us,
            "side_stream_names": collections.Counter(o["name"] for o in side).most_common(8), "span_us": span}


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


def summarize(cyc: Dict, min_cycles: int = 5) -> Dict:
    """Window shares by mode and per-step means of clean decode cycles, grouped by batch size."""
    cs = cyc["cycles"]
    total = sum(c["wall_us"] for c in cs)
    by_mode = collections.Counter()
    for c in cs:
        by_mode[c["mode"]] += c["wall_us"]
    clean = [c for c in cs if c["mode"] == "DECODE" and c["next_mode"] == "DECODE"]
    groups: Dict[int, List[Dict]] = collections.defaultdict(list)
    for c in clean:
        groups[c["bs"]].append(c)

    def agg(g: List[Dict]) -> Dict:
        wall = _mean([c["wall_us"] for c in g])
        cls = collections.Counter()
        for c in g:
            cls.update(c["class_us"])
        cls_mean = {k: v / len(g) for k, v in sorted(cls.items(), key=lambda kv: -kv[1])}
        r = {"n_cycles": len(g), "wall_us": wall,
             "fwd_span_us": _mean([c["fwd_span_us"] for c in g])}
        for k in ("n_kernels", "n_graph_nodes", "n_ops", "busy_us", "gap_us", "graph_gap_us", "pre_gap_us",
                  "post_gap_us", "pre_us", "graph_us", "post_us", "small_n", "small_us"):
            r[k] = _mean([c[k] for c in g])
            r[k + "_median"] = _median([c[k] for c in g])
        r["wall_us_median"] = _median([c["wall_us"] for c in g])
        r["gap_share"] = r["gap_us"] / wall
        r["small_share"] = r["small_us"] / wall
        r["class_us"] = cls_mean
        r["class_share"] = {k: v / wall for k, v in cls_mean.items()}
        return r

    per_bs = {bs: agg(g) for bs, g in sorted(groups.items()) if len(g) >= min_cycles}
    all_clean = agg(clean) if clean else {}
    return {
        "window_us": total,
        "mode_share": {m: v / total for m, v in by_mode.items()} if total else {},
        "n_cycles": collections.Counter(c["mode"] for c in cs),
        "n_clean_decode": len(clean),
        "decode_after_extend_wall_us": _mean([c["wall_us"] for c in cs
                                              if c["mode"] == "DECODE" and c["next_mode"] != "DECODE"]),
        "clean_decode": all_clean,
        "clean_decode_by_bs": per_bs,
        "side_stream_us": cyc["side_stream_us"],
        "side_stream_names": cyc["side_stream_names"],
    }


def kernel_table(cyc: Dict, mode: str = "DECODE") -> List[Dict]:
    """Per kernel name over clean cycles of `mode`: launches and time per cycle, class, graph or eager."""
    clean = [c for c in cyc["cycles"] if c["mode"] == mode and c["next_mode"] == mode]
    agg: Dict[str, List[float]] = collections.defaultdict(lambda: [0, 0.0])
    for c in clean:
        for n, (k, us) in c["names"].items():
            agg[n][0] += k
            agg[n][1] += us
    n = max(1, len(clean))
    rows = [{"name": name, "class": op_class(name), "launches_per_cycle": k / n, "us_per_cycle": us / n,
             "us_per_launch": us / k if k else 0.0} for name, (k, us) in agg.items()]
    return sorted(rows, key=lambda r: -r["us_per_cycle"])


def graph_sequence(extracted: Dict, which: int = -1) -> List[Dict]:
    """The ordered kernels of one decode graph replay (the `which`-th clean decode forward)."""
    ops, steps = extracted["ops"], extracted["steps"]
    dec = [s for i, s in enumerate(steps[:-1]) if s["mode"] == "DECODE" and steps[i + 1]["mode"] == "DECODE"]
    s = dec[which]
    seq = [o for o in ops if o["graph"] and s["ts"] <= o["ts"] < s["ts"] + s["dur"]]
    t_prev = None
    rows = []
    for o in seq:
        rows.append({"name": o["name"], "class": op_class(o["name"]), "dur_us": o["dur"],
                     "gap_before_us": (o["ts"] - t_prev) if t_prev is not None else 0.0, "grid": o["grid"]})
        t_prev = o["ts"] + o["dur"]
    return rows


def analyse(trace_path: str) -> Dict:
    ex = extract(load_trace(trace_path))
    cyc = cycles(ex)
    res = summarize(cyc)
    res["trace"] = trace_path
    res["n_graph_launches"] = ex["n_graph_launches"]
    res["kernels_decode"] = kernel_table(cyc)[:80]
    res["kernels_extend"] = kernel_table(cyc, "EXTEND")[:40]
    res["_extracted"] = ex
    return res


def profile_points(ref_name: str, points: Sequence[int], windows: Sequence[float], steps: int, out_dir: str,
                   window_s: float, load_name: str = "inflight") -> List[Dict]:
    """One server; per point: flush, the replay, profiler windows at the given replay seconds.

    The in-flight load gets a 60 s warm-up and a `window_s` window; any other gate load (e.g. pthink30)
    keeps its own warm-up and window, with only the concurrency replaced.
    """
    import msgspec
    import requests

    from gate import config, hostwatch, runner, server
    from workload import schema

    tok, sessions = runner.load_tokenizer(), schema.read(config.SESSIONS_PATH)
    ref = server.load_ref(ref_name)
    done = []
    with hostwatch.host_lock(), server.Server(ref, os.path.join(out_dir, "server.log"),
                                              extra_args=runner.server_extra_args(ref)) as srv:
        runner.warm_up(srv, sessions, tok, "k2-profile")
        for c in points:
            base = config.LOADS[load_name]
            load = msgspec.structs.replace(base, concurrency=c, name=f"{load_name}-C{c}-k2profile")
            if load_name == "inflight":
                load = msgspec.structs.replace(load, warmup_s=60.0, window_s=window_s)
            srv.flush_cache()
            result: Dict = {}
            t = threading.Thread(target=lambda: result.update(
                runner.replay_leg(srv, sessions, tok, load, "k2-profile", "start")))
            t_start = time.time()
            t.start()
            traces = []
            for i, at in enumerate(windows):
                time.sleep(max(0.0, t_start + at - time.time()))
                tdir = os.path.join(out_dir, f"C{c}", f"w{i}")
                r = requests.post(f"{srv.url}/start_profile", timeout=900,
                                  json={"output_dir": tdir, "num_steps": steps, "activities": ["CPU", "GPU"],
                                        "with_stack": False, "record_shapes": False})
                r.raise_for_status()
                traces.append({"dir": tdir, "requested_at_s": time.time() - t_start})
            t.join()
            summary = result.get("summary", {})
            rec = {"concurrency": c, "load": msgspec.to_builtins(load), "windows": traces,
                   "summary": {k: summary.get(k) for k in ("e2e_p50_s", "e2e_p90_s", "e2e_p99_s",
                                                           "output_tok_s_per_gpu", "n_failed", "n_requests")},
                   "gauges": result.get("gauges"), "decode_steps": result.get("decode_steps"),
                   "retractions": result.get("retractions")}
            done.append(rec)
            with open(os.path.join(out_dir, "points.json"), "w") as f:
                json.dump(done, f, indent=1)
    return done


def _fmt_row(c: Dict) -> str:
    return (f"wall {c['wall_us']:.0f} us, kernels {c['n_kernels']:.0f} (graph {c['n_graph_nodes']:.0f}), "
            f"busy {c['busy_us']:.0f}, gap {c['gap_us']:.0f} ({100 * c['gap_share']:.1f}%; pre-graph "
            f"{c['pre_gap_us']:.0f}, in-graph {c['graph_gap_us']:.0f}, post {c['post_gap_us']:.0f}), "
            f"small<{SMALL_US:g}us {c['small_n']:.0f} / {c['small_us']:.0f} us excl.")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--ref", default="final-hc-cp2048-lpm")
    s.add_argument("--concurrency", default="8,12,28")
    s.add_argument("--load", default="inflight", help="gate load name (inflight, pthink30, ...)")
    s.add_argument("--windows", default="75,150", help="replay seconds at which profiler windows start")
    s.add_argument("--window-s", type=float, default=180.0)
    s.add_argument("--steps", type=int, default=400)
    a = sub.add_parser("analyse")
    a.add_argument("--trace", required=True)
    a.add_argument("--out")
    a.add_argument("--dump-graph", help="write one decode graph replay's kernel sequence (tsv)")
    args = p.parse_args()

    if args.cmd == "serve":
        from gate import config, runner

        out_dir = os.path.join(config.RUNS_DIR, runner.new_exp_id(f"k2-profile-{args.ref}"))
        os.makedirs(out_dir)
        print(f"run dir: {out_dir}", file=sys.stderr, flush=True)
        points = [int(x) for x in args.concurrency.split(",")]
        windows = [float(x) for x in args.windows.split(",")]
        profile_points(args.ref, points, windows, args.steps, out_dir, args.window_s, args.load)
        for tr in sorted(glob.glob(os.path.join(out_dir, "C*", "w*", "*.trace.json*"))):
            res = analyse(tr)
            res.pop("_extracted")
            with open(os.path.join(os.path.dirname(tr), "k2_profile.json"), "w") as f:
                json.dump(res, f, indent=1)
            print(tr, json.dumps(res["mode_share"]), _fmt_row(res["clean_decode"]) if res["clean_decode"] else "")
        return

    res = analyse(args.trace)
    ex = res.pop("_extracted")
    if args.dump_graph:
        with open(args.dump_graph, "w") as f:
            f.write("i\tname\tclass\tdur_us\tgap_before_us\tgrid\n")
            for i, r in enumerate(graph_sequence(ex)):
                f.write(f"{i}\t{r['name']}\t{r['class']}\t{r['dur_us']:.2f}\t{r['gap_before_us']:.2f}\t{r['grid']}\n")
    out = args.out or os.path.join(os.path.dirname(args.trace), "k2_profile.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"mode_share": res["mode_share"], "n_cycles": res["n_cycles"],
                      "n_clean_decode": res["n_clean_decode"]}, indent=1))
    if res["clean_decode"]:
        print("clean decode:", _fmt_row(res["clean_decode"]))
        for bs, c in res["clean_decode_by_bs"].items():
            print(f"  bs {bs} (n {c['n_cycles']}):", _fmt_row(c))
        print(json.dumps(res["clean_decode"]["class_share"], indent=1))
    print(f"result: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
