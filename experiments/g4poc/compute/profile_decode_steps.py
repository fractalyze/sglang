"""K1 sizing: kernel launches per decode step and the decode steps' share of GPU wall time, at a served load.

A ref is served once under the gate's host lock. For each load point (in-flight N, or a think-time load) the
replay runs past its warm-up, a GPU-only torch-profiler window is taken, and the cache is flushed before the
next point. Decode steps run as CUDA-graph replays: every kernel carries the correlation id of the
`cudaGraphLaunch` that issued it, so one graph launch = one decode step. Per step this reports the kernels
launched, the GPU span (first kernel start to last kernel end), the captured batch size (grid x of the Triton
decode-attention stage-1 kernel), and a per-step histogram of kernel names. Kernels outside graph replays
(extend steps, sampling, copies) are summed separately.

What a launch-removing fusion saves follows from these numbers: removed launches per step x the saving per launch
(gemma4nv T4: about the whole traced 2.0-2.3 us under decode CUDA graphs) x decode steps per second.

  python compute/profile_decode_steps.py --ref final-hc-cp2048-lpm --points inflight:12,inflight:28
  python compute/profile_decode_steps.py --ref final-mem-c1-c2a --points pthink30:72 --delay-s 330
  python compute/profile_decode_steps.py --trace <trace.json.gz>      # re-analyse a trace
"""

import argparse
import collections
import glob
import gzip
import json
import os
import re
import statistics
import sys
import threading
import time
from typing import Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Functor or op names inside an ATen kernel's template arguments; they tell the elementwise ops apart. The
# generic wrappers (a binary op with one scalar operand is BUnaryFunctor<..., MulFunctor>) name no op.
_ATEN_OP = re.compile(r"(\w+Functor\w*|direct_copy_kernel_cuda|\w+_kernel_cuda|\w+_kernel_impl)")
_ATEN_WRAPPERS = frozenset({"BinaryFunctor", "AUnaryFunctor", "BUnaryFunctor", "UnaryFunctor"})


def short_name(name: str) -> str:
    """A kernel name without template arguments or parameters; ATen kernels keep the op they run."""
    base = name.removeprefix("void ").split("<", 1)[0].split("(", 1)[0].strip()
    if base.startswith("at::native::") or base.startswith("at_cuda_detail::"):
        ops = [op for op in _ATEN_OP.findall(name) if op not in _ATEN_WRAPPERS]
        if ops:
            return f"{base}[{ops[0]}]"
    return base


def _batch_size(kernels: Iterable[Dict]) -> Optional[int]:
    for k in kernels:
        if "stage1" in k["name"]:
            grid = k.get("args", {}).get("grid")
            if grid:
                return int(grid[0])
    return None


def graph_steps(events: Iterable[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """(decode steps, kernels outside any graph replay). A step is the kernels of one cudaGraphLaunch."""
    events = list(events)
    launches = {e["args"]["correlation"] for e in events
                if e.get("ph") == "X" and e.get("cat") == "cuda_runtime" and e.get("name") == "cudaGraphLaunch"}
    by_launch: Dict[int, List[Dict]] = collections.defaultdict(list)
    eager: List[Dict] = []
    for e in events:
        if e.get("ph") != "X" or e.get("cat") != "kernel":
            continue
        c = e.get("args", {}).get("correlation")
        (by_launch[c] if c in launches else eager).append(e)
    steps = []
    for ks in by_launch.values():
        ks.sort(key=lambda e: e["ts"])
        steps.append({
            "t0": ks[0]["ts"],
            "span_us": ks[-1]["ts"] + ks[-1]["dur"] - ks[0]["ts"],
            "busy_us": sum(k["dur"] for k in ks),
            "n_kernels": len(ks),
            "bs": _batch_size(ks),
            "names": collections.Counter(short_name(k["name"]) for k in ks),
        })
    steps.sort(key=lambda s: s["t0"])
    return steps, eager


def summarize(events: Iterable[Dict], top: int = 60) -> Dict:
    """Launches per decode step (overall and per captured batch size) and the decode share of the window."""
    events = list(events)
    steps, eager = graph_steps(events)
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"]
    if not kernels:
        raise ValueError("no kernel events")
    t0 = min(e["ts"] for e in kernels)
    span = max(e["ts"] + e["dur"] for e in kernels) - t0
    by_bs: Dict[int, List[Dict]] = collections.defaultdict(list)
    for s in steps:
        by_bs[s["bs"] if s["bs"] is not None else -1].append(s)

    def per_step_names(group: List[Dict]) -> Dict[str, float]:
        total: collections.Counter = collections.Counter()
        for s in group:
            total.update(s["names"])
        return {n: c / len(group) for n, c in total.most_common(top)}

    decode_span = sum(s["span_us"] for s in steps)
    out = {
        "window_span_us": span,
        "n_decode_steps": len(steps),
        "decode_steps_per_s": len(steps) / (span / 1e6) if span else 0.0,
        "decode_span_us": decode_span,
        "decode_share_of_span": decode_span / span,
        "eager_kernels": len(eager),
        "eager_busy_us": sum(e["dur"] for e in eager),
        "eager_busy_share_of_span": sum(e["dur"] for e in eager) / span,
        "kernels_per_step": {
            "median": statistics.median(s["n_kernels"] for s in steps) if steps else None,
            "min": min((s["n_kernels"] for s in steps), default=None),
            "max": max((s["n_kernels"] for s in steps), default=None),
        },
        "by_batch_size": {},
    }
    for bs in sorted(by_bs):
        group = by_bs[bs]
        out["by_batch_size"][str(bs)] = {
            "steps": len(group),
            "kernels_per_step_median": statistics.median(s["n_kernels"] for s in group),
            "span_us_median": statistics.median(s["span_us"] for s in group),
            "busy_us_median": statistics.median(s["busy_us"] for s in group),
        }
    if steps:
        common = max(by_bs, key=lambda b: len(by_bs[b]))
        out["names_per_step"] = {"bs": common, "names": per_step_names(by_bs[common])}
        out["mean_bs"] = statistics.fmean(s["bs"] for s in steps if s["bs"] is not None)
    return out


def load_events(trace_path: str) -> List[Dict]:
    with (gzip.open(trace_path, "rt") if trace_path.endswith(".gz") else open(trace_path)) as f:
        return json.load(f)["traceEvents"]


def parse_points(spec: str) -> List[Tuple[str, int]]:
    """'inflight:12,pthink30:72' -> [('inflight', 12), ('pthink30', 72)]."""
    out = []
    for item in spec.split(","):
        load, conc = item.split(":")
        out.append((load, int(conc)))
    return out


def profile(ref_name: str, points: List[Tuple[str, int]], steps: int, delay_s: Dict[str, float],
            out_dir: str) -> Dict[str, str]:
    """Serves `ref_name` once and profiles `steps` forward steps at each point; returns point label -> trace."""
    import msgspec
    import requests

    from gate import config, hostwatch, runner, server
    from workload import schema

    tok, sessions = runner.load_tokenizer(), schema.read(config.SESSIONS_PATH)
    ref = server.load_ref(ref_name)
    traces: Dict[str, str] = {}
    with hostwatch.host_lock(), server.Server(ref, os.path.join(out_dir, "server.log"),
                                              extra_args=runner.server_extra_args(ref)) as srv:
        runner.warm_up(srv, sessions, tok, "profile")
        for load_name, conc in points:
            label = f"{load_name}-C{conc}"
            base_load = config.LOADS[load_name]
            delay = delay_s.get(load_name, base_load.warmup_s + 30.0)
            # Long enough to cover the profile window; the summary is over this shortened window.
            load = msgspec.structs.replace(base_load, concurrency=conc, name=f"{label}-profile",
                                           window_s=max(120.0, delay - base_load.warmup_s + 120.0))
            trace_dir = os.path.join(out_dir, f"trace-{label}")
            result: Dict = {}
            t = threading.Thread(target=lambda: result.update(
                runner.replay_leg(srv, sessions, tok, load, "profile", "start")))
            t.start()
            time.sleep(delay)
            r = requests.post(f"{srv.url}/start_profile", timeout=900,
                              json={"output_dir": trace_dir, "num_steps": steps, "activities": ["GPU"]})
            r.raise_for_status()
            t.join()
            summary = {k: result["summary"][k] for k in ("e2e_p90_s", "output_tok_s_per_gpu", "n_failed")
                       if k in result.get("summary", {})}
            with open(os.path.join(out_dir, f"replay-{label}.json"), "w") as f:
                json.dump({"load": msgspec.to_builtins(load), "summary": summary,
                           "gauges": result.get("gauges"), "decode_steps": result.get("decode_steps")}, f, indent=1)
            found = sorted(glob.glob(os.path.join(trace_dir, "*.trace.json*")))
            if not found:
                raise RuntimeError(f"no trace written under {trace_dir}")
            traces[label] = found[0]
            srv.flush_cache()
    return traces


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ref")
    p.add_argument("--points", default="inflight:12,inflight:28", help="load:concurrency,...")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--delay-s", type=float, help="replay seconds before each profile (default: warm-up + 30)")
    p.add_argument("--trace", help="analyse this trace instead of serving")
    p.add_argument("--out", help="result json (default: next to the traces)")
    args = p.parse_args()

    if args.trace:
        res = {"trace": summarize(load_events(args.trace))}
        out = args.out or os.path.join(os.path.dirname(os.path.dirname(args.trace)), "decode_steps.json")
    else:
        from gate import config, runner

        points = parse_points(args.points)
        out_dir = os.path.join(config.RUNS_DIR, runner.new_exp_id(f"decsteps-{args.ref}"))
        os.makedirs(out_dir)
        delays = {load: args.delay_s for load, _ in points} if args.delay_s else {}
        traces = profile(args.ref, points, args.steps, delays, out_dir)
        res = {label: dict(summarize(load_events(tr)), trace=tr) for label, tr in traces.items()}
        out = args.out or os.path.join(out_dir, "decode_steps.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    for label, r in res.items():
        brief = {k: r.get(k) for k in ("n_decode_steps", "decode_steps_per_s", "decode_share_of_span",
                                        "eager_busy_share_of_span", "kernels_per_step", "mean_bs")}
        print(label, json.dumps(brief))
        for bs, row in r["by_batch_size"].items():
            print(f"  bs {bs}: {json.dumps(row)}")
    print(f"result: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
