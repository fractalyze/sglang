"""C2-A sizing: the Triton extend-attention kernels' share of GPU time at the gated load.

A ref is served under the gate's host lock and replayed at N always in flight (prefix cache on,
the gate's warm-up first). A GPU-only torch-profiler window is taken once the replay is past its
warm-up, and every kernel in it is attributed:

- extend attention (`_fwd_kernel`, `_fwd_kernel_unified`, `_fwd_kernel_dense_prefill`), split
  into the full layers (head_dim 512) and the sliding layers (head_dim 256) by layer position:
  an extend forward launches one extend kernel per layer, in layer order, and Gemma-4-26B-A4B's
  full layers are 5, 11, 17, 23 and 29 of 30. The launch signature (block, registers, shared
  memory) is reported next to it as a cross-check;
- decode attention (`stage1` / `stage2` kernels) and everything else.

Shares are of the window's wall span, so the predicted end-to-end gain of a kernel speedup s on
a share f is f * (1 - 1/s), summed over the two head dims.

  python compute/profile_extend_share.py --ref base --concurrency 12 --steps 600
  python compute/profile_extend_share.py --trace <trace.json.gz>        # re-analyse a trace
"""

import argparse
import collections
import glob
import gzip
import json
import os
import sys
import threading
import time
from typing import Dict, Iterable, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_LAYERS = 30
FULL_LAYERS = frozenset({5, 11, 17, 23, 29})
EXTEND_KERNELS = ("_fwd_kernel", "_fwd_kernel_unified", "_fwd_kernel_dense_prefill")


def kind(name: str) -> str:
    if name in EXTEND_KERNELS:
        return "extend"
    if "stage1" in name or "stage2" in name:
        return "decode_attn"
    return "other"


def _signature(e: Dict) -> str:
    a = e.get("args", {})
    return json.dumps([e["name"], a.get("block"), a.get("registers per thread"), a.get("shared memory")])


def attribute(kernels: Iterable[Dict], n_layers: int = N_LAYERS, full_layers=FULL_LAYERS) -> Dict:
    """Kernel events (ph "X", cat "kernel") -> time per class, extend time split by head dim."""
    ks = sorted(kernels, key=lambda e: e["ts"])
    if not ks:
        raise ValueError("no kernel events")
    span = ks[-1]["ts"] + ks[-1]["dur"] - ks[0]["ts"]
    us: Dict[str, float] = collections.Counter()
    sigs: Dict[str, Dict] = {}
    extend_i = 0
    for e in ks:
        k = kind(e["name"])
        us[k] += e["dur"]
        if k == "extend":
            layer = extend_i % n_layers
            extend_i += 1
            hd = "hd512" if layer in full_layers else "hd256"
            us[f"extend_{hd}"] += e["dur"]
            sig = _signature(e)
            s = sigs.setdefault(sig, {"signature": sig, "n": 0, "us": 0.0, "hd512": 0, "hd256": 0})
            s["n"] += 1
            s["us"] += e["dur"]
            s[hd] += 1
    busy = sum(e["dur"] for e in ks)
    return {
        "span_us": span,
        "busy_us": busy,
        "n_kernels": len(ks),
        "n_extend_launches": extend_i,
        "extend_launches_whole_forwards": extend_i % n_layers == 0,
        "us": dict(us),
        "share_of_span": {k: v / span for k, v in us.items()},
        "share_of_busy": {k: v / busy for k, v in us.items()},
        # Each signature should map to one head dim; a mixed row means the position split is off.
        "extend_signatures": sorted(sigs.values(), key=lambda s: -s["us"]),
    }


def predict(shares: Dict[str, float], speedups: Dict[str, float]) -> Dict:
    per = {hd: shares.get(f"extend_{hd}", 0.0) * (1 - 1 / s) for hd, s in speedups.items()}
    return {"per_head_dim": per, "e2e_gain": sum(per.values()), "speedups": speedups}


def bench_speedups(bench_json: str) -> Dict[str, float]:
    """Best tile mix speedup per head dim from compute/extend_attn_bench.py's output."""
    with open(bench_json) as f:
        res = json.load(f)["results"]
    return {f"hd{r['head_dim']}": max(b["mix_speedup"] for b in r["best"]) for r in res}


def load_kernels(trace_path: str) -> List[Dict]:
    with (gzip.open(trace_path, "rt") if trace_path.endswith(".gz") else open(trace_path)) as f:
        trace = json.load(f)
    return [e for e in trace["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "kernel"]


def profile(ref_name: str, concurrency: int, steps: int, delay_s: float, out_dir: str) -> str:
    """Serves `ref_name`, replays at `concurrency` in flight, profiles `steps` forward steps; returns the trace."""
    import msgspec
    import requests

    from gate import config, hostwatch, runner, server
    from workload import schema

    tok, sessions = runner.load_tokenizer(), schema.read(config.SESSIONS_PATH)
    load = msgspec.structs.replace(config.INFLIGHT, concurrency=concurrency, name=f"inflight-C{concurrency}-profile",
                                   warmup_s=60.0, window_s=120.0)
    trace_dir = os.path.join(out_dir, "trace")
    ref = server.load_ref(ref_name)
    with hostwatch.host_lock(), server.Server(ref, os.path.join(out_dir, "server.log"),
                                              extra_args=runner.server_extra_args(ref)) as srv:
        runner.warm_up(srv, sessions, tok, "profile")
        result: Dict = {}
        t = threading.Thread(target=lambda: result.update(
            runner.replay_leg(srv, sessions, tok, load, "profile", "start")))
        t.start()
        time.sleep(delay_s)
        r = requests.post(f"{srv.url}/start_profile", timeout=900,
                          json={"output_dir": trace_dir, "num_steps": steps, "activities": ["GPU"]})
        r.raise_for_status()
        t.join()
        summary = {k: result["summary"][k] for k in ("e2e_p90_s", "output_tok_s_per_gpu", "n_failed")
                   if k in result.get("summary", {})}
        with open(os.path.join(out_dir, "replay_summary.json"), "w") as f:
            json.dump({"load": msgspec.to_builtins(load), "summary": summary,
                       "gauges": result.get("gauges")}, f, indent=1)
    traces = sorted(glob.glob(os.path.join(trace_dir, "*.trace.json*")))
    if not traces:
        raise RuntimeError(f"no trace written under {trace_dir}")
    return traces[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ref", default="base")
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--delay-s", type=float, default=90.0, help="replay seconds before the profile starts")
    p.add_argument("--trace", help="analyse this trace instead of serving")
    p.add_argument("--bench", default="/data/jooman/g4poc/c2/bench.json")
    p.add_argument("--out", help="result json (default: next to the trace)")
    args = p.parse_args()

    trace = args.trace
    if not trace:
        from gate import config, runner

        out_dir = os.path.join(config.RUNS_DIR, runner.new_exp_id(f"profile-{args.ref}-C{args.concurrency}"))
        os.makedirs(out_dir)
        trace = profile(args.ref, args.concurrency, args.steps, args.delay_s, out_dir)
    res = attribute(load_kernels(trace))
    res["trace"] = trace
    if os.path.exists(args.bench):
        res["prediction"] = predict(res["share_of_span"], bench_speedups(args.bench))
    out = args.out or os.path.join(os.path.dirname(os.path.dirname(trace)), "extend_share.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("span_us", "n_extend_launches", "extend_launches_whole_forwards",
                                          "share_of_span", "share_of_busy") if k in res}, indent=1))
    if "prediction" in res:
        print(json.dumps(res["prediction"], indent=1))
    for s in res["extend_signatures"][:6]:
        print(s)
    print(f"result: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
