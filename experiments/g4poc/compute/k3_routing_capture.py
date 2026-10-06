"""K3: real decode routing of the served model, per forward pass and layer (SGLang's expert-distribution recorder).

The fused_moe weight floor at decode depends on how many distinct experts a step touches, which depends on how the
running requests' tokens route, not on uniform top-8 draws. This serves a ref with `--expert-distribution-recorder-mode
per_pass` (k3/refs.json), runs the gate's in-flight replay at each concurrency, records a window of passes inside the
timed window (`/start_expert_distribution_record` ... `/dump_expert_distribution_record`, a .pt per dump under the run
dir), and summarizes the decode passes: per batch size, the distinct experts per layer and the per-expert token loads.

  python compute/k3_routing_capture.py serve [--ref ...] [--concurrency 12,28]
  python compute/k3_routing_capture.py summarize <dump.pt> [...]
"""

import argparse
import collections
import glob
import json
import os
import sys
import threading
import time
from typing import Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOP_K = 8


def decode_passes(records: Sequence[Dict], max_tokens: int = 64) -> List[Dict]:
    """Per pass: tokens and the [layer, expert] token counts, for passes of at most max_tokens tokens (decode)."""
    out = []
    for r in records:
        counts = r["global_physical_count"]
        counts = counts.reshape(counts.shape[-2], counts.shape[-1]) if counts.dim() > 2 else counts
        per_layer = counts.sum(dim=1)
        tokens = int(per_layer.max().item()) // TOP_K
        if 0 < tokens <= max_tokens:
            out.append({"tokens": tokens, "counts": counts})
    return out


def summarize(passes: Sequence[Dict]) -> Dict:
    """Per batch size: mean distinct experts per layer, the max per-expert load, and the share of slots on the busiest
    experts."""
    by = collections.defaultdict(list)
    for p in passes:
        by[p["tokens"]].append(p["counts"])
    out = {}
    for bs, cs in sorted(by.items()):
        distinct = [float((c > 0).sum(dim=1).float().mean()) for c in cs]
        maxload = [float(c.max(dim=1).values.float().mean()) for c in cs]
        out[bs] = {"n_passes": len(cs), "distinct_experts_per_layer": sum(distinct) / len(distinct),
                   "distinct_min": min(distinct), "distinct_max": max(distinct),
                   "max_tokens_per_expert": sum(maxload) / len(maxload),
                   "uniform_expectation": 128 * (1 - (1 - TOP_K / 128) ** bs)}
    return out


def load_records(paths: Sequence[str]) -> List[Dict]:
    import torch

    recs = []
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        for r in d["records"]:
            spd = r.get("single_pass_data", r)
            recs.append({"global_physical_count": spd["global_physical_count"]})
    return recs


def capture(ref_name: str, points: Sequence[int], out_dir: str, start_s: float, record_s: float) -> None:
    import msgspec
    import requests

    from gate import config, hostwatch, runner, server
    from workload import schema

    tok, sessions = runner.load_tokenizer(), schema.read(config.SESSIONS_PATH)
    ref = server.load_ref(ref_name)
    srv_obj = server.Server(ref, os.path.join(out_dir, "server.log"), extra_args=runner.server_extra_args(ref),
                            extra_env={"SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR": out_dir})
    with hostwatch.host_lock(), srv_obj as srv:
        runner.warm_up(srv, sessions, tok, "k3-route")
        for c in points:
            load = msgspec.structs.replace(config.INFLIGHT, concurrency=c, name=f"inflight-C{c}-k3route",
                                           warmup_s=60.0, window_s=start_s + record_s)
            srv.flush_cache()
            t = threading.Thread(target=lambda: runner.replay_leg(srv, sessions, tok, load, "k3-route", "start"))
            t0 = time.time()
            t.start()
            time.sleep(max(0.0, t0 + 60.0 + start_s - time.time()))
            requests.post(f"{srv.url}/start_expert_distribution_record", timeout=60).raise_for_status()
            time.sleep(record_s)
            before = set(glob.glob(os.path.join(out_dir, "*.pt")))
            requests.post(f"{srv.url}/dump_expert_distribution_record", timeout=600).raise_for_status()
            t.join()
            new = sorted(set(glob.glob(os.path.join(out_dir, "*.pt"))) - before)
            for p in new:
                os.rename(p, os.path.join(out_dir, f"C{c}-{os.path.basename(p)}"))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--ref", default="final-mem-c1-c2a-cp2048-lpm-glue-c1-route")
    s.add_argument("--concurrency", default="12,28")
    s.add_argument("--start-s", type=float, default=20.0, help="seconds into the timed window before recording")
    s.add_argument("--record-s", type=float, default=40.0)
    m = sub.add_parser("summarize")
    m.add_argument("dumps", nargs="+")
    args = p.parse_args()
    if args.cmd == "serve":
        from gate import config, runner

        out_dir = os.path.join(config.RUNS_DIR, runner.new_exp_id(f"k3-route-{args.ref}"))
        os.makedirs(out_dir)
        print(f"run dir: {out_dir}", file=sys.stderr, flush=True)
        capture(args.ref, [int(x) for x in args.concurrency.split(",")], out_dir, args.start_s, args.record_s)
        dumps = {c: sorted(glob.glob(os.path.join(out_dir, f"C{c}-*.pt"))) for c in args.concurrency.split(",")}
    else:
        dumps = {"all": args.dumps}
    res = {str(k): summarize(decode_passes(load_records(v))) for k, v in dumps.items() if v}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
