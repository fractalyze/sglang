"""Step-0 capacity of one ref: KV pool sizes, then the most simultaneous sessions without retraction.

One server lifetime (gate.server: host lock, memory-capped scope, watchdog). The pool sizes
and memory figures come from the server log; then bursts of N simultaneous requests
(random token ids, so no prefix reuse; exactly `output_len` tokens each) run in increasing
N. A burst is clean when the decode batch reached N running requests with no retraction.
The ref's capacity is the largest clean N. GPU memory (nvidia-smi) is sampled during every
burst, so an activation peak close to the card's limit shows before it turns into an OOM.

  python -m memory.capacity --ref mem-base --n 14,16,18,20 [--input-len 5000] [--long 10000x4]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import requests

from baseline import probe
from gate import config, fidelity, hostwatch, runner, server

# Log lines that size the pools (model_runner / memory pool logging of this tree).
POOL_PATTERNS = {
    "weights_gib": r"Load weight end\..*?mem usage=([\d.]+) GB",
    "avail_after_weights_gib": r"Load weight end\..*?avail mem=([\d.]+) GB",
    "full_tokens": r"Full KV Cache is allocated\..*?#tokens: (\d+)",
    "full_k_gib": r"Full KV Cache is allocated\..*?K size: ([\d.]+) GB",
    "full_v_gib": r"Full KV Cache is allocated\..*?V size: ([\d.]+) GB",
    "swa_tokens": r"SWA KV Cache is allocated\..*?#tokens: (\d+)",
    "swa_k_gib": r"SWA KV Cache is allocated\..*?K size: ([\d.]+) GB",
    "swa_v_gib": r"SWA KV Cache is allocated\..*?V size: ([\d.]+) GB",
    "avail_after_pool_gib": r"Memory pool end\. avail mem=([\d.]+) GB",
    "decode_graph_gib": r"Capture target decode CUDA graph end\..*?mem usage=([\d.]+) GB",
    "max_running_requests": r"max_running_requests=(\d+)",
    "chunked_prefill_size": r"chunked_prefill_size=(\d+)",
    "available_gpu_mem_gib": r"available_gpu_mem=([\d.]+) GB",
}


def pool_facts(log_text: str) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for key, pat in POOL_PATTERNS.items():
        m = re.search(pat, log_text)
        out[key] = (float(m.group(1)) if "." in m.group(1) else int(m.group(1))) if m else None
    return out


def is_clean(n: int, scan: Dict) -> bool:
    return scan["peak_running"] >= n and scan["retract_lines"] == 0


class GpuSampler:
    """Peak `memory.used` (MiB) of GPU 0 while the context is open."""

    def __init__(self, period_s: float = 0.25):
        self.period_s, self.peak_mib, self._stop = period_s, 0, threading.Event()

    def _run(self) -> None:
        while not self._stop.is_set():
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                                  "-i", "0"], capture_output=True, text=True).stdout.strip()
            if out:
                self.peak_mib = max(self.peak_mib, int(out))
            self._stop.wait(self.period_s)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def burst(srv: server.Server, n: int, input_len: int, output_len: int) -> Dict:
    requests.post(f"{srv.url}/flush_cache", timeout=60)
    offset = srv.log_offset()
    t0 = time.time()
    with GpuSampler() as gpu, ThreadPoolExecutor(n) as ex:
        res = list(ex.map(lambda i: probe.one(srv.url, input_len, output_len, 1000 * n + i + input_len), range(n)))
    wall = time.time() - t0
    time.sleep(2)
    lat = sorted(r["latency_s"] for r in res)
    scan = probe.scan_log(srv.log_path, offset)
    return {"n": n, "input_len": input_len, "output_len": output_len, "wall_s": wall,
            "output_tok_per_s": sum(r["completion_tokens"] for r in res) / wall,
            "latency_p50_s": lat[len(lat) // 2], "latency_max_s": lat[-1], "gpu_peak_mib": gpu.peak_mib,
            "clean": is_clean(n, scan), **scan}


def _parse_long(spec: str) -> List[tuple]:
    return [tuple(int(x) for x in item.split("x")) for item in spec.split(",") if item]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--n", required=True, help="comma list of burst sizes, run in increasing order")
    ap.add_argument("--input-len", type=int, default=5000)
    ap.add_argument("--output-len", type=int, default=300)
    ap.add_argument("--long", default="", help="extra bursts INPUTxN run first, e.g. 10000x4 (activation peak)")
    ap.add_argument("--stop-after-dirty", type=int, default=2, help="stop after this many unclean bursts in a row")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    ref = server.load_ref(a.ref)
    out_dir = os.path.join(config.RUNS_DIR, runner.new_exp_id(f"cap-{a.ref}{a.label}"))
    os.makedirs(out_dir)
    res: Dict = {"ref": ref, "bursts": [], "long_bursts": []}
    with hostwatch.host_lock(), server.Server(ref, os.path.join(out_dir, "server.log"),
                                              extra_args=runner.server_extra_args(ref)) as srv:
        with open(srv.log_path, errors="replace") as f:
            res["pool"] = pool_facts(f.read())
        res["commit"] = srv.commit
        print(json.dumps(res["pool"]), file=sys.stderr, flush=True)
        for input_len, n in _parse_long(a.long):
            b = burst(srv, n, input_len, a.output_len)
            res["long_bursts"].append(b)
            print(json.dumps({k: b[k] for k in ("n", "input_len", "clean", "peak_running", "gpu_peak_mib")}),
                  file=sys.stderr, flush=True)
        dirty = 0
        for n in sorted(int(x) for x in a.n.split(",")):
            b = burst(srv, n, a.input_len, a.output_len)
            res["bursts"].append(b)
            print(json.dumps({k: b[k] for k in ("n", "clean", "peak_running", "retract_lines", "peak_full_usage",
                                                "peak_swa_usage", "peak_swa_tokens", "gpu_peak_mib",
                                                "latency_max_s")}), file=sys.stderr, flush=True)
            dirty = 0 if b["clean"] else dirty + 1
            if dirty >= a.stop_after_dirty:
                break
    res["host"] = srv.host_summary
    res["max_clean_sessions"] = max((b["n"] for b in res["bursts"] if b["clean"]), default=0)
    fidelity.save_json(os.path.join(out_dir, "capacity.json"), res)
    print(json.dumps({"max_clean_sessions": res["max_clean_sessions"], "pool": res["pool"]}, indent=1))
    print(f"run dir: {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
