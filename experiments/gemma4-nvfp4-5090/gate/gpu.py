"""GPU quiescence gate and in-window telemetry (nvidia-smi only, no NVML binding)."""

import csv
import io
import logging
import subprocess
import threading
import time
from typing import Dict, List, Optional, Set

from gate import config

log = logging.getLogger(__name__)

_GPU_FIELDS = (
    "timestamp",
    "temperature.gpu",
    "clocks.sm",
    "clocks.mem",
    "power.draw",
    "utilization.gpu",
    "memory.used",
    "clocks_event_reasons.active",
)
# Clock-event reasons that mean the clock was pulled down by heat or hardware,
# not by the normal power/boost governor.
_THROTTLE_BITS = {
    0x0000000000000008: "hw_slowdown",
    0x0000000000000020: "sw_thermal",
    0x0000000000000040: "hw_thermal",
    0x0000000000000080: "hw_power_brake",
}


def _smi(args: List[str]) -> str:
    return subprocess.run(["nvidia-smi", *args], check=True, capture_output=True, text=True).stdout


def gpu_state() -> Dict:
    out = _smi([f"--query-gpu={','.join(_GPU_FIELDS)}", "--format=csv,noheader,nounits"])
    row = next(csv.reader(io.StringIO(out)))
    state = dict(zip(_GPU_FIELDS, (v.strip() for v in row)))
    state["temperature.gpu"] = int(state["temperature.gpu"])
    state["utilization.gpu"] = int(state["utilization.gpu"])
    return state


def compute_processes() -> List[Dict]:
    out = _smi(["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    procs = []
    for row in csv.reader(io.StringIO(out)):
        if not row:
            continue
        pid, name, mem = (v.strip() for v in row)
        procs.append({"pid": int(pid), "name": name, "used_mib": mem})
    return procs


def descendants(pid: int) -> Set[int]:
    """The pid and every process below it (sglang spawns scheduler/detokenizer children)."""
    out = subprocess.run(["ps", "-e", "-o", "pid=,ppid="], check=True, capture_output=True, text=True).stdout
    children: Dict[int, List[int]] = {}
    for line in out.split("\n"):
        if line.strip():
            p, pp = (int(x) for x in line.split())
            children.setdefault(pp, []).append(p)
    seen, stack = set(), [pid]
    while stack:
        p = stack.pop()
        if p not in seen:
            seen.add(p)
            stack.extend(children.get(p, []))
    return seen


def foreign_processes(own_root_pid: Optional[int]) -> List[Dict]:
    own = descendants(own_root_pid) if own_root_pid else set()
    return [p for p in compute_processes() if p["pid"] not in own]


def wait_quiet(own_root_pid: Optional[int], events: List[Dict], max_temp_c: int = config.MAX_START_TEMP_C) -> Dict:
    """Blocks until no foreign compute process runs and the GPU is cool; logs every wait.

    Raises TimeoutError after QUIESCE_TIMEOUT_S so a busy box fails the run
    instead of timing under contention.
    """
    deadline = time.time() + config.QUIESCE_TIMEOUT_S
    while True:
        foreign = foreign_processes(own_root_pid)
        state = gpu_state()
        quiet = not foreign and state["temperature.gpu"] <= max_temp_c
        if quiet:
            return state
        events.append({"t": time.time(), "event": "not_quiet", "foreign": foreign, "gpu": state})
        log.info("waiting for quiet GPU: foreign=%s temp=%s", foreign, state["temperature.gpu"])
        if time.time() > deadline:
            raise TimeoutError(f"GPU not quiet after {config.QUIESCE_TIMEOUT_S}s: {foreign} {state}")
        time.sleep(config.QUIESCE_POLL_S)


class Telemetry:
    """Samples nvidia-smi every 200 ms during a timed window."""

    def __init__(self, own_root_pid: int):
        self._own = own_root_pid
        self._samples: List[Dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.foreign_seen: List[Dict] = []

    def _run(self):
        n = 0
        while not self._stop.is_set():
            try:
                s = gpu_state()
                s["t"] = time.time()
                self._samples.append(s)
                if n % 5 == 0:
                    foreign = foreign_processes(self._own)
                    if foreign:
                        self.foreign_seen.append({"t": time.time(), "foreign": foreign})
            except subprocess.CalledProcessError as e:
                self._samples.append({"t": time.time(), "error": str(e)})
            n += 1
            self._stop.wait(0.2)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def summary(self) -> Dict:
        ok = [s for s in self._samples if "error" not in s]
        throttles: Dict[str, int] = {}
        for s in ok:
            bits = int(s["clocks_event_reasons.active"], 16)
            for bit, name in _THROTTLE_BITS.items():
                if bits & bit:
                    throttles[name] = throttles.get(name, 0) + 1

        def stat(key):
            vals = [float(s[key]) for s in ok]
            return {"min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals)} if vals else None

        return {
            "n_samples": len(ok),
            "temp_c": stat("temperature.gpu"),
            "sm_clock_mhz": stat("clocks.sm"),
            "mem_clock_mhz": stat("clocks.mem"),
            "power_w": stat("power.draw"),
            "thermal_or_hw_throttle_samples": throttles,
            "foreign_seen": self.foreign_seen,
        }
