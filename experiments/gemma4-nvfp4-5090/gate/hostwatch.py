"""Host-safety protocol for every engine process (SGLang, vLLM, JIT prebuild).

Two shared hosts were OOM-killed on 2026-10-02 by uncapped launches (FlashInfer
ran ninja with nproc+2 parallel nvcc jobs on CUTLASS FP4 MoE units). So every
engine process: holds host.lock, runs in a memory-capped systemd scope without
swap, refuses to start on a busy host, and has a watchdog beside it that logs
host memory to hostmem.csv every 2 s and kills the process group past the
limits. Usable in-process (Watchdog) or as a wrapper:

  python -m gate.hostwatch --csv runs/x/hostmem.csv --log runs/x/engine.log -- <cmd...>
"""

import argparse
import csv
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

from gate import config

_COMPILERS = ("nvcc", "cicc", "ptxas", "cudafe++", "nvlink", "fatbinary", "c++", "cc1plus", "g++")
# Server log lines that open each phase; the first match wins in order.
SGLANG_PHASES: Tuple[Tuple[str, str], ...] = (
    ("weight_load", r"Load weight begin"),
    ("autotune", r"Running FlashInfer autotune"),
    ("graph_capture", r"Capture cuda graph begin|Capture .*cuda graph"),
    ("serving", r"The server is fired up|Uvicorn running"),
)


class HostUnsafe(RuntimeError):
    pass


def meminfo_gb() -> Dict[str, float]:
    vals = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            vals[k] = int(v.split()[0]) / 1024**2
    return {"mem_available_gb": vals["MemAvailable"], "swap_used_gb": vals["SwapTotal"] - vals["SwapFree"]}


def load1() -> float:
    with open("/proc/loadavg") as f:
        return float(f.read().split()[0])


def _process_table() -> List[Tuple[int, int, int, str]]:
    """(pid, ppid, rss_kb, comm) for every process."""
    out = subprocess.run(["ps", "-e", "-o", "pid=,ppid=,rss=,comm="], check=True, capture_output=True,
                         text=True).stdout
    rows = []
    for line in out.split("\n"):
        parts = line.split(None, 3)
        if len(parts) == 4:
            rows.append((int(parts[0]), int(parts[1]), int(parts[2]), parts[3].strip()))
    return rows


def tree_stats(root_pid: int) -> Dict:
    rows = _process_table()
    children: Dict[int, List[int]] = {}
    for pid, ppid, _, _ in rows:
        children.setdefault(ppid, []).append(pid)
    tree, stack = set(), [root_pid]
    while stack:
        p = stack.pop()
        if p not in tree:
            tree.add(p)
            stack.extend(children.get(p, []))
    in_tree = [r for r in rows if r[0] in tree]
    compilers = [r for r in in_tree if r[3] in _COMPILERS]
    return {
        "tree_rss_gb": sum(r[2] for r in in_tree) / 1024**2,
        "n_compilers": len(compilers),
        "compiler_rss_gb": sum(r[2] for r in compilers) / 1024**2,
        "max_compiler_rss_gb": max((r[2] for r in compilers), default=0) / 1024**2,
    }


def foreign_gpu_gb(own_root_pid: Optional[int] = None) -> float:
    from gate import gpu

    return sum(float(p["used_mib"]) / 1024 for p in gpu.foreign_processes(own_root_pid))


def preflight() -> Dict:
    """Refuses to start an engine on a host that is short of RAM, swapping or GPU-shared."""
    m = meminfo_gb()
    foreign = foreign_gpu_gb()
    problems = []
    if m["mem_available_gb"] < config.MIN_HOST_AVAILABLE_GB:
        problems.append(f"MemAvailable {m['mem_available_gb']:.1f} GB < {config.MIN_HOST_AVAILABLE_GB}")
    if m["swap_used_gb"] > config.MAX_SWAP_USED_GB:
        problems.append(f"swap used {m['swap_used_gb']:.1f} GB > {config.MAX_SWAP_USED_GB}")
    if foreign > config.MAX_FOREIGN_GPU_GB:
        problems.append(f"other users hold {foreign:.1f} GB of GPU memory > {config.MAX_FOREIGN_GPU_GB}")
    if problems:
        raise HostUnsafe("; ".join(problems))
    return {**m, "load1": load1(), "foreign_gpu_gb": foreign}


def memory_cap_prefix() -> List[str]:
    if shutil.which("systemd-run") is None:
        raise HostUnsafe("systemd-run not found; refusing to launch an uncapped engine")
    return ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={config.SERVER_MEMORY_MAX}",
            "-p", "MemorySwapMax=0"]


@contextmanager
def host_lock():
    """host.lock serializes every engine process on the host; gpu.lock is kept for older callers."""
    os.makedirs(os.path.dirname(config.HOST_LOCK), exist_ok=True)
    with open(config.HOST_LOCK, "a") as hf, open(config.GPU_LOCK, "a") as gf:
        fcntl.flock(hf, fcntl.LOCK_EX)
        fcntl.flock(gf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(gf, fcntl.LOCK_UN)
            fcntl.flock(hf, fcntl.LOCK_UN)


class Watchdog:
    """Samples the host every 2 s into a CSV, tracks per-phase peaks, kills the group past the limits."""

    _FIELDS = ("t", "phase", "mem_available_gb", "swap_used_gb", "load1", "tree_rss_gb", "n_compilers",
               "compiler_rss_gb", "max_compiler_rss_gb")

    def __init__(self, root_pid: int, csv_path: str, log_path: Optional[str] = None,
                 phases: Sequence[Tuple[str, str]] = SGLANG_PHASES, interval_s: float = 2.0):
        self._root = root_pid
        self._csv_path = csv_path
        self._log_path = log_path
        self._phases = [(name, re.compile(rx)) for name, rx in phases]
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._log_offset = 0
        self.phase = "start"
        self.peaks: Dict[str, Dict[str, float]] = {}
        self.tripped: Optional[str] = None

    def set_phase(self, name: str) -> None:
        self.phase = name

    def _advance_phase_from_log(self) -> None:
        if not self._log_path or not os.path.exists(self._log_path):
            return
        with open(self._log_path, errors="replace") as f:
            f.seek(self._log_offset)
            text = f.read()
            self._log_offset = f.tell()
        for name, rx in self._phases:
            if rx.search(text):
                self.phase = name

    def _sample(self) -> Dict:
        row = {"t": round(time.time(), 1), **meminfo_gb(), "load1": load1(), **tree_stats(self._root)}
        self._advance_phase_from_log()
        row["phase"] = self.phase
        peak = self.peaks.setdefault(self.phase, {"min_mem_available_gb": 1e9})
        peak["min_mem_available_gb"] = min(peak["min_mem_available_gb"], row["mem_available_gb"])
        for k in ("tree_rss_gb", "load1", "swap_used_gb", "n_compilers", "compiler_rss_gb", "max_compiler_rss_gb"):
            peak[f"peak_{k}"] = max(peak.get(f"peak_{k}", 0.0), row[k])
        return row

    def _kill(self, reason: str) -> None:
        self.tripped = reason
        try:
            os.killpg(os.getpgid(self._root), signal.SIGKILL)
        except ProcessLookupError:
            pass

    def _run(self) -> None:
        os.makedirs(os.path.dirname(self._csv_path), exist_ok=True)
        with open(self._csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._FIELDS)
            if f.tell() == 0:
                w.writeheader()
            while not self._stop.is_set():
                try:
                    row = self._sample()
                except (subprocess.CalledProcessError, OSError):
                    self._stop.wait(self._interval)
                    continue
                w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
                f.flush()
                if row["mem_available_gb"] < config.KILL_MEM_AVAILABLE_GB:
                    self._kill(f"MemAvailable {row['mem_available_gb']:.1f} GB < {config.KILL_MEM_AVAILABLE_GB}")
                elif row["load1"] > config.KILL_LOAD1:
                    self._kill(f"load1 {row['load1']:.0f} > {config.KILL_LOAD1}")
                self._stop.wait(self._interval)

    def start(self) -> "Watchdog":
        self._thread.start()
        return self

    def stop(self) -> Dict:
        self._stop.set()
        self._thread.join()
        return {"peaks_by_phase": self.peaks, "tripped": self.tripped, "csv": self._csv_path}


def run_wrapped(cmd: List[str], csv_path: str, log_path: str, env: Optional[Dict[str, str]] = None,
                timeout_s: Optional[float] = None) -> Dict:
    """Preflight, cap, watch and run `cmd` to completion (caller holds host.lock)."""
    pre = preflight()
    with open(log_path, "a") as log:
        proc = subprocess.Popen([*memory_cap_prefix(), *cmd], stdout=log, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        dog = Watchdog(proc.pid, csv_path, log_path).start()
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            rc = proc.wait()
        summary = dog.stop()
    return {"returncode": rc, "preflight": pre, **summary}


def main() -> None:
    p = argparse.ArgumentParser(prog="gate.hostwatch")
    p.add_argument("--csv", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--timeout-s", type=float, default=None)
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    a = p.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    with host_lock():
        res = run_wrapped(cmd, a.csv, a.log, timeout_s=a.timeout_s)
    with open(os.path.splitext(a.csv)[0] + ".summary.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    sys.exit(res["returncode"])


if __name__ == "__main__":
    main()
