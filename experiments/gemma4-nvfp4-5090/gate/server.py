"""Refs, source trees and server lifetimes.

A ref (gate/refs.json) names everything that defines a configuration: the
SGLang commit, the python interpreter, server flags and env. Each ref commit is
checked out once under trees/<sha> and put first on PYTHONPATH, ahead of the
venv's editable install, so control and candidate run different code from one
venv. A ref that needs other compiled packages names its own `python`.
"""

import json
import logging
import os
import re
import signal
import subprocess
import time
from typing import Dict, List, Optional

import requests

from gate import config

log = logging.getLogger(__name__)

REFS_PATH = os.path.join(os.path.dirname(__file__), "refs.json")
_ALLOWED_KEYS = {"commit", "python", "server_args", "env", "extends", "description", "weight_layout_change"}


def load_ref(name: str) -> Dict:
    with open(REFS_PATH) as f:
        refs = json.load(f)
    if name not in refs:
        raise KeyError(f"unknown ref {name!r}; known: {sorted(refs)}")
    ref = dict(refs[name])
    unknown = set(ref) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(f"ref {name}: unknown keys {unknown}")
    if "extends" in ref:
        parent = load_ref(ref.pop("extends"))
        merged = dict(parent)
        merged.update({k: v for k, v in ref.items() if k not in ("server_args", "env")})
        merged["server_args"] = parent.get("server_args", []) + ref.get("server_args", [])
        merged["env"] = {**parent.get("env", {}), **ref.get("env", {})}
        ref = merged
    ref["name"] = name
    ref.setdefault("python", config.VENV_PYTHON)
    ref.setdefault("env", {})
    ref.setdefault("weight_layout_change", False)
    return ref


def _git(*args: str, cwd: str = config.SRC_REPO) -> str:
    return subprocess.run(["git", "-C", cwd, *args], check=True, capture_output=True, text=True).stdout.strip()


def resolve_commit(commit: str) -> str:
    try:
        return _git("rev-parse", "--verify", f"{commit}^{{commit}}")
    except subprocess.CalledProcessError:
        _git("fetch", "origin", commit)
        return _git("rev-parse", "--verify", f"{commit}^{{commit}}")


def tree_for(commit: str) -> str:
    """A pristine detached worktree of `commit` (created once, verified clean every use)."""
    sha = resolve_commit(commit)
    path = os.path.join(config.TREES_DIR, sha[:12])
    if not os.path.exists(path):
        os.makedirs(config.TREES_DIR, exist_ok=True)
        _git("worktree", "add", "--detach", path, sha)
    if _git("rev-parse", "HEAD", cwd=path) != sha or _git("status", "--porcelain", cwd=path):
        raise RuntimeError(f"tree {path} is not a clean checkout of {sha}")
    return path


def harness_commit() -> Dict:
    here = os.path.dirname(os.path.abspath(__file__))
    sha = _git("rev-parse", "HEAD", cwd=here)
    dirty = bool(_git("status", "--porcelain", "--", ".", cwd=here))
    return {"commit": sha, "dirty": dirty}


class Server:
    """One server lifetime for one ref, logging to `log_path`."""

    def __init__(self, ref: Dict, log_path: str, extra_args: Optional[List[str]] = None):
        self.ref = ref
        self.log_path = log_path
        self.extra_args = extra_args or []
        self.url = f"http://127.0.0.1:{config.PORT}"
        self.proc: Optional[subprocess.Popen] = None
        self.tree = tree_for(ref["commit"])
        self.commit = resolve_commit(ref["commit"])

    def command(self) -> List[str]:
        return [
            self.ref["python"], "-m", "sglang.launch_server",
            "--model-path", config.MODEL_DIR,
            "--port", str(config.PORT),
            # Every decode step is logged with its cuda-graph flag; the gate
            # refuses a leg whose timed window ran a step outside a graph.
            "--decode-log-interval", "1",
            *self.ref.get("server_args", []),
            *self.extra_args,
        ]

    def start(self, timeout_s: int = 900) -> None:
        env = dict(os.environ)
        env.update(self.ref["env"])
        env["PYTHONPATH"] = os.path.join(self.tree, "python") + os.pathsep + env.get("PYTHONPATH", "")
        self._log_file = open(self.log_path, "w")
        self.proc = subprocess.Popen(
            self.command(), stdout=self._log_file, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited with {self.proc.returncode}; see {self.log_path}")
            try:
                if requests.get(f"{self.url}/health_generate", timeout=5).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(3)
        raise TimeoutError(f"server not healthy after {timeout_s}s; see {self.log_path}")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        if getattr(self, "_log_file", None):
            self._log_file.close()

    def __enter__(self):
        try:
            self.start()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *exc):
        self.stop()

    def flush_cache(self) -> None:
        requests.post(f"{self.url}/flush_cache", timeout=60).raise_for_status()

    def weight_checksum(self) -> Dict:
        """sha256 over every loaded parameter (dequantized), via /weights_checker."""
        try:
            r = requests.post(f"{self.url}/weights_checker", json={"action": "checksum"}, timeout=600)
            body = r.json()
        except (requests.RequestException, ValueError) as e:
            return {"ok": False, "error": repr(e)}
        if r.status_code != 200 or not body.get("success"):
            return {"ok": False, "error": body.get("message", r.text[:500])}
        per = body.get("per_engine_checksum") or []
        digest = per[0].get("per_gpu_checksum") if per and isinstance(per[0], dict) else json.dumps(per)
        return {"ok": True, "checksum": digest, "n_tensors": len(per[0].get("checksums", {})) if per else None}

    def server_info(self) -> Dict:
        return requests.get(f"{self.url}/get_server_info", timeout=30).json()

    def log_offset(self) -> int:
        self._log_file.flush()
        return os.path.getsize(self.log_path)

    def decode_steps_since(self, offset: int) -> Dict:
        """Counts decode-step log lines after `offset`, split by cuda-graph use."""
        with open(self.log_path, errors="replace") as f:
            f.seek(offset)
            text = f.read()
        graph = len(re.findall(r"Decode batch.*cuda graph: True", text))
        eager = len(re.findall(r"Decode batch.*cuda graph: False", text))
        jit = re.findall(r".*(?:Compiling|compiling|JIT|autotun).*", text)
        return {"graph_steps": graph, "eager_steps": eager, "compile_lines": jit[:20]}

    def backend_report(self) -> Dict:
        """Which kernels serve MoE, dense GEMM, attention and KV, from the server's own log."""
        with open(self.log_path, errors="replace") as f:
            text = f.read()
        lines = [
            ln[:400]
            for ln in text.split("\n")
            if re.search(r"(?i)backend|KV Cache is allocated|quant|moe runner|fp4|cutlass|marlin|triton attention", ln)
            and "server_args=" not in ln
        ]
        args = re.search(r"server_args=(\{.*\})", text)
        picked = {}
        if args:
            for key in ("attention_backend", "moe_runner_backend", "fp4_gemm_runner_backend", "bf16_gemm_backend",
                        "fp8_gemm_runner_backend", "sampling_backend", "kv_cache_dtype", "quantization",
                        "chunked_prefill_size", "cuda_graph_config", "mem_fraction_static"):
                m = re.search(rf"'{key}': ([^,]*(?:\{{.*?\}}\}})?)", args.group(1))
                if m:
                    picked[key] = m.group(1)
        return {"server_args": picked, "log_lines": lines[:60]}
