"""Refs, source trees and server lifetimes.

A ref (gate/refs.json) names everything that defines a configuration: the
SGLang commit, the python interpreter, server flags and env. Each ref commit is
checked out once under trees/<sha> and put first on PYTHONPATH, ahead of the
venv's editable install, so control and candidate run different code from one
venv. A ref that needs other compiled packages names its own `python`.
"""

import glob
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import time
from typing import Dict, List, Optional

import requests

from gate import config, hostwatch, treehash

log = logging.getLogger(__name__)

REFS_PATH = os.path.join(os.path.dirname(__file__), "refs.json")
# Workstreams keep their refs beside their own code (e.g. memory/refs.json) so they never
# edit the gate's file; a ref name must be unique across all of them.
WORKSTREAM_REFS_GLOB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "*", "refs.json")
_ALLOWED_KEYS = {"commit", "python", "server_args", "env", "extends", "description", "weight_layout_change",
                 "numerics_unchanged"}


def all_refs(paths: Optional[List[str]] = None) -> Dict[str, Dict]:
    if paths is None:
        paths = sorted({os.path.abspath(REFS_PATH), *map(os.path.abspath, glob.glob(WORKSTREAM_REFS_GLOB))})
    refs: Dict[str, Dict] = {}
    for path in paths:
        with open(path) as f:
            for name, ref in json.load(f).items():
                if name in refs:
                    raise ValueError(f"ref {name!r} is defined twice (again in {path})")
                refs[name] = ref
    return refs


def load_ref(name: str) -> Dict:
    refs = all_refs()
    if name not in refs:
        raise KeyError(f"unknown ref {name!r}; known: {sorted(refs)}")
    ref = dict(refs[name])
    unknown = set(ref) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(f"ref {name}: unknown keys {unknown}")
    if "extends" in ref:
        parent = load_ref(ref.pop("extends"))
        # numerics_unchanged states a ref's own delta to its control, so it is never inherited.
        merged = {k: v for k, v in parent.items() if k != "numerics_unchanged"}
        merged.update({k: v for k, v in ref.items() if k not in ("server_args", "env")})
        merged["server_args"] = parent.get("server_args", []) + ref.get("server_args", [])
        merged["env"] = {**parent.get("env", {}), **ref.get("env", {})}
        ref = merged
    ref["name"] = name
    ref.setdefault("python", config.VENV_PYTHON)
    ref.setdefault("env", {})
    ref.setdefault("weight_layout_change", False)
    ref.setdefault("numerics_unchanged", False)
    return ref


def _git(*args: str, cwd: str = config.SRC_REPO) -> str:
    return subprocess.run(["git", "-C", cwd, *args], check=True, capture_output=True, text=True).stdout.strip()


def resolve_commit(commit: str) -> str:
    try:
        return _git("rev-parse", "--verify", f"{commit}^{{commit}}")
    except subprocess.CalledProcessError:
        _git("fetch", "origin", commit)
        return _git("rev-parse", "--verify", f"{commit}^{{commit}}")


def tree_for(commit: str, overlay: Optional[str] = None) -> str:
    """A pristine detached worktree of `commit` (created once, verified clean every use).

    With `overlay` (a patch file), the tree is `commit` plus that patch staged, kept in its
    own directory named by the patch's hash; every use checks that the staged diff is still
    exactly what applying the patch produced and that nothing else changed.
    """
    sha = resolve_commit(commit)
    if overlay is None:
        path = os.path.join(config.TREES_DIR, sha[:12])
    else:
        path = os.path.join(config.TREES_DIR, f"{sha[:12]}+{overlay_digest(overlay)}")
    if not os.path.exists(path):
        os.makedirs(config.TREES_DIR, exist_ok=True)
        _git("worktree", "add", "--detach", path, sha)
        if overlay is not None:
            _git("apply", "--index", os.path.abspath(overlay), cwd=path)
            with open(path + ".staged.sha256", "w") as f:
                f.write(_staged_digest(path))
    if _git("rev-parse", "HEAD", cwd=path) != sha:
        raise RuntimeError(f"tree {path} is not at {sha}")
    # Porcelain lines are "XY path"; an overlay tree may differ from HEAD only in the index (Y blank).
    status = subprocess.run(["git", "-C", path, "status", "--porcelain"], check=True, capture_output=True,
                            text=True).stdout.splitlines()
    if [ln for ln in status if overlay is None or ln[1] != " "]:
        raise RuntimeError(f"tree {path} is not a clean checkout of {sha}")
    if overlay is not None:
        with open(path + ".staged.sha256") as f:
            if f.read() != _staged_digest(path):
                raise RuntimeError(f"tree {path}: staged overlay changed since it was applied")
    return path


def overlay_digest(patch: str) -> str:
    with open(patch, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def _staged_digest(path: str) -> str:
    diff = subprocess.run(["git", "-C", path, "diff", "--cached", "--binary", "HEAD"], check=True,
                          capture_output=True).stdout
    return hashlib.sha256(diff).hexdigest()


def harness_commit(here: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) -> Dict:
    """The experiments commit that is actually on disk at `here` (the experiments directory).

    `commit` comes from git when `here` is tracked and clean, else from gate/deploy.sh's stamp when the
    tree still hashes to the stamped value, else it is None and only `tree_sha256` identifies the
    harness. `src_head` is the enclosing checkout's HEAD, which an rsync deploy leaves unrelated.
    """
    tree = treehash.tree_sha256(here)
    out = {"commit": None, "source": "tree_sha256", "tree_sha256": tree, "src_head": None, "dirty": True}
    try:
        out["src_head"] = _git("rev-parse", "HEAD", cwd=here)
        tracked = bool(_git("ls-files", "--", ".", cwd=here))
        clean = not _git("status", "--porcelain", "--ignored=no", "--", ".", cwd=here)
    except (subprocess.CalledProcessError, OSError):
        tracked = clean = False
    if tracked and clean:
        out.update(commit=out["src_head"], source="git", dirty=False)
        return out
    stamp_path = os.path.join(here, treehash.STAMP)
    if os.path.exists(stamp_path):
        with open(stamp_path) as f:
            stamp = json.load(f)
        out["stamp"] = stamp
        if stamp.get("tree_sha256") == tree:
            out.update(commit=stamp["commit"], source="deploy_stamp", dirty=False)
    return out


def parse_weights_checksum(body: Dict) -> Dict:
    """A successful `/weights_checker` checksum body -> the engine digest and the tensor count.

    `per_engine_checksum` is the sha256 over every rank's per-GPU digest; `ranks` holds each
    rank's per-tensor checksums.
    """
    digest, ranks = body.get("per_engine_checksum"), body.get("ranks") or []
    if not isinstance(digest, str):
        return {"ok": False, "error": f"no per_engine_checksum string in {sorted(body)}"}
    return {"ok": True, "checksum": digest, "n_tensors": len(ranks[0].get("checksums", {})) if ranks else None}


class Server:
    """One server lifetime for one ref, logging to `log_path`."""

    def __init__(self, ref: Dict, log_path: str, extra_args: Optional[List[str]] = None,
                 overlay: Optional[str] = None, extra_env: Optional[Dict[str, str]] = None):
        self.ref = ref
        self.log_path = log_path
        self.extra_args = extra_args or []
        # Measurement-only additions (gate spec-run's replay hook): never part of a ref.
        self.overlay = overlay
        self.extra_env = extra_env or {}
        self.url = f"http://127.0.0.1:{config.PORT}"
        self.proc: Optional[subprocess.Popen] = None
        self._watchdog: Optional[hostwatch.Watchdog] = None
        self._log_file = None
        self.preflight: Optional[Dict] = None
        self.host_summary: Optional[Dict] = None
        self.tree = tree_for(ref["commit"], overlay)
        self.commit = resolve_commit(ref["commit"])

    def command(self) -> List[str]:
        return [*hostwatch.memory_cap_prefix(), *self.server_command()]

    def server_command(self) -> List[str]:
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

    def launch_env(self, base: Dict[str, str]) -> Dict[str, str]:
        owned = sorted(set(self.ref["env"]) & set(config.SERVER_ENV))
        if owned:
            raise ValueError(f"ref sets gate-owned env {owned} (config.SERVER_ENV)")
        env = dict(base)
        env["MAX_JOBS"] = str(config.JIT_MAX_JOBS)
        env.update(config.SERVER_ENV)
        env.update(self.ref["env"])
        env.update(self.extra_env)
        env["PYTHONPATH"] = os.path.join(self.tree, "python") + os.pathsep + env.get("PYTHONPATH", "")
        return env

    def start(self, timeout_s: int = 900) -> None:
        self.preflight = hostwatch.wait_preflight()
        env = self.launch_env(dict(os.environ))
        self._log_file = open(self.log_path, "w")
        self.proc = subprocess.Popen(
            self.command(), stdout=self._log_file, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        self._watchdog = hostwatch.Watchdog(
            self.proc.pid, os.path.join(os.path.dirname(self.log_path), "hostmem.csv"), self.log_path
        ).start()
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"server exited with {self.proc.returncode} (watchdog: {self._watchdog.tripped}); see {self.log_path}"
                )
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
        if self._watchdog is not None:
            self.host_summary = self._watchdog.stop()
            self._watchdog = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def __enter__(self):
        try:
            self.start()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *exc):
        self.stop()

    def flush_cache(self, timeout_s: float = 60.0) -> int:
        """Flushes once the scheduler is idle; returns the refused attempts.

        A client sees a stream's last token before the scheduler releases the
        request, and SGLang answers 400 while any request is running or queued.
        """
        deadline, delay, refused = time.time() + timeout_s, 0.05, 0
        while True:
            r = requests.post(f"{self.url}/flush_cache", timeout=60)
            if r.status_code == 200:
                return refused
            if r.status_code != 400 or time.time() > deadline:
                raise RuntimeError(f"flush_cache failed after {refused + 1} attempts: {r.status_code} {r.text[:200]}")
            refused += 1
            time.sleep(delay)
            delay = min(delay * 2, 2.0)

    def weight_checksum(self) -> Dict:
        """sha256 over every loaded parameter (dequantized), via /weights_checker."""
        try:
            r = requests.post(f"{self.url}/weights_checker", json={"action": "checksum"}, timeout=600)
            body = r.json()
        except (requests.RequestException, ValueError) as e:
            return {"ok": False, "error": repr(e)}
        if r.status_code != 200 or not body.get("success"):
            return {"ok": False, "error": body.get("message", r.text[:500])}
        return parse_weights_checksum(body)

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
