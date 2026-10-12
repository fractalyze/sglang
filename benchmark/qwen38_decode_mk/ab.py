# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""Qwen3.8-27B at batch 1 on one RTX 5090: stock SGLang's arms beside the
SGLANG_DECODE_MK_* switches, timed and evaluated in one session.

    python ab.py time <arm>... --out-dir <dir> [--repeats 5]
    python ab.py eval <arm> <benchmark> --out-dir <dir> [--num-examples N]

Run it with the Python of a venv holding stock SGLang (fractalyze/sglang
1d59ce7, the commit the stock arms were measured at) and its CUDA 13.0
toolkit wheels: control/qwen38/setup.sh in fractalyze/decode-mk makes one.
The stock arms serve that venv's installed SGLang; ours serve this checkout's
python/ tree, the same commit plus the switches, in the same venv.

`time` serves each arm in turn and sends each prompt --repeats times, one
request at a time, after one warm-up request, flushing the prefix cache before
each. Thinking is off and decoding is greedy, capped at 512 tokens. The
protocol, prompts included, is fractalyze/decode-mk control/qwen38/bench.py
at bc571b0, so its numbers compare with that baseline's.

`eval` serves one arm and scores it with sgl-eval, which it runs from
$SGL_EVAL (default: `sgl-eval` on PATH), one request at a time, greedy, thinking
off.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import signal
import statistics
import subprocess
import sys
import sysconfig
import time
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import msgspec
import requests

PORT = 30000
URL = f"http://127.0.0.1:{PORT}"
MAX_TOKENS = 512
_REPO = Path(__file__).resolve().parents[2]

INT4 = ("cyankiwi/Qwen3.8-27B-AWQ-INT4", "6e134bae811fb5adac50ee042ae5f029ac6779aa")
NVFP4 = ("nvidia/Qwen3.8-27B-NVFP4", "482ca0f3832238542f8f5295dde86b5f22711d80")

# SGLang's cookbook RTX 5090 recipe for Qwen3.8-27B with its panel defaults,
# as control/qwen38/serve.sh runs it: without MTP the static memory fraction
# is 0.85, not the cookbook's 0.9, at which this commit runs out of memory.
_COOKBOOK = (
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--attention-backend",
    "flashinfer",
    "--max-running-requests",
    "1",
    "--cuda-graph-max-bs-decode",
    "1",
    "--mamba-radix-cache-strategy",
    "extra_buffer",
    "--mamba-ssm-dtype",
    "float32",
    "--mamba-full-memory-ratio",
    "4.59",
)
_COOKBOOK_MTP = (
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--enable-linear-replayssm-spec",
    "--mem-fraction-static",
    "0.94",
)
# What python/sglang/kernels/decode_mk/README.md requires of the switches.
_DECODE_MK = (
    "--max-running-requests",
    "1",
    "--disable-radix-cache",
    "--disable-cuda-graph",
    "--context-length",
    "16384",
    "--chunked-prefill-size",
    "-1",
    "--mem-fraction-static",
    "0.92",
)


def _decode_mk_mtp(drafts: int) -> tuple[str, ...]:
    return (
        "--speculative-algorithm",
        "DECODE_MK_MTP",
        "--disable-overlap-schedule",
        "--speculative-num-steps",
        str(drafts),
    )


class Arm(msgspec.Struct, frozen=True):
    checkpoint: tuple[str, str]
    flags: tuple[str, ...]
    # Serve this checkout's python/ tree, not the venv's installed SGLang.
    fork: bool = False
    env: Mapping[str, str] = MappingProxyType({})


_ON = MappingProxyType(
    {"SGLANG_DECODE_MK_DECODE": "1", "SGLANG_DECODE_MK_PREFILL": "1"}
)
_ON_MTP = MappingProxyType({**_ON, "SGLANG_DECODE_MK_MTP": "1"})

ARMS = {
    "int4": Arm(INT4, _COOKBOOK + ("--mem-fraction-static", "0.85")),
    "int4-mtp": Arm(INT4, _COOKBOOK + _COOKBOOK_MTP),
    "nvfp4": Arm(NVFP4, _COOKBOOK + ("--mem-fraction-static", "0.85")),
    "nvfp4-mtp": Arm(NVFP4, _COOKBOOK + _COOKBOOK_MTP),
    "ours": Arm(INT4, _DECODE_MK, fork=True, env=_ON),
    **{
        f"ours-mtp{k}": Arm(
            INT4, _DECODE_MK + _decode_mk_mtp(k), fork=True, env=_ON_MTP
        )
        for k in (1, 2, 3)
    },
}


def snapshot(checkpoint: tuple[str, str]) -> str:
    """The checkpoint's directory in the Hugging Face cache, where it must
    already be: the switches read their weights from a directory, not a
    hub id, so every arm is served from one."""
    from huggingface_hub.constants import HF_HUB_CACHE

    model, revision = checkpoint
    path = (
        Path(HF_HUB_CACHE)
        / f"models--{model.replace('/', '--')}"
        / "snapshots"
        / revision
    )
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"{model} at {revision} is not in {HF_HUB_CACHE}")
    return str(path)


def launch_command(arm: Arm, model_path: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--trust-remote-code",
        "--model-path",
        model_path,
        *arm.flags,
        "--reasoning-parser",
        "qwen3",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
    ]


def launch_env(arm: Arm) -> dict[str, str]:
    """The server's environment: the venv's toolkit and kernel cache, and for
    ours, this checkout ahead of the installed SGLang."""
    env = dict(os.environ)
    for name in list(env):
        if name.startswith("SGLANG_DECODE_MK_"):
            del env[name]
    venv = Path(sys.prefix)
    purelib = Path(sysconfig.get_paths()["purelib"])
    env.update(
        PATH=f"{venv / 'bin'}:{env['PATH']}",
        CUDA_HOME=str(purelib / "nvidia/cu13"),
        SGLANG_CACHE_DIR=str(venv / "cache"),
        **arm.env,
    )
    env.pop("PYTHONPATH", None)
    if arm.fork:
        env["PYTHONPATH"] = str(_REPO / "python")
    return env


def gpu_used_gb() -> float:
    """Memory in use on the GPU by every process, from the driver."""
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; f, t = torch.cuda.mem_get_info(); print((t - f) / 1e9)",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return float(out)


def wait_gpu_idle() -> None:
    for _ in range(100):
        if gpu_used_gb() < 1.0:
            return
        time.sleep(3)
    raise RuntimeError("the GPU still holds memory; another process is using it")


class Server:
    """One arm's server for the length of a `with` block."""

    def __init__(self, arm: Arm, log: Path):
        self.arm, self.log = arm, log

    def __enter__(self) -> Server:
        wait_gpu_idle()
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log, "w")
        command = launch_command(self.arm, snapshot(self.arm.checkpoint))
        self.proc = subprocess.Popen(
            command,
            env=launch_env(self.arm),
            cwd="/",
            stdout=self._log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self.__exit__()
                raise RuntimeError(f"the server exited; see {self.log}")
            try:
                if requests.get(f"{URL}/health", timeout=5).ok:
                    return self
            except requests.ConnectionError:
                pass
            time.sleep(5)
        self.__exit__()
        raise RuntimeError(f"the server was not ready after 30 minutes; see {self.log}")

    def __exit__(self, *exc) -> None:
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        self._log.close()
        wait_gpu_idle()


def _records(n: int) -> str:
    """n lines of synthetic sensor records, the same for every run."""
    rng = random.Random(n)
    sites = ["north pier", "grain silo", "river gauge", "hill station", "rail yard"]
    return "\n".join(
        f"Record {i + 1:05d}: the {rng.choice(sites)} sensor read "
        f"{rng.uniform(-5, 35):.1f} C and {rng.randint(10, 99)}% humidity at "
        f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}."
        for i in range(n)
    )


REPORT = (
    "Write a detailed report on the trends in these sensor records, site by site, "
    "with your reasoning for each trend."
)
# About 30, 1,000 and 8,500 prompt tokens; each reply's exact count is recorded.
PROMPTS = {
    "short": "Explain in detail how a lighthouse keeps ships safe at night, step by step.",
    "1k": f"{_records(30)}\n\n{REPORT}",
    "8k": f"{_records(250)}\n\n{REPORT}",
}


def decode_rate(completion_tokens: int, first: float, last: float) -> float | None:
    """Tokens per second after the first token; None when there is no span."""
    if completion_tokens < 2 or last <= first:
        return None
    return (completion_tokens - 1) / (last - first)


def model_id() -> str:
    r = requests.get(f"{URL}/v1/models", timeout=30)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def request(model: str, prompt: str) -> dict:
    """One streamed greedy reply, with its timings."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    first = last = None
    usage = None
    with requests.post(
        f"{URL}/v1/chat/completions", json=body, stream=True, timeout=600
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            msg = json.loads(line[6:])
            usage = msg.get("usage") or usage
            deltas = [c.get("delta") or {} for c in msg.get("choices", [])]
            if any(d.get("content") or d.get("reasoning_content") for d in deltas):
                now = time.perf_counter() - t0
                first = first if first is not None else now
                last = now
    if first is None or usage is None:
        raise RuntimeError("the reply carried no tokens or no usage")
    tokens = usage["completion_tokens"]
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": tokens,
        "ttft": first,
        "decode_tok_s": decode_rate(tokens, first, last),
    }


def summarize(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Median, min and max of each timing over the replies that report it."""
    out = {}
    for key in ("ttft", "decode_tok_s"):
        values = [r[key] for r in rows if r[key] is not None]
        if values:
            out[key] = {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
    return out


def server_state() -> dict:
    """The server's version, launch command and mean accepted tokens per MTP step."""
    r = requests.get(f"{URL}/server_info", timeout=30)
    r.raise_for_status()
    info = r.json()
    return {
        "version": info["version"],
        "launch_command": info["launch_command"],
        "avg_spec_accept_length": info["internal_states"][0].get(
            "avg_spec_accept_length"
        ),
    }


def gpu() -> str:
    name = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.cuda.get_device_name())"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    # nvidia-smi fails while the loaded driver and its user-space library
    # differ; the kernel module's own version line does not.
    driver = re.search(
        r"\s(\d+\.\d+(?:\.\d+)?)\s", Path("/proc/driver/nvidia/version").read_text()
    ).group(1)
    return f"{name}, {driver}"


def time_arm(repeats: int) -> dict:
    model = model_id()
    request(model, "Say hello in one short sentence.")
    result = {"model": model, "max_tokens": MAX_TOKENS, "prompts": {}}
    for name, prompt in PROMPTS.items():
        rows = []
        for k in range(repeats):
            requests.post(f"{URL}/flush_cache", timeout=60).raise_for_status()
            r = request(model, prompt)
            rows.append(r)
            rate = r["decode_tok_s"]
            rate_text = "n/a" if rate is None else f"{rate:.1f} tok/s"
            print(
                f"{name} r{k}: {r['prompt_tokens']} in, {r['completion_tokens']} out, "
                f"ttft {r['ttft'] * 1e3:.1f} ms, decode {rate_text}",
                flush=True,
            )
        result["prompts"][name] = {"replies": rows, "summary": summarize(rows)}
    result["gpu"] = gpu()
    result["server"] = server_state()
    return result


def eval_arm(benchmark: str, num_examples: int | None, out_dir: Path) -> None:
    cmd = [
        os.environ.get("SGL_EVAL", "sgl-eval"),
        "run",
        benchmark,
        "--base-url",
        f"{URL}/v1",
        "--out-dir",
        str(out_dir),
        "--temperature",
        "0",
        "--top-p",
        "1",
        "--max-tokens",
        "2048",
        "--num-threads",
        "1",
        "--no-thinking",
        "--chat-template-kwarg",
        "enable_thinking=false",
    ]
    if num_examples is not None:
        cmd += ["--num-examples", str(num_examples)]
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("time")
    t.add_argument("arms", nargs="+", choices=sorted(ARMS))
    t.add_argument("--out-dir", type=Path, required=True)
    t.add_argument("--repeats", type=int, default=5)
    e = sub.add_parser("eval")
    e.add_argument("arm", choices=sorted(ARMS))
    e.add_argument("benchmark")
    e.add_argument("--out-dir", type=Path, required=True)
    e.add_argument("--num-examples", type=int)
    args = ap.parse_args()

    if args.command == "time":
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for name in args.arms:
            with Server(ARMS[name], args.out_dir / f"{name}.log"):
                result = time_arm(args.repeats)
            (args.out_dir / f"{name}.json").write_text(
                json.dumps(result, indent=1) + "\n"
            )
    else:
        out = args.out_dir / f"{args.arm}-{args.benchmark}"
        with Server(ARMS[args.arm], args.out_dir / f"{args.arm}-{args.benchmark}.log"):
            eval_arm(args.benchmark, args.num_examples, out)
            (out / "server.json").write_text(
                json.dumps(server_state(), indent=1) + "\n"
            )


if __name__ == "__main__":
    main()
