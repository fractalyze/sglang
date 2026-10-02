"""Fixed constants of the gemma4nv verify gate.

Everything a trial must not change lives here: the workloads, the timing
protocol, the fidelity budgets and the host paths. A change to this file is a
gate change, not a trial, and invalidates comparisons against older runs.
"""

import os

import msgspec

STUDY = "gemma4nv"
ROOT = os.environ.get("G4", "/data/jooman/gemma4nv")
MODEL_DIR = os.environ.get("G4_MODEL_DIR", os.path.join(ROOT, "models", "Gemma-4-26B-A4B-NVFP4"))
VENV_PYTHON = os.path.join(os.environ.get("G4_VENV", os.path.join(ROOT, "venv")), "bin", "python")
MODEL_ID = "nvidia/Gemma-4-26B-A4B-NVFP4"
MODEL_REVISION = "a19cfe00be84568a6867111c9a68c9c44fdcffe6"
RUNS_DIR = os.path.join(ROOT, "runs")
LEDGER = os.path.join(ROOT, "ledger", "evaluations.jsonl")
TREES_DIR = os.path.join(ROOT, "trees")
SRC_REPO = os.path.join(ROOT, "src-gate")
# The fidelity prompts are materialized only on the host, never committed, so
# trial authors cannot tune against them (Yukon "hidden set").
HIDDEN_DIR = os.path.join(ROOT, "hidden")
REFERENCE_DIR = os.path.join(ROOT, "reference")
GPU_LOCK = os.path.join(ROOT, "gpu.lock")
PORT = 31000
# Host-safety protocol (gate/hostwatch.py) after two host OOM crashes on
# 2026-10-02; limits set by the study coordinator for 60 GB shared hosts.
HOST_LOCK = os.path.join(ROOT, "host.lock")
SERVER_MEMORY_MAX = "24G"
MIN_HOST_AVAILABLE_GB = 30
MAX_SWAP_USED_GB = 2
MAX_FOREIGN_GPU_GB = 4
KILL_MEM_AVAILABLE_GB = 10
KILL_LOAD1 = 48
# One CUTLASS FP4 MoE cicc peaked at 9.6 GB, so 2 jobs fit the 24 GB scope.
JIT_MAX_JOBS = 2


class Workload(msgspec.Struct, frozen=True, kw_only=True):
    name: str
    concurrency: int
    prompt_tokens: int
    decode_tokens: int
    reps_per_leg: int
    gated: bool


# W8 is the Yukon gemma track shape and carries the composite score.
W8 = Workload(name="W8", concurrency=8, prompt_tokens=1024, decode_tokens=128, reps_per_leg=4, gated=True)
# W1 gates single-stream TPOT.
W1 = Workload(name="W1", concurrency=1, prompt_tokens=1024, decode_tokens=256, reps_per_leg=3, gated=True)
# W32 is a secondary throughput number, reported but never gated.
W32 = Workload(name="W32", concurrency=32, prompt_tokens=1024, decode_tokens=128, reps_per_leg=1, gated=False)
WORKLOADS = (W8, W1, W32)

PREFILL_EXPONENT = 0.25
DECODE_EXPONENT = 0.75

MIN_PAIRS = 4
# Promotion bar = max(NOISE_SIGMAS * sigma_aa, MIN_BAR).
NOISE_SIGMAS = 3.0
MIN_BAR = 0.01

# Quiescence: wait until no foreign GPU process and the GPU is cool and idle.
MAX_START_TEMP_C = 50
QUIESCE_TIMEOUT_S = 1800
QUIESCE_POLL_S = 10

# Fidelity budgets. KL thresholds are calibrated from baseline-vs-baseline
# nondeterminism (gate calibrate) and stored next to the reference outputs.
TOKEN_MATCH_MIN = 0.90
FIDELITY_MAX_NEW_TOKENS = 192
TOP_LOGPROBS = 20
KL_CALIBRATION_FACTOR = 3.0
KL_MEAN_FLOOR = 1e-3
KL_P99_FLOOR = 1e-2

# Quality: absolute accuracy may drop at most this many points vs baseline.
QUALITY_TOLERANCE_PT = 1.0
GSM8K_N = 200
