"""Fixed constants of the g4poc gate.

Everything a trial must not change lives here: the session loads, the decision
metrics' parameters, the fidelity budgets and the host paths. A change to this
file is a gate change, not a trial, and invalidates comparisons against older
runs. Infrastructure constants (host safety, quiescence, fidelity) are the
gemma4nv gate's, unchanged.
"""

import os
from typing import Tuple

import msgspec

STUDY = "g4poc"
ROOT = os.environ.get("G4POC", "/data/jooman/g4poc")
MODEL_ID = "google/gemma-4-26B-A4B-it"
MODEL_REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
MODEL_DIR = os.environ.get("G4POC_MODEL_DIR", os.path.join(ROOT, "models", "gemma-4-26B-A4B-it"))
VENV_PYTHON = os.path.join(os.environ.get("G4_VENV", "/data/jooman/gemma4nv/venv"), "bin", "python")
# bs2's /data is nearly full; runs can live elsewhere (e.g. /home).
RUNS_DIR = os.environ.get("G4POC_RUNS_DIR", os.path.join(ROOT, "runs"))
LEDGER = os.path.join(ROOT, "ledger", "evaluations.jsonl")
# The gemma4nv source checkout and its per-commit worktrees are shared: refs name commits of the same repo.
TREES_DIR = os.environ.get("G4POC_TREES_DIR", "/data/jooman/gemma4nv/trees")
SRC_REPO = os.environ.get("G4POC_SRC_REPO", "/data/jooman/gemma4nv/src-gate")
HIDDEN_DIR = os.path.join(ROOT, "hidden")
REFERENCE_DIR = os.path.join(ROOT, "reference")
# Session files (workload/generate.py output) live on the host, next to the runs.
WORKLOAD_DIR = os.path.join(ROOT, "workload")
SESSIONS_PATH = os.path.join(WORKLOAD_DIR, "sessions.jsonl")
PORT = 31000

# Host-safety protocol (gate/hostwatch.py), unchanged from gemma4nv. The lock files
# are gemma4nv's so the two studies never launch engines on one host at once.
HOST_LOCK = os.environ.get("G4POC_HOST_LOCK", "/data/jooman/gemma4nv/host.lock")
GPU_LOCK = os.environ.get("G4POC_GPU_LOCK", "/data/jooman/gemma4nv/gpu.lock")
# The engine's memory scope. A run may raise it (up to SERVER_MEMORY_MAX_CEILING) when it has a measured need,
# e.g. a pinned HiCache host pool: SGLang keeps 10 GiB of the scope's headroom free beyond the pool it pins.
SERVER_MEMORY_MAX = os.environ.get("G4POC_SERVER_MEMORY_MAX", "24G")
SERVER_MEMORY_MAX_CEILING_GB = 28
MIN_HOST_AVAILABLE_GB = 30
MAX_SWAP_USED_GB = 2
MAX_FOREIGN_GPU_GB = 4
KILL_MEM_AVAILABLE_GB = 10
KILL_LOAD1 = 48
MAX_START_LOAD1 = 24
JIT_MAX_JOBS = 2
MAX_START_TEMP_C = 50
QUIESCE_TIMEOUT_S = 1800
QUIESCE_POLL_S = 10

# ---------------------------------------------------------------------------
# Workload shape (workload/generate.py defaults). Input ~5K typical, 10K max,
# output <= 300 tokens, non-streaming multi-turn role-play.
# ---------------------------------------------------------------------------
MAX_INPUT_TOKENS = 10240
MAX_OUTPUT_TOKENS = 300

# ---------------------------------------------------------------------------
# Decision metrics.
# ---------------------------------------------------------------------------
# E2E SLO: p90 end-to-end latency of a non-streaming reply (up to 300 output
# tokens). The customer has not given one, so sweeps report every SLO in
# SLOS_E2E_P90_S; the default is the gated load's bar.
DEFAULT_SLO_E2E_P90_S = 10.0
SLOS_E2E_P90_S: Tuple[float, ...] = (6.0, 10.0, 15.0)
# GPU rental prices ($/GPU-hour) every cost table is reported at. Illustrative
# points spanning RTX 5090 marketplace and on-demand prices; no single price is
# a fact about the customer's cost.
GPU_PRICES_USD_PER_HR: Tuple[float, ...] = (0.40, 0.70, 1.00, 1.50)


class SessionLoad(msgspec.Struct, frozen=True, kw_only=True):
    """An offered load, in one of two arrival modes.

    "poisson" (session layer, a fixed offered load): sessions arrive at rate
    concurrency / expected_session_s (Little's law), independent of how fast the
    server answers; each session's turns follow think time after the previous reply.
    "slots" (in-flight layer): `concurrency` slots each replay sessions back to back,
    a turn sent think_s x think_scale after the previous reply (0: always in flight).
    """

    name: str
    concurrency: int
    arrival: str = "poisson"
    warmup_s: float
    window_s: float
    # Mean session duration the arrival rate is sized with: turns x (think + E2E) at the SLO.
    expected_session_s: float
    # "scripted": the history carries the session's scripted replies and each request
    # decodes exactly the scripted reply's length (ignore_eos), so both arms of a pair
    # send identical prompts. "closed": the model's own replies enter the history.
    mode: str = "scripted"
    temperature: float = 0.0
    think_scale: float = 1.0
    # Requests still in flight this long after the window are abandoned and counted failed.
    drain_timeout_s: float = 300.0


# The gated load: a fixed offered load the FP8 base answers well inside the SLO.
# Sized on the first smoke run; re-pin with `gate sweep` (a gate change).
L64 = SessionLoad(name="L64", concurrency=64, warmup_s=120.0, window_s=480.0, expected_session_s=150.0)
# A short load for smoke tests and warm-up.
L8_SMOKE = SessionLoad(name="L8-smoke", concurrency=8, warmup_s=20.0, window_s=60.0, expected_session_s=60.0,
                       think_scale=0.2)
# In-flight layer: N requests always outstanding (sweeps pass the concurrency).
INFLIGHT = SessionLoad(name="inflight", arrival="slots", concurrency=16, warmup_s=60.0, window_s=240.0,
                       expected_session_s=0.0, think_scale=0.0)
# Stability soak: the in-flight layer held for 30 min (pass the concurrency with --concurrency).
SOAK = SessionLoad(name="soak", arrival="slots", concurrency=32, warmup_s=60.0, window_s=1800.0,
                   expected_session_s=0.0, think_scale=0.0)
LOADS = {w.name: w for w in (L64, L8_SMOKE, INFLIGHT, SOAK)}
GATED_LOAD = L64

MIN_PAIRS = 4
NOISE_SIGMAS = 3.0
MIN_BAR = 0.01

# Fidelity budgets (gemma4nv's, unchanged; thresholds recalibrated on the FP8 base).
TOKEN_MATCH_MIN = 0.90
FIDELITY_MAX_NEW_TOKENS = 192
TOP_LOGPROBS = 20
KL_CALIBRATION_FACTOR = 3.0
KL_MEAN_FLOOR = 1e-3
KL_P99_FLOOR = 1e-2
TIMED_OUTPUT_AGREEMENT_MIN = 0.5
AGREEMENT_MIN_MARGIN = 0.02

# Guards: GSM8K + tool-JSON (gemma4nv's sets) and the multilingual role-play set.
QUALITY_TOLERANCE_PT = 1.0
GSM8K_N = 200
# Role-play pairwise judge: the candidate fails when the 95% CI lower bound of its
# (loss rate - win rate) exceeds this many points. Arbitrary; set before any trial.
RP_MAX_NET_LOSS_PT = 5.0
# Role-play reference consistency: mean per-token NLL of the baseline's replies under
# the candidate may rise at most this much (nats/token) over the baseline's own.
RP_MAX_NLL_RISE = 0.02
RP_MAX_NEW_TOKENS = 300
