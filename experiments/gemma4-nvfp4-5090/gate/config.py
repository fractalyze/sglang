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
# Start only on a host well under the kill limit: other tenants' builds reached load 55.
MAX_START_LOAD1 = 24
# One CUTLASS FP4 MoE cicc peaked at 9.6 GB, so 2 jobs fit the 24 GB scope.
JIT_MAX_JOBS = 2


class Workload(msgspec.Struct, frozen=True, kw_only=True):
    name: str
    concurrency: int
    prompt_tokens: int
    decode_tokens: int
    reps_per_leg: int
    gated: bool
    # Nonempty: every leg of every pair times the same prompts, drawn from this
    # seed, so a pair's gain varies only with timing noise. Empty: each pair
    # draws fresh prompts from its own seed.
    fixed_prompt_seed: str = ""


# W8 is the Yukon gemma track shape and carries the composite score.
W8 = Workload(name="W8", concurrency=8, prompt_tokens=1024, decode_tokens=128, reps_per_leg=4, gated=True)
# W1 gates single-stream TPOT. Design v2 (W10, 2026-10-03): 24 fixed prompts
# instead of 3 fresh ones per pair. Under speculative decoding TPOT follows each
# prompt's acceptance (per-prompt log-gain sigma 0.28 in T-SPEC2b), so fresh
# prompts made the per-pair spread a prompt-sampling spread (23.7% over 4 pairs).
# Verdicts gated before v2 stand on the old design.
W1 = Workload(name="W1", concurrency=1, prompt_tokens=1024, decode_tokens=256, reps_per_leg=24, gated=True,
              fixed_prompt_seed="W1-fixed-v2")
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

# Timed-output agreement (integrity): the two arms' greedy outputs on the timed
# prompts. Hard only for a candidate that declares numerics_unchanged (or A/A):
# any batch-composition or kernel-numerics change flips greedy near-ties, so for
# the rest the teacher-forced fidelity gate decides and agreement is reported.
# The hard threshold is calibrated from an A/A run (gate set-noise): the lowest
# A/A pair mean minus max(NOISE_SIGMAS * sigma, AGREEMENT_MIN_MARGIN). Before a
# calibration exists the uncalibrated floor applies.
TIMED_OUTPUT_AGREEMENT_MIN = 0.5
AGREEMENT_MIN_MARGIN = 0.02

# Quality: absolute accuracy may drop at most this many points vs baseline.

QUALITY_TOLERANCE_PT = 1.0
GSM8K_N = 200

# Spec rule (gate spec-run, W14). Replay legs decode the control's own greedy
# text with SPEC_REPLAY_ACCEPT_LEN tokens accepted per verify (bonus included),
# about the control's W8/W32 tau (3.2 / 2.8 on bs2), so both arms verify the
# same tokens in the same rounds. A workload drawn per pair gets the fixed seed
# "<name>-<SPEC_REPLAY_SEED_SUFFIX>" so all its prompts are in the replay file.
# Tau is measured free-running on the hidden set, one prompt at a time, to a
# fixed length.
SPEC_REPLAY_ACCEPT_LEN = 3
SPEC_REPLAY_SEED_SUFFIX = "replay-v1"
TAU_MAX_NEW_TOKENS = 256
TAU_CONCURRENCY = 1
