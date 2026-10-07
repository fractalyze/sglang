# DeepSeek V3.2 AWQ: the dsv32/base stack against the max-tuned baseline

This directory measures the whole `dsv32/base` stack on one node of eight H100s
against the max-tuned SGLang baseline. These levers are switched on by default:

- `--moe-runner-backend w4a16_sm90` runs the MoE experts on the W4A16 SM90
  grouped GEMM.
- `SGLANG_USE_W4A16_SM90_GEMM=1` runs the dense AWQ projections at M <= 192 on
  the W4A16 SM90 GEMM instead of Marlin.
- `--enable-prefill-delayer` holds a rank's prefill until the other ranks have
  prefill work too, so their prefills share a step (see "Step types under DP
  attention" below).

The baseline is `s3://fractalyze-dsv32-use2/runs/baseline-current/`. Its
`summary.md` holds the numbers. Its `launch.sh` and `bench.sh` start and bench
every server here, unmodified.

## Sessions

`run_session.sh` runs one session under `gpu-lease 8`. It starts a server, then
runs GSM8K and the long-context check, benches with `bench.sh`, and records
decode-stage torch-profiler traces of a wave of c concurrent requests shaped
like `bench.sh`'s. Each profile
starts once every request is admitted and prefill has drained, or at once when
the KV pool cannot hold every request. An EAGLE profile sends four waves of
prompts, since a single EAGLE wave finishes decoding before the profiler arms.
`PROFILE_ONLY=1` re-records a session's profiles without GSM8K and the benches.

| session | arm | `launch.sh` mode | request cap | GSM8K, long context | bench | profile |
|---|---|---|---|---|---|---|
| `stack-nospec` | stack | `nospec` | 1024 | yes | c512, c1024 | c512, c1024 |
| `stack-eagle` | stack | `eagle` | 512 (its own) | yes | c128, c256 | c128, c256 |
| `base-nospec` | baseline | `nospec` | 1024 | no | c1024 | c512, c1024 |
| `base-eagle` | baseline | `eagle` | 512 (its own) | no | none | c128, c256 |
| `stack-nospec-fp8kv` | stack, `--kv-cache-dtype fp8_e4m3` | `nospec` | 1536 | yes | c512, c1024, c1536 | c512, c1024, c1536 |
| `stack-nospec-ab` | stack | `nospec` | 1024 | no | c256, c512 | c512 |
| `stack-nospec-comm` | stack + `comm` | `nospec` | 1024 | yes | c256, c512 | c512 |

`launch_stack.sh` starts the stack. It is `launch.sh` with these changes:

- the `dsv32/base` `python/sglang` tree mounted over the image's;
- each lever in `LEVERS` switched on: `moe` adds `--moe-runner-backend
  w4a16_sm90`, `dense` sets `SGLANG_USE_W4A16_SM90_GEMM=1`, `delayer` adds
  `--enable-prefill-delayer`, `comm` sets `SGLANG_OPT_USE_PUSH_AG_RS=1`, and
  `profile` mounts the step record hook;
- any server flags passed after the mode.

`LEVERS` defaults to `moe dense delayer`; `LEVERS=""` is the baseline plus the passed
flags, which is how the baseline arm adds its request cap. A new lever is one
more `LEVERS` entry. The launcher checks that each change landed in its copy of
`launch.sh` and fails otherwise.

```sh
# On the node, as root, under gpu-lease 8:
./launch_stack.sh <baseline-current dir> <python/sglang tree> nospec --max-running-requests 1024
```

The baseline image is release/v0.5.21 e00930c5, the commit `dsv32/base`
branches from, so nothing else differs. A request cap of 1024 is
`--max-running-requests 1024`. DP attention divides it across the eight ranks,
so each rank takes 128. Each session saves the derived launcher's diff against
`launch.sh` as `launch.diff`, and the per-rank KV pool
(`max_total_num_tokens`) with the scheduler's retraction count as `kv.txt`.

`stack-nospec-ab` and `stack-nospec-comm` are the `comm` lever's A/B. They run
the same commit, one without the lever and one with it, so the lever is the only
difference.

`stack-nospec-fp8kv` is a KV-capacity point, not part of the stack-vs-baseline
comparison. With the bf16 KV cache, a rank's pool cannot hold 128 requests of
2048 tokens, so c1024 measures retractions rather than kernels.

The long-context check is `niah.py`: a needle-in-a-haystack recall over 8k,
16k and 32k token contexts, eleven depths each. It is where a lossy KV cache
shows first, which GSM8K's short prompts do not exercise. It writes
`niah.txt` (recall per length) and `niah.json` (every reply).

```sh
git archive --format=tar.gz -o src.tar.gz HEAD python/sglang test/registered/kernels/benchmark/gemm
# On the node, as root, with launch_stack.sh and niah.py next to run_session.sh:
./run_session.sh stack-nospec <run-name> src.tar.gz
```

## Step types under DP attention

With DP attention every rank runs every step, because the MoE layers need all
eight ranks' tokens. A step where one rank prefills is a prefill step on all of
them: the decode-only ranks wait for it, running their decode batch as 1-token
extends. When such a step's time follows the largest per-rank prefill rather
than how many ranks prefill (`step_report.py` prints both), lining the ranks'
prefills up into shared steps cuts the stall, which is what the `delayer`
lever does. `run_steps.sh` measures how much of a run those steps take.

- `step_profile/` is a `sitecustomize` that wraps `Scheduler.run_batch` and
  appends one row per launched step to `$STEP_PROFILE_DIR/steps-dp<r>-tp<t>.csv`.
  `launch_stack.sh` mounts it when `LEVERS` includes `profile`; the mounted
  tree is not changed.
- `step_report.py` lines the ranks' rows up and splits the steps into DECODE
  (no rank prefills), PREFILL-SOME (some ranks prefill, others only decode)
  and PREFILL-ALL. For each type it reports count, step time, share of wall
  time and CUDA graph replay. It also reports the decode stall: the time the
  PREFILL-SOME steps take beyond a median decode step.
- `run_steps.sh` serves the stack in its c512 configuration with the `LEVERS`
  under test (`LEVERS="moe dense"` is the control without the delayer), plus
  any server flags. It benches each workload in `WORKLOADS` on
  varied-length random traffic (`--random-range-ratio 0.25` by default), then
  writes a `<len>_c<c>_r<i>.steps.txt` table per bench; `REPEATS=<n>` benches each
  workload n times on the same server. `GSM8K=<n>` runs GSM8K n
  times first.

`tests/test_step_report.py` checks the hook and the report on synthetic
batches and rows, without GPUs. It needs `absl-py`:
`uv run --no-project --with absl-py python tests/test_step_report.py`.

Use varied lengths for these A/Bs. With fixed lengths a closed-loop bench
finishes its requests in waves, so the steps synchronize in a way that served
traffic does not.

```sh
# On the node, as root, with launch_stack.sh, step_profile/ and step_report.py next to it:
LEVERS="moe dense" WORKLOADS="2048:512:1024" ./run_steps.sh control <run-name> src.tar.gz
WORKLOADS="2048:512:1024" ./run_steps.sh delayer <run-name> src.tar.gz [server flag ...]
python3 step_report.py <session>/steps --start-ts <t0> --end-ts <t1>
```

## Reading the traces

`benchmark/dsv32_megakernel/trace_split.py` splits each decode step into op
classes against the floor. `dense_dispatch.py` counts, for each dense AWQ
projection, how many calls the W4A16 SM90 GEMM served and how many fell back
to Marlin.

Under DP attention the attention projections run at the rank's own batch. The
shared expert, and the dense MLP of the first layers, run after the all-gather
at the batch summed over all ranks.

```sh
python benchmark/dsv32_stack/dense_dispatch.py <session>/profile-c512/trace
```

An EAGLE server's target forward is annotated VERIFY and runs the draft tokens
plus one per request. Pass `--stage VERIFY --tokens-per-request 3` to both
scripts.
