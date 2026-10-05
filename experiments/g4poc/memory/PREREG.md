# g4poc memory levers: preregistered predictions

Written before each lever runs; never edited afterwards (a miss is explained in REPORT.md).
Vault trial pages carry the same numbers (`trials/g4poc/g4poc-l*.md`).

## Control and step-0 metric

Control `mem-base` (`refs.json`) = PA's r03 on build-server-3: `--mem-fraction-static 0.93
--swa-full-tokens-ratio 0.3 --disable-prefill-cuda-graph`, FP8 weights and KV, context 16384.
Step-0 metric `max_clean_sessions_5k300` (`memory/capacity.py`): the largest N simultaneous
5,000-token random prompts decoding 300 tokens each that all run at once with no retraction.
Control: **17** (run `cap-mem-base-20261004-153634-build-server-3-d773ef`).

Measured per-session use in that run (peaks of the N = 14/16/17 bursts):
- full pool 5,299 tokens per session (prompt + output), 10,240 B/token;
- sliding pool 1,322 N + 2,544 tokens (linear fit, exact at all three N), 102,400 B/token;
- KV budget 3,693,383,680 B; GPU peak 30,990 of 32,607 MiB (1,617 MiB never touched), the same
  with 6 x 10K prompts (activations are bounded by the 4,096-token prefill chunk).

Capacity model: N = min(floor(F / 5,299), floor((S - 2,544) / 1,322)) with F full and S sliding
tokens; the KV budget grows 1:1 with any memory freed before pool sizing.

Safety rule for any change that grows the static pool: GPU memory never used at the peak of
the step-0 bursts (including 6 x 10K prompts) stays >= 512 MiB. A ref that breaks it is not
kept, whatever its capacity.

## 2026-10-05, flag levers (all numerics-unchanged: pool sizes only)

| trial | change vs mem-base | prediction | falsified if |
|---|---|---|---|
| g4poc-l9 | `--json-model-override-args '{"max_position_embeddings": 16384}'` | RoPE cos/sin caches (FP32, 262,144 positions: 256 MiB sliding + 512 MiB full) shrink to ~16.8K positions (~49 MiB): weights line 25.12 -> ~24.42 GiB; KV +0.70 GiB; full-limited at ratio 0.3: **17 -> 20 (+17.6%)** | weights line drops < 0.6 GiB, or capacity < 19 |
| g4poc-l7 | `--mem-fraction-static 0.96` | +0.92 GiB KV (0.03 x 30.71 GiB pre-load); **17 -> 21 (+23.5%)**; never-used memory at peak ~673 MiB (passes the 512 MiB rule). 0.97 probed too: 23 sessions but ~359 MiB never used, predicted to fail the rule | capacity < 20, or 0.96 breaks the safety rule |
| g4poc-l10 | `--max-running-requests 64` | req_to_token sized for 2,818 x 16,384 int32 (184.7 MB) drops to 65 rows; it is allocated after the KV pool is sized, so the pool is unchanged: **17 -> 17 (0%)** and available memory after graph capture 1.73 -> ~1.90 GiB. Value only through L7 (+~0.005 of mem fraction) | capacity changes, or available memory after capture rises < 0.12 GiB |
| g4poc-l4 | `--swa-full-tokens-ratio 0.276` | refit to the measured mix at the same budget: F 95,927, S 26,476: **17 -> 18 (+5.9%)**; knife-edge (only ratios ~0.273-0.278 give 18 at this budget; 0.27 gives 17 sliding-limited) | capacity != 18 |

Not additive: each frees bytes the pool ratio must re-split. The stack (L9 + L7 + L10 + refit
ratio) is registered separately after these step-0 results.

Risk carried from the vault (gemma4nv-b3-w32a): a request with input logprobs over a 4,096-token
chunk materializes ~2-4 GiB of logits over the 262K vocab, more than the 0.5-1.7 GiB these refs
leave free after graph capture. The role-play deployment sends no logprob requests; any quality guard that does
(teacher-forced NLL, KL) must run with logprob_start_len near the reply, or on a lower-fraction
server.
