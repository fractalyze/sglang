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

## 2026-10-05, stack1 = L9 + L10 + mem fraction 0.955 + ratio 0.268 (numerics unchanged)

Inputs measured in step-0 queue 1 (results in REPORT.md): L9 frees 0.69 GiB and pools 108,092 / 32,427
tokens at 0.93; L10 frees 178 MiB of post-capture memory and leaves the pool unchanged; the per-session
fit held on every ref (full 5,299; sliding 1,322 N + 2,544).

Safety margin corrected before this registration: torch reports the card's capacity as 31.40 GiB
(32,154 MiB), not nvidia-smi's 32,607 MiB. Measured against 32,154, mem-l7-f096's burst peak (31,958 MiB)
left only ~196 MiB, and mem-l7-f097 OOMed on its first 10K prefill (176 MiB requested, 172 MiB free). From
here the rule is: nvidia-smi burst peak <= 31,642 MiB (>= 512 MiB below the CUDA-visible capacity).
L9 + L10 at 0.93 put the peak near 30,788 MiB; each +0.01 of mem fraction adds 314.5 MiB, so 0.955 is the
highest step that keeps the corrected margin (estimate ~580 MiB).

Ref `mem-stack1`: mem-base + `--json-model-override-args {"max_position_embeddings": 16384}`
`--max-running-requests 64 --mem-fraction-static 0.955 --swa-full-tokens-ratio 0.268`.
KV budget 5.25 GB (L9's 4,427,386,880 B + 0.025 x 30.71 GiB); ratio 0.268 equalizes the two pools at
the capacity model's optimum.

| trial | metric | prediction | falsified if |
|---|---|---|---|
| g4poc-s1 | max_clean_sessions (burst 5K/300) | **17 -> 26 (+52.9%)**, interval 25-27; burst peak <= 31,642 MiB (est. 31,574) | capacity < 25, or the peak breaks the corrected margin |
| g4poc-s1 (secondary, same run plan) | in-flight capacity at E2E p90 <= 10 s, `gate sweep --load inflight` on bs3 (scripted multi-turn sessions, prefix cache on), points 8/12/16/20/24(/28/32) | mem-base **12 -> stack1 20** (+66.7%); at C16 mem-base's prefix-cache hit rate collapses below 0.3 (PB's bs2 sweep of its provisional base: 0.70 at C12, 0.22 at C16) while stack1 keeps >= 0.6 | stack1's 10 s capacity <= mem-base's, or any stack1 point has a failed request |

## 2026-10-05, L5: release the slid-out part of the tree-locked SWA window during decode (code, exact)

Mechanism (code-read, REPORT.md section 2): with the radix cache on, prefill end inserts the prompt
and the request locks the tree's last SWA window (1,024 slots, one node); its own decode eviction never
reaches below the tree-protected prefix, so a session holds 1,024 + up to 300 decoded sliding slots.
Change, behind `SGLANG_OPT_SWA_RELEASE_SLID_WINDOW` (default off): every `SGLANG_SWA_EVICTION_INTERVAL`
decode tokens, split the locked window node at the slide frontier (`seqlen - 1 - window`), drop the
request's SWA lock on the part below it and move the receipt's segment boundary up to the split. The
released part becomes evictable (LRU), so the pool can reclaim it; nothing the request still reads is
released, so numerics are exact. The final window stays locked until finish, so the insert at finish
still leaves the next turn a full window.

Ref `mem-stack2` = mem-stack1 + switch on + `SGLANG_SWA_EVICTION_INTERVAL=32` + ratio refit.
stack1's step-0 fit at N 22-25 (sliding 1,426 N + 1,048; full 5,299 N) puts ~104 tokens/session of
prefill transient on top of the 1,324-slot decode hold. With L5 the decode hold becomes ~1,025 + half an
interval (~1,041), so the slope drops to ~1,164 and the budget of 5.25 GB fits ~29.7 sessions at ratio
~0.226.

| trial | metric | prediction | falsified if |
|---|---|---|---|
| g4poc-l5 | max_clean_sessions (burst 5K/300) | stack1 25 -> **stack2 29 (+16%)**, interval 27-30; peak sliding tokens per running session in the 5K bursts <= 1,250 (stack1: ~1,470) | capacity < 27, or any outputs differ from stack1 on the same scripted turns (exactness), or a tree sanity failure |

## 2026-10-05, L8: FP8 E4M3 vocab table for both the embedding lookup and the tied LM head (code, numerics change)

Change, behind `SGLANG_OPT_GEMMA4_FP8_VOCAB_TABLE` (default off): after load, quantize the tied BF16
table (262,144 x 2,816, 1.375 GiB) to E4M3 with one fp32 scale per row (0.6875 GiB + 1 MiB), drop the
BF16 table before the KV pool is sized, embed by gathering FP8 rows and dequantizing, and compute logits
with the tree's Triton FP8 vocab-head kernel (`triton_small_m_fp8_vocab_head`, max M 48) over 48-row
chunks. Unlike the existing `SGLANG_OPT_GEMMA4_FP8_LM_HEAD` (an extra FP8 copy for speed, +0.74 GB),
this removes the BF16 table.

Prior evidence (vault, other checkpoint): gemma4nv-b2-tspec6c ran the same per-row FP8 head on this
model's tied head (NVFP4 experts): full GSM8K +0.08 pt (CI -0.44..+0.59), tool-JSON 40/40, logit KL
mean 0.00043, argmax agreement 99.5%. L8 additionally quantizes the input embedding.

Ref `mem-stack3` = the best stack at run time (stack2 if L5 holds, else stack1) + switch on +
`--cuda-graph-max-bs-decode 48` (decode batches above 32 would otherwise run eager).

| trial | metric | prediction | falsified if |
|---|---|---|---|
| g4poc-l8 | max_clean_sessions (burst 5K/300) | weights line -0.69 GiB (24.43 -> ~23.74); **+4 sessions** over the stack it extends (interval +3..+5; stack2 29 -> 33) | weights drop < 0.6 GiB, or gain < 3 sessions |
| g4poc-l8 (quality guard) | GSM8K 200 greedy, tool-JSON 40, role-play reference NLL (gate quality / rp-quality vs the same stack without the switch) | GSM8K within -1.0 pt; tool-JSON 40/40; rp NLL rise <= 0.02 nats/token; rp language rate not lower | any guard fails |
