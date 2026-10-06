# Round 2 K1: gemma4nv T4's decode-glue fusion on the g4poc FP8 finals (registered 2026-10-06 ~10:40 KST)

**Change.** `SGLANG_OPT_GEMMA4_FUSED_GLUE=2` (gemma4nv T4, commit 1d859709ef, default off). Per layer:
- one Triton kernel does the q/k/v RMSNorm, RoPE and the FP8 E4M3 KV store, with bit-identical KV bytes;
- post-attention RMSNorm + FusedAddRMSNorm run in one kernel;
- router norm + pre-FF-2 RMSNorm run as one two-output kernel;
- the next layer's input norm (the final norm after layer 29) folds into the dual-norm epilogue.

Both round-1 trees already contain T4: a0491db764 (in-flight final) and 1425761173 (chat final). The candidates
therefore differ from their finals by the env alone (refs in `glue/refs.json`):
- `final-hc-cp2048-lpm-glue` vs `final-hc-cp2048-lpm`, in flight, 28G;
- `final-mem-c1-c2a-glue` vs `final-mem-c1-c2a`, chat with think time, 24G.

**Fit to this stack (code read, 10-06).** The fused path engages only on the static NHD FP8 pool under the Triton
backend. Both finals run exactly that:
- hybrid `SWAKVPool` with E4M3 sub-pools: sliding 8 kv heads x 256, full 2 x 512;
- full layers project V with K's weights (`attention_k_eq_v`; the loader copies k_proj into the v shard), so the
  kernel's separate v slice is correct;
- sliding layers write at the backend's `swa_out_cache_loc`. In a decode graph that is the capture-stable buffer;
  in extend it is the per-forward translation;
- the unfused store is `div_` x2 by a unit scale + cast x2 + `store_kvcache`, the four elementwise kernels per
  layer in the profile;
- the L9 RoPE cache has 16,384 rows, covering every position of the 16K context;
- HiCache write-through copies are fenced behind `forward_stream`, the stream the fused kernel writes on (same as
  the unfused store);
- the dense FP8 chunked-MHA path (q in E4M3, head 192) does not apply.

Unit test: `test_gemma4_fused_glue_hybrid_pool.py` (code branch `jumanzii/g4poc-r2-k1-glue`). It checks every byte of
both sub-pools through this pool and routing, against the backend's own store.

## Measured inputs (bs2, 10-06 10:01-10:23)

`compute/profile_decode_steps.py`. A decode step is one CUDA-graph replay (kernels grouped by their
`cudaGraphLaunch` correlation id).

| point | runs | kernels / decode step | decode step span | decode share of GPU wall | extend + eager busy share | decode steps/s |
|---|---|---|---|---|---|---|
| C12 in flight | `decsteps-final-hc-cp2048-lpm-20261006-100122-*` | 1,061 (bs 12) | 14.32 ms | 0.732 | 0.203 | 51.1 |
| C28 in flight | same | 1,061 (bs 24 / 32) | 17.64 / 18.93 ms | 0.627 | 0.304 | 34.3 |
| T30, pthink30 C72 | `decsteps-final-mem-c1-c2a-20261006-101203-*` | 1,001-1,061 (bs 4-16, mean 10) | 12.9 ms (bs 8), 13.9 ms (bs 12) | 0.560 | 0.404 | 41.9 |

Glue per decode step, per layer x 30:
- `_gemma_qkv_rmsnorm`, `fused_rope`, 4 ATen elementwise (scale mul + E4M3 cast for K and V) and `store_kvcache`
  become one kernel: -6 per layer;
- the norm pairs and the input-norm fold: -3 per layer.

That is 270 launches removed per step, the same set as on the NVFP4 stack. The other ~790 kernels (FP8 per-token
quant x6, CUTLASS GEMMs, MoE, attention) are untouched.

## Prediction

Vault rule `c-gemma4nv-fused-glue-launch-saves-its-traced-time`: under decode CUDA graphs on this GPU, each removed
glue launch saves about its whole traced time. That was 2.2 us per launch at W8 and 1.6 us at W1, so use
u = 1.6-2.3 us (center 2.0). Per decode step: s = 270 x u = 0.43-0.62 ms (center 0.54).

Extend (prefill chunks of <= 2048 tokens) loses the same launches. It also loses the elementwise DRAM passes (RoPE
over q/k, the four quantize passes, three norm passes). gemma4nv T4 measured +3.0% on an 8192-row prefill; here the
chunk carries a ~5K-token prefix, so attention is a larger share. Take e = 1-3% of extend time; this term is not
measured on this stack.

GPU wall saved: r = decode share x s / step span + extend share x e. Throughput gain = r / (1 - r). In flight
(closed loop at fixed concurrency), E2E p90 scales with r.

| point | r (wall saved) | E2E p90 | output tok/s | $/1M output at $0.70 |
|---|---|---|---|---|
| C28 in flight (deciding) | 1.7-3.0% | **-1.5% .. -3.5%** (center -2.4%) | +1.5% .. +3.5% | -1.5% .. -3.4% (0.203 -> ~0.198) |
| C12 in flight | 2.4-3.8% | -2.0% .. -4.5% (center -3.2%) | +2.0% .. +4.5% | -2.0% .. -4.3% (0.300 -> ~0.290) |
| T30 (pthink30 C72) | 2.2-3.8% | -3% .. -10% | unchanged (fixed offered load) | -3% .. -5% via capacity |

T30 runs at a fixed offered load, about 67 live sessions, with p90 near the 10 s edge. A service-time cut r lowers
p90 directly and through queueing. Round 1's slope is +0.24 s p90 per live session near C72-C76: bs3 C72 had 67.3
live at 9.21 s, C76 73.0 live at 10.59 s. A cut of r ~ 3% is worth ~2 sessions of load, so the p90 change is about
-5% .. -8%, widened to -3% .. -10% for the slope's noise. Capacity at the 10 s SLO would rise from ~70 to ~72-74
sessions/GPU, i.e. -3% .. -5% $/1M.

Guards:
- C28 E2E p99 within +/-5% of the final;
- retractions in the A-B-B-A at most 2x the final's;
- 0 failed requests;
- no change in prefix-cache hit rate (the KV bytes are identical).

Numerics: the KV store is bit-exact; the three norm kernels reorder the sum of squares (up to 1 bf16 ulp per
element), expected tier `reorder`.

**Falsified if** any of:
- the C28 p90 gain is below 1.005 or its output tok/s gain is below +0.5%;
- C12 regresses (p90 gain < 0.995);
- a correctness gate below fails.

## Gates and decision rule (fixed before any candidate run)

Order and scripts: `glue/k1_gates.sh` on bs2. The queue stops at the first correctness failure.

0. GPU unit tests of the fused kernels and the hybrid-pool routing.
1. Mechanism: the candidate profile launches <= 811 kernels per decode step (>= 250 removed), and the server log has
   no "running unfused" fallback.
2. KL (`compute/kl_check.py`, 8 long role-play prompts): the candidate's KL mean and p99 are within the tool's 2x the
   final's A/A level, and its top-1 agreement is within its slack.
3. HiCache C1 multi-turn exactness 12/12 with identical `cached_tokens` (`hicache/exactness_mt.py`). Control: the
   device-only `final-mem-c1-c2a-cp2048-lpm-glue`. Candidate: `final-hc-cp2048-lpm-glue-smallpool` (16K device
   pool, so every later turn loads back from host KV that the fused kernel wrote).
4. Quality against the bs2 base anchor: GSM8K 1319 (`quality-compare` pass, CI low >= -1 pt) and tool JSON with no
   drop.
5. Role-play arm, paired per item on bs2: control `final-cpl-qr`, candidate `final-cpl-glue-qr`, both against the
   mem-qr-base reference copied from bs3 (sha256 2d6cad79...). Passes if:
   - the NLL rise vs the control is <= 0.02 nats/token (the rp budget);
   - net language flips out are <= 2. Round 1 HiCache arms of one config scored 65/67/68 of 80 across runs.
6. A-B-B-A at 12 and 28 in flight (`compute/sweep_abba.sh`, two pairs per point).
7. A-B-B-A at T30: pthink30 at 72, the same seeded plan in all four sweeps.

**Adopt into the in-flight final** if all of:
- gates 0-5 pass;
- the C28 p90 gain is >= 1.01 or the output tok/s gain is >= +1%, larger than the control drift;
- C12 does not regress;
- ABBA retractions are <= 2x the final's;
- a 30-min C28 soak of the final + glue on bs2 has 0 failed requests, retractions <= 2 x 0.73% = 1.46%, and
  p99 <= 1.3 x 11.62 s = 15.1 s. The reference is round 1's bs2 C28 soak of the final:
  `runs/sweep-final-hc-cp2048-lpm-20261006-074917-build-server-2-f796c4`, 9,720 requests, p99 11.62 s, 71 retracted.

**Adopt into the chat final** (`final-mem-c1-c2a`) if gates 0-5 pass (same kernels and numerics) and the T30 p90
gain is >= 1.0.
