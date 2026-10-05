# g4poc compute levers (PB, bs2): pre-registered predictions

Write-once. Each entry is committed before its gated run; a miss is explained in the
record, never by editing the number here. Vault trial ids `g4poc-c<N>` (study-g4poc).

Common setup: control `base` (PA's r03, `gate/refs.json`), gated load `inflight-C12`
(12 requests always in flight, 60 s warm-up + 240 s window), 4 ABBA pairs, fresh server
per leg, build-server-2 only. Deciding metric: E2E p90 gain (control/candidate), bar =
max(3 sigma of the A/A noise at the same load, 1%). Fidelity on pair 0; the candidate must
not regress output tok/s past its bar.

## C1: tuned Triton fused_moe config (registered 2026-10-05 ~11:50 KST)

Change: `SGLANG_MOE_CONFIG_DIR` points at a config file produced by SGLang's
`benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` (run through
`compute/moe_tune.py`; fp8_w8a8, per-channel, E=128, N=704, 1,920 configs x 18 token
counts) instead of the default config. Nothing else changes. One file covers both MoE
GEMMs (no `_down` file).

Prediction: E2E p90 at inflight-C12 improves by 1% to 6% (delta -6% .. -1%).
Falsified if the gain is under 1%, or the fidelity tier is worse than reorder (a tile
change reorders the K reduction, nothing more).

Basis (analytic): MoE is ~21% of GPU time on the base. The default config's microbench
on bs2 (`moe-tune/c1/bench-default.log`) is at 80-95% of the HBM bound for 4-32 tokens
(decode: 120 us at 4, 318 us at 16, 423 us at 32) and ~267 TFLOPS at 4,096 tokens
(1.46 ms, the prefill chunk). Headroom: ~10-20% of decode MoE time, ~15-30% of prefill
MoE time, so 2-6% of E2E at most.

## C3: scheduling flags across the hit-rate cliff (registered 2026-10-05 ~14:00 KST)

Changes (flags only, each against `base`): C3a `--schedule-policy lpm` (ref `c3-lpm`; the base is fcfs),
C3b `--chunked-prefill-size 2048` (ref `c3-cp2048`; the base is 4096).

Setup: nested in-flight sweeps base, c3-lpm, c3-cp2048, c3-cp2048, c3-lpm, base
(`compute/sweep_nested.sh`, one server per sweep) at 8, 12, 16 and 20 in flight; per point 60 s warm-up +
240 s window, cache flushed between points. Per candidate and point (`compute/sweep_abba.py`): E2E p90 gain
(control/candidate) and output tok/s gain (candidate/control), geometric mean of the two control/candidate
pairs. Bar per point: max(the gate's C12 bar from the A/A, |log| of the control's drift at that point
(first over last control sweep)).

Keep rule: a flag is kept if at 16 or 20 in flight E2E p90 gain or output tok/s gain clears the bar with
both pairs on the same side, and no point loses more than its bar on either metric. Inside the bar
everywhere = neutral: C3a is dropped; C3b stays available to PC's memory stack (its smaller prefill
transient is a memory lever, measured by PC, not here).

Prediction C3a (lpm): inside the bar at 8 and 12 (the base never queues there: mean queue 0.0, so the
order cannot differ). At 16 (mean queue 1.1, hit 0.22) E2E p90 delta -5% .. 0% and hit rate +0 .. +0.10;
at 20 inside the bar (hit 0.002: no cached prefix survives for lpm to prefer). Falsified if any point gains
more than 5% or loses more than its bar.

Prediction C3b (cp2048): E2E p90 delta -1% .. +3% and output tok/s -3% .. +1% at every point. Decode
steps wait behind half as many prefill tokens, but a cold or re-prefilled ~5.6K-token prompt takes 3
forwards instead of 2, and from 16 up every turn re-prefills, so the cost shows most there. Falsified if any
point gains more than 1% or loses more than 3%.

Basis: the base sweep's server log (COMPUTE.md section 1: queue means 0.0 / 0.0 / 1.1 / 4.6 at 8 / 12 / 16
/ 20, hit 0.74 / 0.70 / 0.22 / 0.002); vault gemma4nv-b2-t1 (chunk 8192 on bs2 moved waiting between the
TTFT and decode terms, composite -0.74%, inside its bar).

## C2-A: sm120 FP8-KV Triton extend tiles (registered 2026-10-05 ~14:35 KST)

Change: ref `c2-extend-tiles` = base + `SGLANG_OPT_TRITON_EXTEND_SM120_FP8_KV_TILES=1` at SGLang a10694ad32
(python differs from the base's commit only in `extend_attention.py` and `environ.py`; env off = base tiles).
Extend (prefill) attention tiles: head_dim 512 (32, 32, 64, 8 warps, 1 stage) instead of (32, 32, 32, 8, 1);
head_dim 256 (32, 32, 32, 4, 1) instead of (64, 64, 64, 8, 1). Decode attention, MoE and everything else unchanged.

Gate: (1) `compute/kl_check.py` on 8 long role-play first turns (one per language, ~5K tokens): batched
teacher-forced KL mean / p99 within 2x of the base's batched-vs-serial A/A level and worst top-1 agreement
within 2 points of it; greedy token identity one prompt at a time reported. (2) Deciding: `gate run` base vs
c2-extend-tiles at inflight-C12, 4 ABBA pairs, bar max(3 sigma A/A, 1%) = 1% (A/A aa-c12-20261005-132013).
(3) Confirming: `compute/sweep_abba.sh base c2-extend-tiles 8,16` (A B B A sweeps).

Prediction: E2E p90 at inflight-C12 delta -12% .. -5%; output tok/s up by a similar amount. At 8 in flight
-12% .. -4%; at 16 in flight -20% .. -5% (past the cliff more of each turn is prefill). Falsified if the C12
gain is under 5% (delta above -5%), or the KL check fails (a fidelity tier worse than reorder).

Basis (microbench extrapolation): base profile at C12 (`compute/profile_extend_share.py`,
runs/profile-base-C12-20261005-132505, 10 s window, 39 extend forwards): the extend kernels are 14.9% of
wall time, head_dim 512 8.8% and 256 6.1% (split by layer position; the launch signatures agree). The
microbench's request-weighted mixes speed them up 2.93x and 2.02x (`c2/bench.json`), so the GPU-time saving
is 0.088 x (1 - 1/2.93) + 0.061 x (1 - 1/2.02) = 8.8%; with the slowest shape's speedups alone (2.28x,
1.71x) 7.4%. The profile's mean kernel time per launch (4.5 ms at 512, 0.62 ms at 256) is above the
microbench mix's (3.2 ms, 0.40 ms), i.e. the served mix leans to the large chunks with the larger speedups.
At a fixed in-flight count E2E scales with GPU time per reply, and p90 replies carry the larger prefills.

## C4 and the C3 flags on the final stack (registered 2026-10-05 ~19:50 KST)

Control: `final-hc` (the study's final: mem-final + C1 + C2-A + HiCache, tree a0491db764, 28G scope). Candidates
(each the control plus one flag): `final-hc-kvs16` (C4: `--triton-attention-num-kv-splits 16`, default 8),
`final-hc-lpm` (C3a on the final), `final-hc-cp2048` (C3b on the final). Each is smoke-run first (a candidate
whose server fails is dropped and reported). Nested in-flight sweeps control, candidates..., candidates
reversed, control (`compute/sweep_nested.sh`) at 24 and 32 in flight, the final's operating range.

Keep rule (`compute/c4_pick.py`): a flag is kept if its mean log E2E p90 gain over 24 and 32 exceeds log 1.01
and neither point's gain is below 0.99. The kept flags are combined into one ref (final-hc plus all of them),
which gets the long role-play KL check against final-hc (`compute/kl_check.py`, batched KL within 2x the A/A
level) and a confirming A-B-B-A at 24 and 32; the morning replicate adds them only if both hold.

Prediction C4 (kv-split cap 16): E2E p90 delta -1.5% .. +0.5% at both 24 and 32 (inside the bar). Basis: the
decode microbench (`compute/decode_split_bench.py`, c4/decode_split.json): with the served kv-head count the
backend's split heuristic wants more than 8 splits, but the decode attention step only gains at small batches
(-16% at 12, 0% at 20, -4.4% at 32 for cap 16; cap 32 and 64 no better), and decode attention is ~8-12% of GPU
time, so under 0.5% of E2E at the final's operating points. Falsified if either point gains more than 1.5% or
loses more than 1%.

Prediction C3a on the final (lpm): E2E p90 delta -5% .. +1% at both points. On the base lpm only acted past the
cliff (-6.3% at 16, where turns queue); with HiCache the cliff lies past 32, but the device pool still holds
only ~27 histories, so at 32 some turns wait and lpm may order them by device-cached prefix. Falsified if
either point gains more than 5% or loses more than 1%.

Prediction C3b on the final (chunk 2048): E2E p90 delta -4% .. +1% at both points. On the base it gained 1-4%
at every point with a higher hit rate below the cliff; on mem-final it cost 2 burst sessions (PC, 27 vs 29),
which HiCache's host tier should absorb. Falsified if either point gains more than 4% or loses more than 1%.

## HS1': host-RAM sizing slope inside the 28G scope (registered 2026-10-05 ~22:20 KST)

Arms: `final-hc` (12 GB host pool) and `final-hc-hc6` (the same with `--hicache-size 6`), each one `gate sweep
--load think30 --concurrency 64` (64 concurrent sessions, think time x1.676 = 30 s mean, 240 s warm-up + 480 s
window; a closed population whose sessions' turn 0 has no think time, so ~24 s mean think per turn), 28G scope,
build-server-2, after the overnight queue.

Model (`compute/fleet_model.py`): the storage bound is (device full pool 159,724 + 26,537 host tokens per GB) /
~5,900 tokens per history = **54 sessions at 6 GB and 77 at 12 GB**; the compute bound at ~24 s think is ~150, so
storage binds in both arms. 64 sessions sit between the two bounds.

Prediction, 12 GB arm: the live histories fit, so it holds: prefix hit 0.50-0.80, output 380-480 tok/s (64
sessions / (24 s + ~3 s E2E) x ~181 tokens), E2E p90 <= 6 s.

Prediction, 6 GB arm: past its storage bound, so histories are evicted before most sessions return (partial
thrash; with lognormal think times some return within the cache's residence time): prefix hit 0.05-0.40, E2E p90
+30% .. +300% over the 12 GB arm, output tok/s -20% .. 0% (the turn rate is set mostly by think time; the GPU
absorbs the re-prefill at this load).

Falsified if the 6 GB arm's hit rate is within 0.15 of the 12 GB arm's, or the 12 GB arm's hit rate is under 0.40.
With PC2's queue24 bracket (12 GB at 48/72/96 sessions, bs3) this gives two points on the storage-bound line.

## HS1'': host-RAM sizing re-aimed between the inclusive-mirror bounds (registered 2026-10-05 ~23:00 KST)

PC4's code read (tree a0491db764): under write_through the host pool mirrors the device, so a GPU holds
max(device, host) tokens of history, at ~6.5K tokens per stored session: **~25 sessions at 6 GB** (the 6 GB host
pool, 159K tokens, is no larger than the device pool, 160K, so it adds nothing) and **~49 at 12 GB**. HS1' (64
sessions) lies past both bounds and cannot separate them; it stays registered and runs after HS1'' if time allows.

Arms: `final-hc` (12 GB) and `final-hc-hc6` (6 GB), each `gate sweep --load think30 --concurrency 36` (36 sessions,
~24 s mean think per turn), 28G scope, build-server-2.

Prediction, 12 GB arm (36 < 49, the live histories fit): prefix hit 0.29 .. 0.75 (PC2 measured 0.29 at 48 sessions,
at the bound), E2E p90 <= 7 s, output 200-290 tok/s (36 / (24 s + ~2.5 s) x ~181 tokens).
Prediction, 6 GB arm (36 > 25): thrashes like the stack without HiCache: prefix hit <= 0.15, E2E p90 +10% .. +150%
over the 12 GB arm.
Falsified if the 6 GB arm's hit rate is within 0.10 of the 12 GB arm's, or the 12 GB arm's hit rate is under 0.29.

## HS2: chunk 2048 + lpm under think time, same-host A/B (registered 2026-10-06 ~03:45 KST)

The final config's poisson think30 sweep on bs2 (`final-hc-cp2048-lpm`, `runs/sweep-final-hc-cp2048-lpm-20261006-030116-
build-server-2-654b74`) ran before this registration. At 48 sessions (41.6 live) it held a prefix hit of **0.236**
(p90 5.59 s); PC2's `final-hc` on bs3 at the same load and the same session plan (digest 99c2b3e877e4, 41.3 live)
held **0.462** (p90 5.15 s). At 64: 0.038 / p90 9.31 s against 0.116 / 8.05 s. Device evictions (3.47M vs 3.51M
tokens) and turn rates (1.538 vs 1.544/s) match, so the gap is in host-tier hits. Read through the retention model,
the final writes ~0.34 GB of host pool per turn against ~0.22 for final-hc (retention ~23 s vs ~35 s at 12 GB).
Mechanism proposed: chunk 2048 splits a ~4.3K-token uncached prefill into 3 chunks, and every chunk boundary leaves
a sliding-window node (~1K tokens x 102 KB) that write_through copies to the SWA host pool; fewer hits mean more
uncached tokens and more chunks. lpm only reorders the waiting queue, which averaged 0.03 requests (max 2) here.

Arm: `final-hc` (chunk 4096, fcfs), `gate sweep --load pthink30 --concurrency 48,64` (48 alone if 64 does not fit by
07:55), 28G scope, build-server-2, after PC4's bs2 queue. The other arm is the final's sweep above.

Prediction (cp2048 costs think-time hit rate): final-hc on bs2 at 48 sessions holds a prefix hit of 0.38 .. 0.55 and
E2E p90 <= 5.4 s; at 64, hit 0.07 .. 0.18 and p90 <= 8.8 s. Falsified if final-hc's hit at 48 on bs2 is <= 0.30 (the
gap is the host or the run, not chunking).

Outcome (10-06 ~04:20): the bs2 final-hc arm was not run. PC2 ran the final on bs3 at C64 on final-hc's bs3 plan
(same host): p90 9.40 s, hit 0.034, against final-hc's 8.05 s, 0.116, and the final's bs2 point (9.31 s, 0.038)
matches it. At 48 sessions, across hosts, final-hc's hit 0.462 lies in the predicted 0.38 .. 0.55 and the final's
p90 is +8.7% (interval +3 .. +20%). Not falsified; vault g4poc-hs2 retired (the flags stay for in-flight traffic).

## M128: PC4's bug-4 fix on the final config (registered 2026-10-06 ~05:40 KST)

`final-hc-cp2048-lpm-m128` = the final + `SGLANG_OPT_SWA_PREFILL_WINDOW_MARGIN=128` on tree cbf56143b5 (a tree insert
keeps 128 SWA tokens below the window, and a branch-inserted prompt holds window + margin through decode). PC4's
result on final-hc (bs2, C28, ABBA): +7.1% output tok/s, -7.3% p90, prefix hit 0.71 -> 0.80, retractions 2-3 -> 7-8
per 240 s window. The final already holds a higher hit (0.77-0.79: chunk 2048's finer windows), so less is left.

Runs (`compute/m128_run.sh`, bs2, 28G): (a) A-B-B-A at 28 in flight, final vs -m128; (b) quality vs the base anchor;
(c) 30-min soak of -m128 at 28; (d) 6 s point; (e) a second A-B-B-A if it fits.

Prediction, (a): E2E p90 gain 1.01 .. 1.06, output tok/s +1% .. +5%, prefix hit 0.80 .. 0.86, retractions 2-4x the
final's, E2E p99 between -5% and +20%. (b): GSM8K within the 1 pt tolerance, tool JSON 40/40. (c): 0 failed
requests, retractions <= 3% of requests, p99 <= 1.3x the final's C32 soak p99 scaled to C28 (no 28 soak of the final
on bs2; bs3's C28 soak is the reference).
Falsified (for promotion) if the p90 gain is < 1.0, or p99 rises > 20%, or the soak fails a request or retracts > 3%.
Soak reference, fixed before the run (the final itself has no C28 soak): PC2's bs3 30-min soaks of final-hc gave
p99 9.9 s with 19 retractions of 9,353 requests at C28 and 11.0 s / 31 at C32; the final's bs2 C32 soak gave
p99 37.1 s / 115 of 9,569. (c) passes if -m128 at C28 fails no request, retracts <= 3% and keeps p99 <= 1.3 x 37.1 s.
Amendment (10-06 ~05:45, coordinator; before (c) starts): the C32 soak is the wrong reference for a C28 soak. The
reference is the final's own 30-min C28 soak, PC2 on bs3 (same config, cross-host), finishing ~05:56, before any
M128 run starts. (c) passes if -m128 fails no request, its retraction rate is <= 2x that soak's and its p99 is
<= 1.3x that soak's p99. (a)'s A-B-B-A gives the same-host C28 pairs.
Reference values (PC2, bs3, `runs/sweep-final-hc-cp2048-lpm-20261006-052216-build-server-3-d12892`, soak-C28, read
10-06 ~05:57 before (c) starts): 9,656 requests, 0 failed, p90 8.59 s, p99 11.13 s, 952 tok/s, hit 0.768, 67
retracted (0.69%). So (c) passes if -m128 fails no request, retracts <= 1.39% of requests and keeps p99 <= 14.47 s.
