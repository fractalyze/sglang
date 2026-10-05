# g4poc HiCache SWA admission fix: preregistration

Registered 2026-10-05 ~16:00 KST, before any GPU run of the fix. Owner: PC3 (Claude agent loop).

## HC2: mem-hc + SWA admission pin of the HiCache load-back window (`mem-hc-fix`)

**Root cause (traced, run `sweep-mem-hc-dbg-20261005-152911-build-server-3-f9f0db`).** mem-hc crashed
at C28 on a batch of a 2071-token chunk continuation plus a 247-token HiCache load-back request
(host hit 5286 FULL / 1019 SWA tokens). Prefill admission pinned only the device-matched `last_node`.
After `init_load_back` the request's sliding-window lock is anchored at `best_match_node`, and that
window also covered two device-resident SWA nodes (169 + 181 = 350 tokens). The check had counted
them as evictable, and `swa_host_hit_length` did not charge them. Admission saw `rem_swa` 1567 for
1442 needed; after the load it was -225. The allocator then had 2270 free + evictable for 2318.

**Change.** `SGLANG_OPT_HICACHE_PIN_LOAD_BACK_WINDOW=1` (commit 9ce3db9471, default off): admission
also pins `best_match_node` with the same scoped lock as `last_node`. The pinned SWA leaves the
evictable budget before the check, and an over-committing candidate waits instead. The full-layer
side needs no change: its charge already bills every token past the device match as new.
CPU regression: `test/registered/unit/managers/test_prefill_adder.py::TestHiCacheLoadBackSWAWindowPin`
(the trace's numbers; fails without the pin).

**Runs (bs3, 28G scope, harness-pc3).**
1. `gate sweep --ref mem-hc-fix --load inflight --concurrency 20,24,28,32`.
2. Exactness: `python -m hicache.exactness_mt run` for mem-final (control: device hits) and
   mem-hc-fix-smallpool (16K-token device pool: every later turn is a host load-back); 4 role-play
   sessions x 3 turns, concurrency 1, greedy, 128 output tokens.

**Predictions.**
- No scheduler failure at any point C20-C32.
- C24 matches mem-hc C24 within noise: output tok/s 801 +/- 5%, p90 9.08 +/- 0.5 s.
- C28 output tok/s above mem-final's 468: registered interval +20% to +80% (562-842).
- C20 p90 <= 10 s (the 10 s SLO still holds where mem-final's capacity was).
- Exactness: 12/12 outputs token-identical to the control, with identical `cached_tokens`.

**Falsified if** any point fails or retracts into errors, C28 output tok/s <= 468, C24 is outside
the band above, C20 p90 > 10 s, or exactness < 12/12.

## HC3: mem-hc-fix + write-through fence (`mem-hc-fix2`)

Registered 2026-10-05 ~17:00 KST. The byte-trace run had started (16:58) but its result was unread;
the exactness and the deciding sweep had not started.

**Root cause of HC2's exactness miss (5/12)**, from the byte-level trace
`exactmt-mem-hc-fix-smallpool-rt-20261005-165220-build-server-3-34932b`: write-through D2H copies are
ordered only after the scheduler stream, while under the overlap scheduler the forward writing the
inserted node's KV is still queued on the forward stream. Each finished turn's host copy holds one
half-written token (full layer 0, SWA layers 5-24); two chunked prompt nodes were stale in every full
layer. Attribution: mem-base (no study levers) + HiCache is also 5/12; the device-only A/A and HiCache
without load-backs are 12/12. Fix: `SGLANG_OPT_HICACHE_FENCE_WRITE_THROUGH=1` (commit 19e850a228) makes
the D2H stream wait on the forward stream before each write submit. CPU regression:
`test/registered/unit/mem_cache/test_hicache_write_fence.py` (fails without the fence).

**Predictions.**
- Byte trace (`mem-hc-fix2-smallpool-rt`): 0 device/host mismatches at every write and load ack.
- Exactness (`mem-hc-fix2-smallpool` vs mem-final): 12/12 token-identical, identical `cached_tokens`.
- Sweep (`mem-hc-fix2`, C20-C32): no failure; output tok/s at each point within -5%..+3% of mem-hc-fix
  (714 / 797 / 813 / 835); C20 p90 <= 10 s.

**Falsified if** any mismatch in the byte trace, exactness < 12/12, any sweep failure, or a point more
than 5% below mem-hc-fix.

## HC4: HiCache + both fixes on the current final stack (`final-hc`)

Registered 2026-10-05 ~17:50 KST (bs3 clock), before any final-hc run. `final-hc` = final-mem-c1-c2a
(mem-final flags + C1 MoE config + C2-A sm120 FP8-KV tiles) + HiCache (12 GB, write-through, kernel io,
page_first) + `SGLANG_OPT_HICACHE_PIN_LOAD_BACK_WINDOW=1` + `SGLANG_OPT_HICACHE_FENCE_WRITE_THROUGH=1`, on
tree 85ad37af45 (= 1425761173 + the two fix commits, branch jumanzii/g4poc-final-hicache), 28G scope.
Control: final-mem-c1-c2a on bs3 (PC2's sweep: C24 833 tok/s, p90 8.59 s; C28 616 tok/s, hit 0.15).

**Predictions.**
- No failure at C20-C32.
- C28 output tok/s +30% to +70% over final-mem-c1-c2a's 616 (801-1047): the host pool keeps the hit
  rate near HC2's 0.71 where the device-only final falls to 0.15.
- C24 output tok/s within -3%..+25% of 833 (HC2 gained +17% at C24 over its own control).
- C20 p90 <= 10 s.
- Multi-turn exactness (`final-hc-smallpool` vs `final-mem-c1-c2a`, C1): 12/12.

**Falsified if** any failure, C28 <= 616 tok/s, C24 below 808 tok/s, C20 p90 > 10 s, or exactness < 12/12.
