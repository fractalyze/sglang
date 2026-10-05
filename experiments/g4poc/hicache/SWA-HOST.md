# g4poc HiCache on hybrid SWA: why the host tier holds so few idle sessions

Owner: PC4 (Claude agent). Tree read: final-hc `a0491db764` (branch `jumanzii/g4poc-final-hicache`).
Question from the think-time runs (PC2, bs3, think30, closed population): final-hc's prefix hit
rate is 0.286 at 48 sessions and 0.076 at 72, but the fleet model's storage bound
(device + host KV tokens / history) predicted ~77 sessions at a 12 GB host pool.

## 1. Code read (CPU, 2026-10-05 23:10 KST)

File references are to `python/sglang/srt/mem_cache/` on `a0491db764`.

**Write-through makes the host tier inclusive, so distinct capacity = host, not device + host.**
- With `--hicache-write-policy write_through` every inserted node is backed up on insert
  (`unified_radix_cache.py:503` sets threshold 1; `unified_tree_core.py:1068-1088`).
- Host eviction only removes H-leaves: nodes already evicted from device, backed up, childless and
  unlocked on both tiers (`unified_tree_core.py:2147-2167`, `unified_cache/components/full.py:234-261`).
- Reclaiming host copies of device-resident nodes runs only under write_back
  (`unified_tree_core.py:1747-1766`).
- So everything on device is mirrored on host. Distinct KV is the host pool: 318K full + 85K SWA
  tokens, not 478K + 128K. The fleet model's storage bound should use max(device, host).

**The SWA host pool stores about one window per session, not whole prefixes.**
- Before a prompt is inserted, its out-of-window SWA slots are freed
  (`cache_unfinished_req` -> `free_out_of_window_slots`, `unified_radix_cache.py:1176-1186`;
  `SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS` defaults on). The inserted prefix is an SWA
  tombstone except for its last window.
- A backup copies SWA only for device-SWA nodes within one 1,024-token window above the node
  (`unified_cache/components/swa.py:107-128`, `1133-1149`).
- Each turn therefore writes its new extension (the previous reply plus the user message, ~300 tokens)
  and its own reply leaf. That leaf is dead: the next prompt renders the reply without the empty
  thought channel, so it never matches again. A chunked first prompt also keeps the window at each
  chunk boundary. A session's live SWA set is ~1.2-1.7K tokens.

**A host hit without the sliding window is unusable.**
- The match validator resets at any node with neither device nor host SWA. A match end is valid only
  1,024 contiguous SWA tokens past that reset (`swa.py:330-340`).
- SWA host eviction tombstones an internal node's SWA and leaves its Full KV
  (`swa.py:1487-1536`). The Full prefix is then present but cannot be used.

**A write-through backup takes Full and SWA host slots together.**
- `_execute_kv_backup` reclaims Full host space. `HybridCacheController.write` then allocates the KV
  slots and resolves the SWA slots through `HostPoolGroup.resolve_host_transfers`, which reclaims SWA
  host space once and rolls back if it still fails (`pool_host/group.py`).
- On failure the whole backup is dropped (`unified_radix_cache.py:_execute_and_commit_kv_backup`
  returns 0). The node stays device-only, and its Full KV dies when the device evicts it.
- SWA host reclaim can only evict nodes already tombstoned on device. Nodes still holding device SWA
  are not in the SWA host LRU (`swa.py:687-694`).

**How the host pool is split.**
- `--hicache-size` is split across the two pools by device bytes
  (`hybrid_cache/hybrid_pool_assembler.py:179-192`, `407-411`).
- `--hicache-ratio` (size 0) applies one ratio to both pools' token counts (`pool_host/base.py:177-182`).
- Either way, host SWA/full tokens equal the device ratio (`--swa-full-tokens-ratio`, 0.268).
- There is no separate SWA host knob.
- Per token, an SWA slot costs 10x a Full slot: 25 sliding layers at 2 KB of FP8 K per layer,
  against 5 full layers at 1 KB, with V stored separately in both.
- final-hc's 12 GB is split 3.26 GB full (318,445 tokens, 10.2 KB/token) and 8.74 GB SWA
  (85,344 tokens, 102 KB/token). Device: 160,395 full + 42,985 SWA.
- Per live session: ~6.5K Full tokens (history plus recent dead reply leaves) = ~66 MB, and
  ~1.2K SWA tokens = ~120 MB.
- Bounds by pool: Full 3.26 GB / 66 MB ~ 49 sessions; SWA 8.74 GB / 120 MB ~ 70 sessions. Full
  likely binds first. Ended sessions' histories stay until the LRU ages them out (~19% of turns
  end a session), which lowers both bounds.

**write_back is not a safe alternative for hybrid SWA today.**
- write_back makes the tiers exclusive: backup on device eviction, and reclaim of host duplicates.
- But under SWA pressure, device eviction tombstones internal nodes' SWA with no backup
  (`swa.py:713-766` -> `evict_component(DEVICE)`). Only a Full-leaf eviction builds a backup
  (`unified_tree_core.py:1626-1660`).
- A session's live window sits on internal nodes (the dead reply leaf is their child), so it would
  be lost.
- write_back's load-back exactness on this model is also unmeasured.
- Upstream-relevant open item: exclusive host tiering for hybrid SWA, meaning write_back plus an SWA
  backup when a device SWA node is tombstoned.

## 2. Preregistration (2026-10-05 ~23:40 KST, before any GPU run below)

Runs on bs3 after PC2's queue25 (think60 matrix), harness-pc, 28G scope, refs in
`experiments/g4poc/swahost/refs.json` (copied to bs3 `harness-pc/swahost/refs.json`). Both refs run
on the log-only tree `c6f392a28b` (branch `jumanzii/g4poc-swa-host-dbg`). It extends PC3's PFXDBG
admission log with `full_kv` (Full prefix present before the SWA validator), a session-instance
hash, free slots per pool, and counters of failed write-through backups and unbacked device
evictions. Analysis: `experiments/g4poc/swahost/miss_split.py`, which classifies every returning
turn as hit / declined / swa_gone / full_gone (definitions in its docstring).

### SW1 (`swahost-dbg`, think30 x 48, 72): per-turn miss split of final-hc

- The replicate holds final-hc's numbers. At 48 sessions: hit 0.20-0.37 (PC2 0.286) and E2E p90
  within ±12% of 7.07 s. At 72: hit 0.02-0.15 (PC2 0.076) and p90 within ±12% of 11.41 s.
- At 48, full_gone is >= 50% of missed returning turns (the Full host pool binds first), and
  declined is < 5%. At 72, full_gone is >= 60%.

**Falsified if** swa_gone > full_gone at 48 (the SWA host pool binds first, so the lever below
points the wrong way), declined > 10%, or the replicate is outside the bands above.

### SW2 (`swahost-r020`, think30 x 48, 72): shift both splits toward Full

The lever is config-only: `--swa-full-tokens-ratio 0.2`. The device bytes stay the same. Device
pools become ~197K full + ~39K SWA, and host pools ~392K full (4.0 GB) + ~79K SWA (8.0 GB). Bounds
by pool: Full ~60 sessions, SWA ~46-66. Device SWA still covers ~28 concurrent windows, more than
the 72-session in-flight mean (14.5).

- At 48 vs SW1 (same tree): hit +0.10 to +0.35 absolute; E2E p90 -5% to -25%; output tok/s no
  lower than SW1 - 2%.
- At 72 vs SW1: hit +0.03 to +0.20; E2E p90 0% to -20%.

**Falsified if** at 48 the hit rate is <= SW1 + 0.05 or E2E p90 is not lower than SW1's.

Not tested here: the in-flight operating point (C28). A lower SWA share shrinks the device SWA pool
that admits concurrent windows, so the ratio is a per-workload choice. Any change to the final
config would need its own in-flight check.

### SW3 (`swahost-dbg`, think30 x 32, 40): where the hit rate leaves the ceiling

Registered 2026-10-06 ~00:15 KST, after SW1's 48-session point and before any run below. SW1 at 48
(hit 0.284, E2E p90 7.35 s) split the missed returning turns 55% swa_gone and 45% full_gone, with no
failed backups. Both host pools are over capacity there, SWA slightly first.

Model: an idle session costs ~0.23 GB of HiCache host (~66 MB Full + ~150-180 MB SWA), counted
against the host alone. Ended sessions' histories add ~19% until the LRU ages them out. The 12 GB
pool then holds ~44 live sessions. The hit-rate ceiling is ~0.77 (first turns ~0.19 and the
per-turn suffix ~0.04 of prompt tokens are never reusable).

- At 32: hit 0.60-0.77; swa_gone + full_gone <= 25% of returning turns; E2E p90 -10% to -50% vs
  SW1's 48 point.
- At 40: hit 0.45-0.72.

**Falsified if** the 32-session hit rate is below 0.55, or 40 sessions hits higher than 32.

## 3. Results

Classification by `swahost/miss_split.py` on each point's 480 s window. The clock starts at the gate's
pre-point cache flush. Trimmed = the window's last 60 s cut, because harness fd08bf8fbf's slots
generator starts spurious fresh sessions at each window's end (PC2; fixed in dff92efc2b, not deployed
for these runs). Gap tables come from a per-turn gap split (time since the session's previous admission).

### SW1: final-hc, log-only tree (run `sweep-swahost-dbg-20261005-235239-build-server-3-6358fe`)

| sessions | hit | E2E p90 | out tok/s | first | hit turns | swa_gone | full_gone | declined |
|---|---|---|---|---|---|---|---|---|
| 48 | 0.284 | 7.35 s | 334 | 233 | 231 | 230 | 216 | 0 |
| 72 | 0.075 | 11.44 s | 434 | 264 | 70 | 185 | 698 | 0 |

PC2's final-hc on a0491db764 measured 0.286 / 7.07 s and 0.076 / 11.41 s, so the log-only tree
replicates. 0 failures.

- **Missed returning turns.** At 48: 52% swa_gone, 48% full_gone (trimmed 55 / 45). At 72: 21% /
  79% (trimmed 23 / 77).
- **Prompt tokens, trimmed window.** At 48: reused 0.33, first turns 0.18, per-turn suffix 0.04, SWA
  window missing 0.24, Full missing 0.21. At 72: 0.08 / 0.16 / 0.03 / 0.15 / 0.57.
- **No mechanism failures.** 0 failed write-through backups, 0 Full tokens evicted from device
  without a host copy, 0 declined load-backs. Both host pools sit at 0.1-0.5% free all window.
- **All-or-nothing misses.** When either class misses, the usable match is ~0% of the expected
  prefix. No older window survives to fall back to.

Returning turns by idle gap, as the share that hit, missed for SWA, and missed for Full:

| gap since previous turn | 48: hit / SWA / Full | 72: hit / SWA / Full |
|---|---|---|
| 10-20 s | 0.70 / 0.30 / 0.00 | 0.28 / 0.62 / 0.10 |
| 20-30 s | 0.57 / 0.38 / 0.05 | 0.07 / 0.33 / 0.60 |
| 30-40 s | 0.16 / 0.58 / 0.26 | 0.00 / 0.01 / 0.99 |
| 40-60 s | 0.00 / 0.28 / 0.72 | 0.00 / 0.00 / 1.00 |
| > 60 s | 0.00 / 0.04 / 0.96 | 0.00 / 0.00 / 1.00 |

**Reading.** Each host pool keeps an idle session for a fixed time, not a fixed count.
- **SWA window:** ~25 s at 48 sessions, ~12 s at 72.
- **Full prefix:** ~38 s at 48, ~23 s at 72.
- Retention scales with pool bytes over the per-session write rate. SWA's is ~0.6 of Full's at both
  loads, so the preregistered lean (Full binds first) is falsified at 48 and holds at 72.
- The gap distribution (think x 1.676 lognormal, mean ~24 s per turn, plus E2E) straddles both
  retentions. Short-gap misses are SWA's; long-gap misses are Full's.
- Equalizing the two retentions would move ~1 GB from Full to SWA and raise the binding retention by
  only ~12% (25 -> 28 s at 48). The split is not a meaningful lever.

**Why SWA's write rate is high.** About a third of SWA host writes are dead on arrival:
- Every turn's decode-output leaf is written. Gemma-4's template drops the empty thought channel from
  past turns, so it never matches again.
- Every chunked prefill leaves its chunk-boundary window in the tree. The window is needed only
  while the next chunk prefills. The final non-chunked insert walks the chunk nodes and backs them
  up (`unified_tree_core.py:1318`, `swa.py:107-128`): ~1K SWA tokens (~100 MB) per boundary.
- A miss re-prefills the whole history and rewrites ~2K SWA tokens, so the thrash feeds itself.
- `--chunked-prefill-size 2048` (c3b-hc) doubles the boundaries per ~5K prompt. Its effect under
  think-time load with HiCache is unmeasured; expect a shorter SWA retention.

### SW2: `--swa-full-tokens-ratio 0.2` (run `sweep-swahost-r020-20261006-001914-build-server-3-8a4dfc`)

Pools: device 196,752 Full + 39,350 SWA tokens; host 390,623 Full (4.00 GB) + 78,126 SWA (8.00 GB).

| sessions | hit (SW1) | E2E p90 (SW1) | out tok/s (SW1) | hit turns | swa_gone | full_gone | failed |
|---|---|---|---|---|---|---|---|
| 48 | 0.226 (0.284) | 7.54 s (7.35) | 331 (334) | 186 | 269 | 217 | 1 |
| 72 | 0.064 (0.075) | 11.64 s (11.44) | 432 (434) | 70 | 298 | 585 | 0 |

**Falsified, in the direction SW1's retention numbers predicted.**
- The smaller SWA host shortens SWA retention. At 48, returning turns with a 20-30 s gap hit 37% vs SW1's 57%.
- The 23% larger Full host does not reduce full_gone: 217 vs 216 turns at 48.

**Leaf-first eviction strips session tails.** Both tiers evict leaf-first.
- On device, D-leaves are evicted LRU-first. On host, only H-leaves are evictable: nodes already evicted from device and childless (`unified_tree_core.py:2128-2167`).
- An idle session therefore loses its newest node first: the dead reply leaf, then the prompt tail, which holds the session's only SWA window. Its older head stays resident on device and, mirrored, on host.
- A session becomes unusable after ~500 evicted tokens while ~5K tokens of its head still occupy both tiers.
- In SW2-48, 81% of full_gone turns still had >50% of their prefix (median 91%, ~500 tokens missing). In SW1-48, 32% did; SW1-72 has almost none (94% had nothing left).
- A bigger device pool pins more heads, so the host must evict more tails. That is why the larger Full host did not help.

**The failed request (48, end of window) was not a server failure.**
- 3 retractions ("KV cache pool is full", 00:32:38-52) hit three first-turn requests during the generator's end-of-window arrival spike. All three were re-admitted and answered.
- Every admitted load request got HTTP 200: 3,240 `/generate` responses plus the server warmup and one health check. There is no traceback, abort or error log line.
- So the failure was client-side (most likely a transport or body error) and left no server trace. The harness does not persist the error text.
- 15 other think-time points tonight, with up to 3 retractions each, had 0 failures. Not an upstream candidate.

### SW3: final-hc knee, think30 x 32, 40 (run `sweep-swahost-dbg-20261006-004733-build-server-3-a28750`)

| sessions | hit | E2E p90 | out tok/s | hit turns | swa_gone | full_gone | failed |
|---|---|---|---|---|---|---|---|
| 32 | 0.547 | 4.75 s | 233 | 332 | 106 | 35 | 0 |
| 40 | 0.438 | 5.83 s | 303 | 318 | 176 | 103 | 0 |
| 48 (SW1) | 0.284 | 7.35 s | 334 | 231 | 230 | 216 | 0 |
| 72 (SW1) | 0.075 | 11.44 s | 434 | 70 | 185 | 698 | 0 |

**Falsified by a hair.** The 32-session hit is 0.547 against the registered 0.55 floor. 40 sessions
came in at 0.438 against 0.45-0.72. The E2E p90 interval held (-35% vs SW1's 48-session point).
- The hit rate falls smoothly from 32 sessions; there is no flat region at 12 GB.
- At 32 sessions, returning turns hit 81-88% at gaps up to 40 s and 64% at 40-60 s. Retention is
  ~50-60 s for SWA and ~70-90 s for Full.
- The model's per-session cost was ~10-20% optimistic. Most of that gap is the second-turn miss below.

### Every session's second turn misses its prefix (think-time mode)

A short-gap returning turn (< 30 s) hits ~100% after a hit or an swa_gone re-prefill at 32-40 sessions.
After a first turn it mostly misses its SWA window:

| previous turn | SW3-32: hit / swa_gone | SW3-40: hit / swa_gone | SW1-48: hit / swa_gone |
|---|---|---|---|
| hit | 1.00 / 0.00 | 1.00 / 0.00 | 0.89 / 0.11 |
| swa_gone | 1.00 / 0.00 | 1.00 / 0.00 | 0.95 / 0.00 |
| full_gone | 0.50 / 0.50 | 0.58 / 0.42 | 0.38 / 0.59 |
| first | 0.27 / 0.73 | 0.12 / 0.88 | 0.14 / 0.84 |

The outcome after a first turn depends on that first prompt's length (all gaps < 30 s):

| first prompt | SW3: hit | SW1: hit | SW2: hit |
|---|---|---|---|
| <= 4096 tokens (one prefill chunk) | 0 / 67 | 0 / 93 | 0 / 92 |
| 4097-5119 | 0.42 | 0.18 | 0.16 |
| >= 5120 | 0.09 | 0.05 | 0.04 |

In all 122 cases after a <= 4096-token first prompt in SW3 (gaps 9-95 s), `full_kv` equals the
expected prefix and the usable match is 13-15 tokens.

**Mechanism.** Page size 1; Gemma-4's `sliding_window_size` is `sliding_window - 1` = 1023
(`models/gemma4_causal.py:85`).
- After a fresh prefill, `cache_unfinished_req` calls `free_out_of_window_slots(req, pe - 1)`. That
  keeps SWA for exactly [pe-1024, pe), pe being the prompt end (`mem_cache/common.py:54-107`).
- The next turn's prefix ends 4 tokens earlier, at `<|turn>model\n`. The chat template drops the
  4-token empty thought channel from a past model turn (`full_kv` = prev_fill - 4 in every case).
- The match validator needs >= 1023 contiguous SWA tokens behind the match end (`swa.py:330-340`).
  Here it gets 1024 - 4 = 1020, and the match collapses to the shared 13-token root.
- After a hit turn the window is not rebuilt from scratch: the previous window plus the ~300-token
  extension leaves ~1,300 contiguous tokens, so later turns are fine.
- A first prompt of 4097-5119 tokens is partly rescued: the chunk-boundary window at 4096 keeps SWA
  live from 3072. A full_gone re-prefill rebuilds the window from scratch, so its next turn misses
  too.
- **Cost.** Second turns are 14% of prompt tokens (first turns 17%, per-turn suffix 3.5%). The
  think-mode hit ceiling is ~0.66 instead of ~0.80. This does not depend on HiCache: device-only
  stacks take the same path.
- **Open question.** In-flight final-hc sweeps hit 0.71-0.74 at C20-C32, above the 0.66 ceiling,
  so second turns probably hit at a ~0 s gap. The code read finds no time-dependent path. A 10-minute
  in-flight point on the log-only tree (`swahost-dbg --load inflight`) would settle it; bs3 was taken
  by PC2's queue when this was found.
- **Fix (not implemented, pending the coordinator).**
  - At the prefill-insert free, keep M extra SWA tokens below the window, e.g. M = 128 =
    `SGLANG_SWA_EVICTION_INTERVAL`, the slack decode already keeps.
  - Put it behind a default-off `SGLANG_OPT_` switch. Freeing less is always safe.
  - Cost: ~128 more SWA tokens per idle session (~+10%).
  - Validation: CPU regression, C1 multi-turn exactness 12/12, and think30 x 32/48 vs SW3/SW1.

## 4. Open items (code; none implemented tonight)

0. **Second-turn window margin** (see above): ~10 lines, the largest single hit-rate lever found
   (~+0.13 ceiling in think-time mode).
1. **Exclusive host tiering for hybrid SWA (upstream-relevant).**
   - write_through mirrors the device, so distinct capacity is the host pool.
   - write_back is exclusive (+50% distinct here: 478K Full + 128K SWA tokens), but tombstoning an
     internal node's device SWA drops it with no backup (`swa.py:713-766`).
   - The fix: write_back plus an SWA backup when a device SWA node is tombstoned, and the
     load-back exactness check (PC3's `exactness_mt` 12/12 pattern).
   - Expected: both retentions x1.5 (SWA ~37 s, Full ~57 s at 48 sessions).
2. **Session-granular eviction for hybrid SWA.**
   - A multi-turn session is usable only while its tail window survives, so leaf-first eviction
     kills it after ~500 tokens and leaves its head as dead weight.
   - Options:
     - When a node's SWA window is evicted, also evict the Full ancestors that can no longer end a
       valid match, i.e. those with no SWA window left behind them.
     - Order host eviction by session (path) instead of by leaf.
3. **Stop writing dead SWA to host.** About a third of SWA host writes are dead on arrival:
   - **Decode-output leaves.** A default-off switch could skip caching output tokens when the chat
     template re-renders them (Gemma-4 drops the empty thought channel). ~10-15% of SWA host.
   - **Chunk-boundary windows.** Free a chunked prompt's boundary window once the next chunk is
     prefilled, unless it is still inside the final window. ~1K SWA tokens per boundary.
     `--chunked-prefill-size 2048` doubles these per prompt.
4. **A separate host split knob.**
   - `--hicache-size` splits by device bytes, so moving host bytes between Full and SWA also moves
     the device split, and the device SWA pool is what admits concurrent windows.
   - SW1 says the binding retention would gain only ~12% from rebalancing, so this is low priority.
5. **Harness.** Persist failed requests' `error` / `finish_reason`, so a single failure can be
   classified.
