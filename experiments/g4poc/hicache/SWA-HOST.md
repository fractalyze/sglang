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
