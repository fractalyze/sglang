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
