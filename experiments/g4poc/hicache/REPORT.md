# g4poc HiCache workstream (PC3): two SGLang fixes and the final configuration

Gemma-4-26B-A4B-it, official checkpoint with offline per-channel FP8 weights, one RTX 5090 (bs3),
multi-turn role-play (~5K-token prompts, up to 300-token replies), in-flight sweeps with the g4poc
gate. Predictions: [PREREG.md](PREREG.md). Upstream write-up: [UPSTREAM.md](UPSTREAM.md). Run records:
[runs/](runs/).

## Result

**`final-hc` is the study's final configuration.**
- Base: final-mem-c1-c2a (mem-final flags + C1 MoE config + C2-A sm120 FP8-KV extend tiles).
- HiCache flags: `--enable-hierarchical-cache --hicache-size 12 --hicache-write-policy write_through
  --hicache-io-backend kernel --hicache-mem-layout page_first`.
- Fix switches: `SGLANG_OPT_HICACHE_PIN_LOAD_BACK_WINDOW=1 SGLANG_OPT_HICACHE_FENCE_WRITE_THROUGH=1`.
- SGLang tree `a0491db764` on branch `jumanzii/g4poc-final-hicache`, which is 1425761173 + the two fix
  commits + a comment-only correction.
- Memory scope: 28G. SGLang keeps 10 GiB of cgroup headroom free beyond the 12 GB pinned host pool.

| point | p50 s | p90 s | output tok/s | prefix-cache hit | failed |
|---|---|---|---|---|---|
| C20 | 4.86 | 7.37 | 805 | 0.712 | 0 |
| C24 | 5.27 | 7.98 | 905 | 0.742 | 0 |
| C28 | 5.87 | 8.91 | 919 | 0.710 | 0 |
| C32 | 6.87 | 9.55 | **942** | 0.720 | 0 |
| C36 | 7.67 | 10.50 | 927 | 0.713 | 0 |
| C40 | 7.73 | 11.63 | 890 | 0.702 | 0 |
| C48 | 11.00 | 13.95 | 874 | 0.671 | 0 |
| C56 | 17.0 | 23.72 | 525 | 0.002 | 0 |

C20-C32: `sweep-final-hc-20261005-174836-build-server-3-583abd`. C40-C56:
`sweep-final-hc-20261005-183217-build-server-3-c4d30a`. Decode CUDA graphs are captured up to batch
48, so C56 decodes eagerly; this was left unchanged.

**Operating points.**
- **Recommended: C28.** 919 output tok/s at p90 8.91 s in the sweep, $0.212 per 1M output tokens at
  $0.70/GPU-h. This is pending PC2's C28 soak.
- **Sweep capacity and edge at a 10 s p90 SLO: C32.** 942 tok/s at p90 9.55 s, and 940 / 9.53 s in a
  replicate, $0.206 per 1M output. PC2's 30-minute soak at C32 gave p90 9.92 s and 899 tok/s, only
  0.08 s under the SLO, so C32 is the edge rather than an operating point.
- C36 misses the SLO (p90 10.50 s, 927 tok/s; `sweep-final-hc-20261005-191358-build-server-3-46c175`).
  At a 15 s SLO goodput still peaks at C32: C48 meets 15 s but delivers less (874 tok/s).
- The device-only final reaches C24 at 833 tok/s ($0.233).
- The hit rate holds above 0.67 through C48 and collapses at C56, where the host pool no longer holds
  the working set. Every point degrades without a failed request.

**Against the device-only final (final-mem-c1-c2a, bs3):** +8.6% output tok/s at C24 (p90 8.59 s →
7.98 s) and +49% at C28 (616 → 919 tok/s), where the device-only final's hit rate falls to 0.15.

**Exactness:** at concurrency 1, 4 role-play sessions × 3 turns, with greedy decoding and a 16K-token
device pool (so every later turn loads back from host), the outputs are 12/12 token-identical to the
device-only final with identical cached-token counts.
Runs: `exactmt-final-hc-smallpool-20261005-183035` vs `exactmt-final-mem-c1-c2a-20261005-181244`.

**Replicate** (`sweep-final-hc-20261005-185153-build-server-3-dcb048`): C24 910 tok/s at p90 7.97 s
(first run 905 / 7.98 s); C32 940 tok/s at p90 9.53 s (first run 942 / 9.55 s). C32 reproduced within
0.3%.

**Role-play quality and load-back (task b′).** PC2's final-hc role-play arm
(`rp-quality-final-hc-qr-20261005-185028`) showed three adherent→non-adherent language flips against
mem-qr-base, where the re-roll band is 1/80. The same arm was re-run on a log-only debug tree
(53614116f3, branch `jumanzii/g4poc-final-hicache-pfxdbg`). That tree records each admission's device
hit, its full/SWA host hit and load-back size, and whether any restored host slot was stale from SWA
tombstone recovery (`rp-quality-final-hc-qr-pfx-20261005-190542-build-server-3-38647d`, joined with
[prefix_split.py](prefix_split.py)).
- Language adherence was 68/80, equal to the baseline, with an NLL rise of 0.0018. The run passes.
- 48 of 80 items had a host load-back, and none restored a stale (tombstone-recovered) slot.
- s000135 and s001101 had SWA-only load-backs of the shared persona prefix and did not flip in this
  run.
- s000794 flipped (ja→zh) from its first token. Its first admission was a device hit with no load-back;
  it was retracted later and recomputed from a host-loaded prompt. So no flip in this run traces back to
  a HiCache load-back.

## Bug 1: prefill admission under-reserved sliding-window slots on a load-back (scheduler crash)

`mem-hc` (mem-final + HiCache) crashed the scheduler at C28 with an SWA out-of-memory in
`alloc_token_slots`; the device-only control retracts instead. A debug trace of SWA free/evictable
slots at each admission step (`sweep-mem-hc-dbg-20261005-152911-build-server-3-f9f0db`, same crash,
deterministic) showed the batch was a 2071-token chunk continuation plus a 247-token request with a host
hit. Admission pinned only the device-matched `last_node`. After `init_load_back` the request's
sliding-window lock is anchored at `best_match_node`, and that window covered 350 device-resident SWA
tokens the check had counted as evictable and nothing had charged. As a result `rem_swa` went from 1567
(1442 needed) to -225, and the allocation was 48 tokens short. Fix 9ce3db9471: also pin `best_match_node`
during the check. CPU regression `test_prefill_adder.py::TestHiCacheLoadBackSWAWindowPin` reproduces the
trace's numbers and fails without the pin.

`mem-hc-fix` sweep (`sweep-mem-hc-fix-20261005-160605`): C20 714, C24 797, C28 813, C32 835 tok/s,
with no failures.

## Bug 2: write-through copied KV the overlap forward was still writing (inexact load-back)

C1 multi-turn exactness with forced load-backs: 5/12. The same score holds on the base commit with
HiCache (`exactmt-mem-base-hc-smallpool-20261005-165058`). The controls are exact: the device-only A/A
and HiCache without load-backs both give 12/12, and the direct io / layer_first copy path gives 4/12.
A byte-level device/host comparison at every write and load ack (debug tree 9e26c90921) found one stale
token per finished turn in the host copy (full layer 0, SWA layers 5-24). Load-back and the full-to-SWA
mapping were clean. Mechanism: under the overlap scheduler `cache_finished_req` runs while the next
forward, which writes the request's last output token, is still queued on the forward stream.
`start_writing` ordered the D2H copy only after the scheduler stream. Load-back already had this fence
(`load_fence_stream`); write-through had none. Fix 19e850a228: wait on the forward stream before each
write submit. CPU regression `test_hicache_write_fence.py` fails without the fence. With the fence the
byte trace has no per-turn stale token and exactness is 12/12
(`exactmt-mem-hc-fix2-smallpool-20261005-174224`).

## Bug 3 (PD decode only): retraction backups bypass the fence

`UnifiedRadixCache.retraction_backup` (decode host-pool retraction backup, PD-disaggregated decode
only) calls `submit_device_to_host` directly, so it had no fence. The fix is on side branch
`jumanzii/hicache-retraction-fence` (7d0ba713da) with a CPU test. It is not in the final tree, because
this deployment runs with disaggregation off and never takes that path. See UPSTREAM.md, bug 3.

## Open items

- **Host copy kept on SWA tombstone recovery.** When a node whose SWA was evicted adopts a later
  request's FULL slots, it keeps its old host copy. Device and host then hold two separate
  computations of the same prefix, which differ by FP8 rounding. This matters only for bitwise
  reproducibility if such a node is reloaded. Proposed fix in UPSTREAM.md.
- **Upstream report.** UPSTREAM.md is a draft. Nothing has been posted outside the fractalyze org.

## Status log

- 10-05 18:50 KST: final-hc C20-C56 and exactness done.
- 10-05 19:12 KST: replicate and the b′ role-play run done.
- 10-05 19:52 KST: C36 done (misses 10 s). Fence-cost sweep queued (`memlogs/pc3-tail3.sh` on bs3).
