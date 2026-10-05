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
| C40 | 7.73 | 11.63 | 890 | 0.702 | 0 |
| C48 | 11.00 | 13.95 | 874 | 0.671 | 0 |
| C56 | 17.0 | 23.72 | 525 | 0.002 | 0 |

C20-C32: `sweep-final-hc-20261005-174836-build-server-3-583abd`. C40-C56:
`sweep-final-hc-20261005-183217-build-server-3-c4d30a`. Decode CUDA graphs are captured up to batch
48, so C56 decodes eagerly; this was left unchanged.

**Cost-optimal points.**
- At a 10 s p90 SLO: C32, 942 output tok/s, $0.206 per 1M output tokens at $0.70/GPU-h. The device-only
  final reaches C24 at 833 tok/s ($0.233).
- At a 15 s p90 SLO: still C32 by goodput. C48 also meets 15 s but delivers less (874 tok/s).
- Goodput peaks at C32. The hit rate holds above 0.67 through C48 and collapses at C56, where the host
  pool no longer holds the working set. Every point degrades without a failed request.

**Against the device-only final (final-mem-c1-c2a, bs3):** +8.6% output tok/s at C24 (p90 8.59 s →
7.98 s) and +49% at C28 (616 → 919 tok/s), where the device-only final's hit rate falls to 0.15.

**Exactness:** at concurrency 1, 4 role-play sessions × 3 turns, with greedy decoding and a 16K-token
device pool (so every later turn loads back from host), the outputs are 12/12 token-identical to the
device-only final with identical cached-token counts.
Runs: `exactmt-final-hc-smallpool-20261005-183035` vs `exactmt-final-mem-c1-c2a-20261005-181244`.

> [!gap] Replicate at C24/C32, the C36 point (10 s capacity between C32 and C40) and the fence-cost
> sweep of `mem-hc-fix2` against `mem-hc-fix` are running; see the status log at the end.

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

## Open items

- **Host copy kept on SWA tombstone recovery.** When a node whose SWA was evicted adopts a later
  request's FULL slots, it keeps its old host copy. Device and host then hold two separate
  computations of the same prefix, which differ by FP8 rounding. This matters only for bitwise
  reproducibility if such a node is reloaded. Proposed fix in UPSTREAM.md.
- **Upstream report.** UPSTREAM.md is a draft. Nothing has been posted outside the fractalyze org.

## Status log

- 10-05 18:50 KST: final-hc C20-C56 and exactness done. Replicate, C36 and the fence-cost sweep queued
  (`memlogs/pc3-tail.sh` on bs3).
