# Two SGLang HiCache bugs found in g4poc (for an upstream report)

Draft for an upstream issue/PR. Nothing here has been posted outside the fractalyze org; the user
decides whether and where to post.

**Setting.** Gemma-4-26B-A4B-it (30 layers: 25 sliding-window layers with window 1024, 5 full-attention
layers), compressed-tensors FP8 weights, FP8 E4M3 KV, Triton attention, one RTX 5090. SGLang fork at
fractalyze/sglang (base `ac6035c07`; unified radix cache, overlap scheduler on). HiCache:
`--enable-hierarchical-cache --hicache-size 12 --hicache-write-policy write_through
--hicache-io-backend kernel --hicache-mem-layout page_first`. Workload: multi-turn role-play, ~5K-token
prompts, up to 300-token replies, greedy or sampled, in-flight concurrency 20-32.

Both fixes are default-off switches on branch `jumanzii/g4poc-c-hicache`:

| bug | switch | commit | test |
|---|---|---|---|
| 1. SWA admission under-reserves on load-back (scheduler crash) | `SGLANG_OPT_HICACHE_PIN_LOAD_BACK_WINDOW` | 9ce3db9471 | `test/registered/unit/managers/test_prefill_adder.py::TestHiCacheLoadBackSWAWindowPin` |
| 2. Write-through copies KV still being written (inexact load-back) | `SGLANG_OPT_HICACHE_FENCE_WRITE_THROUGH` | 19e850a228 | `test/registered/unit/mem_cache/test_hicache_write_fence.py` |

## 1. Prefill admission under-reserves sliding-window slots on a HiCache load-back

**Symptom.** At 28 in-flight sessions the scheduler died:

```
alloc_for_extend -> alloc_token_slots: RuntimeError: Out of memory. Try to lower your batch size.
Try to allocate 2318 tokens.
Available full tokens: 7949 (full_available_size=4137 + full_evictable_size_=3812)
Available swa: 2270 (available_size=2270 + component_evictable_size_=0)
```

Every later request failed. Without HiCache the same overload retracts requests instead. The crash
replays deterministically with the same workload seed.

**Trace.** An env-gated log of SWA free / evictable / protected slots and the budget offset at each
admission step showed the batch was a 2071-token chunk continuation plus a 247-token request with a
host hit (5286 full tokens, 1019 SWA tokens to load):

| step | swa free | swa evictable | swa offset | rem_swa |
|---|---|---|---|---|
| chunk admitted | 3131 | 508 | 2072 | 1567 |
| load-back request selected (needs 1442) | 3131 | 508 | 2072 | 1567 |
| after `init_load_back` | 2112 | **158** | 2072 | 198 |
| after commit | 2112 | 158 | 2495 | **-225** |

The load-back consumed the 1019 SWA tokens it was charged for, plus 350 evictable tokens it was not.

**Mechanism.** `PrefillAdder.add_one_req` pins only `req.last_node`, the end of the *device* match,
while it checks the budget. After `init_load_back` the request's sliding-window lock is anchored at
`req.best_match_node`. That window covered the host-only SWA segment (charged as
`swa_host_hit_length`) **and two device-resident SWA nodes above it (169 + 181 = 350 tokens)**. The check
counted those as evictable; the load-back locked them.

**Fix.** Also pin `best_match_node` during the admission check (same scoped lock as the `last_node`
pin, released on every exit path), so the device SWA its post-load window will lock leaves the
evictable budget before the check. An over-committing candidate then waits. The full-attention side
needs no change: its charge already bills every token past the device match as a new allocation.

**Result.** `mem-hc-fix`, in-flight sweep, no failure at any point:

| point | p90 s | output tok/s | prefix-cache hit |
|---|---|---|---|
| C20 | 8.42 | 714 | 0.709 |
| C24 | 9.11 | 797 | 0.742 |
| C28 | 10.09 | 813 | 0.713 |
| C32 | 10.79 | 835 | 0.721 |

The device-only control reached 468 tok/s at C28 (hit rate 0.114, 5 retractions).

## 2. Write-through copies KV that the overlap scheduler's forward is still writing

**Symptom.** Multi-turn greedy outputs at concurrency 1 differ between HiCache and a device-only
control whenever a later turn's prefix is loaded back from host. 4 role-play sessions x 3 turns, on a
16K-token device pool so that every later turn is a load-back: 5/12 identical. All 4 first turns
match; 7 of the 8 later turns diverge within their first 0-15 tokens, with identical `cached_tokens`.
The text stays fluent but loses persona and language (German in-character replies turn into
English first-person). The same holds on the unmodified base commit (5/12). Controls: a device-only
A/A is 12/12, and HiCache on a pool large enough that nothing is loaded back is 12/12.

**Byte trace.** A debug hook compared device and host bytes per token and per layer, for the
full-attention and sliding pools, at every write-through ack and every load-back ack:

- every finished turn's output node (128 tokens) had exactly **one** token whose host bytes differed
  from device, in full layer 0 and a contiguous run of sliding layers 5-24;
- two prompt nodes (228 and 229 tokens, stashed as chunked prefills) were stale in **every** token of
  all 5 full layers;
- load-back itself was clean: reloaded slots matched host, and every full-to-SWA mapping checked out.

**Mechanism.** Under the overlap scheduler, `process_batch_result` for step N runs after step N+1 was
launched. A request that finished at step N is still in batch N+1, whose forward writes the KV of its
last output token on `model_runner.forward_stream`; `cache_finished_req` inserts that token and the
write-through starts. Likewise a chunked prefill is stashed (inserted) before its chunk's forward has
run. `HiCacheController.start_writing` -> `L2TransferEngine.submit_device_to_host` makes the D2H stream
wait only on an event recorded on the scheduler's current stream, never on the forward stream, so the
copy can read the KV half-written. Load-back already has the equivalent fence
(`load_fence_stream = forward_stream`, set by the scheduler, used in `start_loading`); write-through
had none. The stale token sits at the end of the previous reply, inside the next turn's sliding
window, which is why the next turn diverges immediately.

**Fix.** Before each write submit, `device_to_host_stream.wait_stream(forward_stream)`.
`wait_stream` orders the copy after the work queued so far only, which at `cache_finished_req` time is
exactly the in-flight forward writing the node; later forwards still overlap with the copy.

**Result.** With the fence, the same 4 x 3 load-back run is **12/12** identical to the device-only
control, with identical `cached_tokens` (`exactmt-mem-hc-fix2-smallpool-rt-20261005-165852-...-5d434c`).
All turn-0 prompts are chunked (4096 + rest), so stashed chunk nodes are covered. In the byte trace,
every per-turn stale token is gone and every full-to-SWA mapping check passes. The in-flight
throughput cost of the fence is measured separately (sweep of `mem-hc-fix2` against `mem-hc-fix`).

## Open item: host copy kept when a node adopts a later request's FULL slots

The fenced byte trace still shows two shared-prefix prompt nodes (228 and 229 tokens) whose FULL host
bytes differ from device in every token of all full layers, while their SWA bytes match. This is not
the race, and the fence does not change it. `SWAComponent.update_component_on_insert_overlap`, branch 1
("recover an SWA tombstone"), makes a node whose SWA was evicted adopt an incoming request's FULL
slots and the SWA rebuilt from them, then frees the old FULL slots. The node's FULL host copy, written
from the first request's computation, is kept. Device and host then hold two separate computations of
the same prefix: equivalent KV whose FP8 bytes differ by the rounding of a different extend layout.
The difference is numerics-level, not corruption, and only matters for bitwise reproducibility if such
a node is evicted and loaded back. Proposed fix: when a node adopts new FULL slots, drop its FULL host
copy (and its SWA host copy, if any) so a later backup rewrites it, or re-issue the backup into the
existing host slots.

## Reproduction notes

- The admission crash needs enough concurrency that the SWA pool runs out while host hits are
  common (here C28 with a 43K-token SWA pool).
- The inexactness needs only a load-back at concurrency 1: any device pool smaller than the sessions'
  total, turn-major order, greedy decoding.
- Bug 2 reproduces on the base commit with none of the study's flags. Bug 1 was observed on the
  study's final flag stack. Between the base commit and that tree, the scheduler and cache code
  differ only by an opt-in slid-window lock release (`SGLANG_OPT_SWA_RELEASE_SLID_WINDOW`, off in
  that run), but a base-commit crash run was not made.
