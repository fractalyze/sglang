# SGLang HiCache and hybrid-SWA prefix-cache bugs found in g4poc (for an upstream report)

Draft for an upstream issue/PR. Nothing here has been posted outside the fractalyze org; the user
decides whether and where to post.

**Setting.** Gemma-4-26B-A4B-it (30 layers: 25 sliding-window layers with window 1024, 5 full-attention
layers), compressed-tensors FP8 weights, FP8 E4M3 KV, Triton attention, one RTX 5090. SGLang fork at
fractalyze/sglang (base `ac6035c07`; unified radix cache, overlap scheduler on). HiCache:
`--enable-hierarchical-cache --hicache-size 12 --hicache-write-policy write_through
--hicache-io-backend kernel --hicache-mem-layout page_first`. Workload: multi-turn role-play, ~5K-token
prompts, up to 300-token replies, greedy or sampled, in-flight concurrency 20-32.

Fixes 1 and 2 are default-off switches on branches `jumanzii/g4poc-c-hicache` and `jumanzii/g4poc-final-hicache`; fix 3 extends fix 2's switch on a side branch:

| bug | switch | commit | test |
|---|---|---|---|
| 1. SWA admission under-reserves on load-back (scheduler crash) | `SGLANG_OPT_HICACHE_PIN_LOAD_BACK_WINDOW` | 9ce3db9471 | `test/registered/unit/managers/test_prefill_adder.py::TestHiCacheLoadBackSWAWindowPin` |
| 2. Write-through copies KV still being written (inexact load-back) | `SGLANG_OPT_HICACHE_FENCE_WRITE_THROUGH` | 19e850a228 | `test/registered/unit/mem_cache/test_hicache_write_fence.py` |
| 3. Retraction backup has the same race (PD decode only) | `SGLANG_OPT_HICACHE_FENCE_WRITE_THROUGH` | 7d0ba713da (branch `jumanzii/hicache-retraction-fence`) | same file, retraction cases |
| 4. A chat's second turn misses its whole prefix: one SWA window kept, the next turn matches 4 tokens short (any hybrid-SWA radix cache, HiCache or not) | `SGLANG_OPT_SWA_PREFILL_WINDOW_MARGIN` (int, 0 = off) | d93fea83df (branch `jumanzii/g4poc-swa-margin`) | `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py::TestSWAPrefillWindowMargin` |

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
- two shared-prefix prompt nodes (228 and 229 tokens) differed in **every** token of all 5 full
  layers; that is a separate effect of SWA tombstone recovery (see the open item below), not the race;
- load-back itself was clean: reloaded slots matched host, and every full-to-SWA mapping checked out.

**Mechanism.** Under the overlap scheduler, `process_batch_result` for step N runs after step N+1 was
launched. A request that finished at step N is still in batch N+1, whose forward writes the KV of its
last output token on `model_runner.forward_stream`; `cache_finished_req` inserts that token and the
write-through starts. `HiCacheController.start_writing` -> `L2TransferEngine.submit_device_to_host` makes the D2H stream
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
throughput cost of the fence, from single runs against the unfenced build at C20-C32, is -1.0% to
-1.8% output tok/s. That is consistent across points, so a small real cost cannot be excluded. The final
configuration with both fixes (`final-hc`) runs 942 output tok/s at C32 under a 10 s p90.

## 3. Retraction backups copy KV the overlap forward is still writing (PD decode)

**Where.** `UnifiedRadixCache.retraction_backup` backs a retracted decode request's KV up to the host
pool. It is reached from `release_req` only when `disaggregation_mode == "decode"` with
`--disaggregation-decode-retraction-backup host_pool`; with disaggregation off a retracted request is
released and recomputed, not backed up.

**Mechanism.** Same class as bug 2. Retraction happens while the next decode forward, which writes the
request's last token's KV, is still queued on the forward stream. `retraction_backup` calls
`l2_transfer_engine.submit_device_to_host` directly, bypassing `start_writing` and its fence, so the
backup can read that token half-written, and `retraction_restore` brings it back stale. The scheduler
also wired the forward stream into the cache controller only when `--enable-hierarchical-cache` was
set, while the host-pool retraction backup builds a controller without it.

**Fix (7d0ba713da).** `HiCacheController.fence_device_to_host()` holds the fence. `start_writing` and
`retraction_backup` both call it, under `SGLANG_OPT_HICACHE_FENCE_WRITE_THROUGH`. The scheduler wires
the forward stream into any cache controller. A CPU test checks that the backup waits on the forward
stream before it copies, and it fails without the fix. Not validated on GPU: the g4poc deployment does
not run PD decode, so a byte-level trace of this path was not made.

## 4. Every chat's second turn misses its prefix: the tree keeps exactly one SWA window

**Where.** Any hybrid-SWA model on the unified radix cache, with or without HiCache. Found with
the per-admission prefix log of `jumanzii/g4poc-swa-host-dbg` (see `SWA-HOST.md`).

**Symptom.** In think-time chat sessions, the turn after a session's first turn reuses none of its
cached prefix: with gaps under 30 s, 0 of 67, 0 of 93 and 0 of 92 such turns hit (runs at 32+40, 48+72 and 48+72 sessions; 0 of 122 at any gap, 9-95 s, in the first),
first prompts <= 4096 tokens. That holds even at 32 sessions, where every other returning turn with
a short gap hits ~100%. In each case the Full-layer prefix is fully present (`full_kv_hit_length` =
the expected prefix) and the usable match is the shared 13-token root. Second turns are 14% of prompt
tokens in this workload, so the hit-rate ceiling drops from ~0.80 to ~0.66.

**Mechanism.** Gemma-4's `sliding_window_size` is `sliding_window - 1` = 1023
(`models/gemma4_causal.py`); page size 1.
1. After a fresh prefill, `cache_unfinished_req` frees the request's out-of-window SWA slots before
   inserting: `free_swa_out_of_window_slots(req, pe - 1)` with threshold
   `pre_len - max(window, page)`. That leaves SWA live for exactly [pe - 1024, pe), pe being the prompt
   end, and the inserted prompt is an SWA tombstone below that.
2. Gemma-4's generation prompt ends `<|turn>model\n<|channel>thought\n<channel|>`, but the template
   renders a past assistant turn as `<|turn>model\n<reply><turn|>`. So the next turn's prompt matches
   the cached one only up to pe - 4. Any client that sends chat messages sees this.
3. `SWAComponent.create_match_validator` accepts a match end only with >= `sliding_window_size`
   contiguous SWA tokens behind it. Behind pe - 4 there are 1024 - 4 = 1020 < 1023, and the match
   falls back to the last valid node, the root.
4. After a turn that hit, the new extension sits on top of the previous window. That leaves ~1,300
   contiguous SWA tokens, so third and later turns are unaffected. A first prompt of 4097-5119 tokens
   is partly rescued by its chunk-boundary window: chunked prefill keeps SWA from 3072.

The same off-by-few affects any template whose rendered history differs from the generation prompt
in its last tokens, and any model with a window-1 convention.

**Fix (d93fea83df).** `SGLANG_OPT_SWA_PREFILL_WINDOW_MARGIN` (int, default 0 = unchanged): the
prefill-insert free keeps `window + margin` SWA tokens behind the insert end
(`SWAComponent._free_out_of_window_slots`). `_maybe_split_leaf_for_swa_lock` caps a fresh in-window
leaf at `window + margin` rather than one window. Otherwise the margin would sit in a separate parent
node that the window-bounded LRU refresh skips, and it would be evicted first. Decode-time eviction
is unchanged. Cost: `margin` extra SWA tokens per cached prompt (128 = `SGLANG_SWA_EVICTION_INTERVAL`
is ~10% of a 1,200-token SWA footprint per idle session).
- The test inserts a 24-token prompt (window 7) and matches a next turn cut 3 tokens short. It
  matches 21 tokens with margin 4 and 0 without.
- A margin keyed to the template's cut is a workaround. Alternatives:
  - make the validator accept a match end within a small distance of a longer valid window;
  - free out-of-window SWA only up to a page-aligned point that tolerates a short re-match.

**Validation.** GPU unit tests, C1 multi-turn exactness (standard and with the next turn cut 4 tokens
short) and think-time sessions at 32/48 are in `SWA-HOST.md` section 5.

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
