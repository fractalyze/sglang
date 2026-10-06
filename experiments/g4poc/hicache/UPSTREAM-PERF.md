# SGLang performance improvements found in g4poc (upstream candidates, not bugs)

Draft notes. Nothing here has been posted outside the fractalyze org; the user decides whether and where to post.
Correctness bugs are in `UPSTREAM.md`.

## 1. Two host syncs per decode step in the Triton backend's sliding-window replay

**Where.** `python/sglang/srt/layers/attention/triton_backend.py`, `update_sliding_window_buffer`, on the CUDA-graph
decode path (`_update_decode_kv_buffers`, and the target-verify path) when the KV pool is a static SWA pool (the
translator does not translate reads, `BaseSWAKVPool`). The window ids are mapped to SWA ids with

```python
kv_last_index = window_kv_indptr[-1]
window_kv_indices[:kv_last_index] = token_to_kv_pool.translate_loc_from_full_to_swa(window_kv_indices[:kv_last_index])
```

Both slices read the GPU scalar `kv_last_index` on the host (`aten::item` -> `cudaStreamSynchronize`), so the scheduler's
`run_batch` for step k+1 waits until graph k has finished, and the GPU idles while the host finishes the replay prep and
launches the next graph. Upstream main (21c9bbdf2c, 2026-10-06) has the same code. It applies to every hybrid
sliding-window model served on the Triton backend with CUDA graphs and a static SWA pool (Gemma-3/4 and similar).

**Fix.** Bound the translate on the host by `bs * sliding_window_size` (no request holds more window ids) and confine it
to the filled ids with a device-side mask (`torch.arange(n) < window_kv_indptr[bs]`); masked lanes gather index 0 and are
written back unchanged. Same ids, no host read. Implemented behind `SGLANG_OPT_SWA_DECODE_NO_HOST_SYNC` on fractalyze
`jumanzii/g4poc-r2-k2` (fae2c5cdca), with a CPU unit test
(`test/registered/unit/layers/attention/test_triton_swa_window_no_host_sync.py`: same ids as the synced path, stale tail
untouched, no tensor read on the host).

**Measured** (Gemma-4-26B-A4B-it FP8, RTX 5090, `experiments/g4poc/COMPUTE.md` section 7): exact (greedy decode
identical, multi-turn HiCache exactness 12/12). The profile's pre-graph gap falls from ~0.5 to ~0.06-0.11 ms per decode
step; untraced, the decode step falls 0.18-0.29 ms (1.7-1.9%), and end-to-end throughput rises 0.4-0.6% at 12-28
requests in flight (p90 -0.6 .. -1.2%). The gain grows where the host is slower or the decode step shorter.
