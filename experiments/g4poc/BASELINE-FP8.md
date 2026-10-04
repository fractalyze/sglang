# g4poc baseline: official Gemma-4-26B-A4B-it in FP8 on one RTX 5090 (bs2), 2026-10-04

Scope: the FP8 baseline of `google/gemma-4-26B-A4B-it` (revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`)
served by SGLang on build-server-2's RTX 5090. It covers the checkpoint format, what each op runs on,
where the 32 GB goes, how many 5K-token sessions fit, and which memory levers are left.
Code: `baseline/` (converter, launch wrapper, concurrency probe, quality anchor). Raw run records:
`baseline/runs/<run>/` (launch command and tree commit, key server log lines, probe JSON, quality JSON,
host-memory summary, kernel table).

Unit note: SGLang logs memory as "GB" but computes bytes / 2^30. This file writes GiB for those
numbers; per-token byte counts are exact bytes.

## 1. Result in one table

All runs: text-only FP8 checkpoint, FP8 E4M3 KV, Triton attention, `--context-length 16384`,
`--cuda-graph-max-bs-decode 32`, chunked prefill 4096 (SGLang default here). Probe: N simultaneous
requests of 5,000 random-token prompts (no shared prefix) decoding exactly 300 tokens.

| run | config delta | full pool (tokens) | sliding pool (tokens) | KV GiB | max 5K/300 sessions in flight | retractions |
|---|---|---|---|---|---|---|
| r00 | SGLang defaults (auto `--mem-fraction-static` = 0.778) | - | - | - | **does not start**: "Loaded weights leave no GPU memory for the KV cache" | - |
| r01 | `--mem-fraction-static 0.90` | 29,061 | 23,248 (ratio 0.8 default) | 2.50 | **5** (full pool 91%, sliding 34%) | 0 |
| r02 | r01 + `--swa-full-tokens-ratio 0.3` | 65,387 | 19,616 | 2.50 | **12** (full 97%, sliding 91%) | 0 |
| r03 | r02 + `--disable-prefill-cuda-graph`, `--mem-fraction-static 0.93` | 89,571 | 26,871 | 3.42 | **16** clean; 17 peak with 1 retraction (full 99%, sliding 89%) | 1 at 17 |

r03 is the recommended baseline config for the study: 3.2x the sessions of the first config that
starts, using only server flags.

## 2. Checkpoint format decision

**Chosen: offline compressed-tensors FP8, per-output-channel weight scales, dynamic per-token
activation scales** (the scheme of `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic`), produced here from the
official BF16 weights by `baseline/make_fp8_ckpt.py` (min-max per channel, no calibration data).

| option | verdict | why |
|---|---|---|
| SGLang online `--quantization fp8` from BF16 | rejected | `Fp8LinearMethod`/`Fp8MoEMethod` allocate BF16 parameters and quantize in `process_weights_after_loading`, after all weights are on the GPU: the load peak is the full 51.6 GB BF16 model (48 GiB), over 32 GB. No streaming per-layer path in this tree. |
| Offline 128x128 block FP8 (DeepSeek style) | rejected | `moe_intermediate_size` 704 and dense `intermediate_size` 2112 are not multiples of 128. |
| Offline per-tensor FP8 (modelopt) | not chosen | Needs static activation scales (calibration). On SM120 the per-tensor dense path is cuDNN/nvjet, but the MoE path for `Fp8MoEMethod` is still Triton (FlashInfer CUTLASS has no FP8 MoE path here and falls back to Triton). |
| **Offline per-channel W8A8 dynamic (compressed-tensors)** | **chosen** | Calibration-free; loads through SGLang's maintained `CompressedTensorsW8A8Fp8(MoE)` path; same scheme as RedHat's published checkpoint (identical FP8 tensor count, 11,725). |

Checkpoint facts:
- Quantized: q/k/v/o, dense MLP gate/up/down, all 128 experts' gate/up/down in every layer, split per
  expert (`experts.<e>.{gate,up,down}_proj`). Kept BF16: embeddings (tied LM head), `router.proj`,
  norms, scalars, vision tower.
- Weight round-trip error: relative Frobenius error 2.6% on sampled expert and attention weights
  (E4M3, 3 mantissa bits).
- Two model dirs share the shards: `text/` (`Gemma4ForCausalLM`, no vision tower is built or loaded)
  and `mm/` (`Gemma4ForConditionalGeneration`). On disk: text 24.2 GiB, vision shard 1.07 GiB.
- Produced twice, byte-identical: on bs3 from the local BF16 copy (17 s) and on bs2 by range-reading the
  pinned Hub revision (26 min; bs2 has no disk for the 51.6 GB BF16 copy, and bs3 cannot reach bs2 over
  SSH under the Tailscale policy). All 8 shard sha256 match.
- Locations: bs3 `/data/jooman/g4poc/models/gemma-4-26B-A4B-it-{bf16,fp8ch}`; bs2
  `/data/jooman/g4poc/models/gemma-4-26B-A4B-it-fp8ch` (bs2 `/data` now ~11 GiB free).

**SGLang gap found (text-only path):** SGLang rewrites Gemma-4's attention dims (sliding = base attrs,
full = `global_*`, into SGLang's base = full, `swa_*` = sliding) only for `model_type == "gemma4"`
(`utils/hf_transformers/config.py`). A `gemma4_text` config therefore loads 512-wide full-attention
weights into 256-wide params and fails. The converter writes the text config already in SGLang's
convention (`sglang_text_config`), so no SGLang change was needed. An upstream fix is one tuple entry.

## 3. What runs each op on SM120 (r03 profile)

`baseline/runs/r03-*/kernels.txt`: 12 profiled steps with 8 concurrent 5,000-token prompts plus 40
decode tokens (prefill-heavy), share of GPU kernel time:

| op | kernel / backend | share |
|---|---|---|
| attention, prefill/extend | Triton `_fwd_kernel` (`--attention-backend triton`, the Gemma-4 default on non-SM100) | 49.8% |
| attention, decode | Triton `_fwd_grouped_kernel_stage1` (split-KV) | 0.1% (decode-light profile) |
| MoE experts | Triton `fused_moe_kernel`, FP8 W8A8 per-channel; **default config** ("Config file not found ... E=128,N=704,device_name=NVIDIA_GeForce_RTX_5090,dtype=fp8_w8a8,per_channel_quant=True") | 20.9% (+1.9% `moe_sum_reduce`, routing/align <0.5%) |
| dense FP8 linears (qkv, o, dense MLP) | sgl-kernel CUTLASS `fp8_scaled_mm`, `MainloopSm120TmaWarpSpecialized` (native SM120) | 19.2% |
| activation quantization | `per_token_quant_fp8` (dynamic per-token) | 1.2% |
| BF16 router, LM head | cuBLAS SM80 fallbacks (`cutlass_80_wmma_tensorop_bf16`, `cutlass_80_tensorop_bf16`, gemv) | ~1% |
| norms, RoPE, KV store | FlashInfer CuTe-DSL RMSNorm, sglang fused RoPE, `store_kvcache`, Gemma qkv-RMSNorm | ~4% |

So at 5K prompts, prefill attention is half the GPU time. That is a throughput lever, not a memory one,
and outside this task's scope. The MoE has no tuned Triton config for the 5090.

## 4. Memory breakdown (GiB of 31.84 GiB = 32,607 MiB)

| component | r01 (0.90, ratio 0.8) | r03 (0.93, ratio 0.3, no prefill graph) | source |
|---|---|---|---|
| CUDA context, torch, NCCL (before load) | 1.16 | 1.16 | 31.84 - "Load weight begin. avail mem=30.68" |
| weights, total | **25.12** | **25.12** | "Load weight end ... mem usage=25.12" |
| - routed experts FP8 (30 x 128 x (1408x2816 + 2816x704) B) | 21.27 | 21.27 | analytic |
| - attention FP8 (25 sliding + 5 full layers) | 1.05 | 1.05 | analytic |
| - dense MLP FP8 | 0.50 | 0.50 | analytic |
| - embeddings = tied LM head, BF16 (262144 x 2816 x 2 B) | 1.37 | 1.37 | analytic; SGLang aliases `lm_head` to `embed_tokens` |
| - routers, norms, scales | 0.06 | 0.06 | analytic |
| - unattributed | ~0.87 | ~0.87 | measured minus analytic; suspect RoPE cos/sin caches built for `max_position_embeddings` 262,144 (estimate, not verified) |
| vision tower + embedder (not loaded in text-only) | (1.10) | (1.10) | r04: mm dir loads 26.22 GiB |
| KV, full layers (K + V) | 0.28 (0.14 + 0.14) | 0.86 (0.43 + 0.43) | "Full KV Cache is allocated" |
| KV, sliding layers (K + V) | 2.22 (1.11 + 1.11) | 2.56 (1.28 + 1.28) | "SWA KV Cache is allocated" |
| pool metadata (req_to_token etc.) | 0.17 | 0.25 | Load-weight avail - KV - "Memory pool end avail" |
| prefill CUDA graphs (breakable, 50 token sizes up to 4096) | 1.12 | 0 (disabled) | "Capture target prefill CUDA graph end ... mem usage" |
| decode CUDA graphs (bs 1..32) | 0.04 | 0.13 | "Capture target decode CUDA graph end" |
| left for activations / workspace (outside the static pool) | 1.70 | 1.72 | "available_gpu_mem" after capture |

Host side (hostwatch, every run): peak process-tree RSS 6.4-8.0 GB during load, 6.7-7.7 GB serving;
load1 <= 1.1; zero compiler processes (no FlashInfer/CUTLASS JIT on this FP8 path; Triton compiles
in-process). The 24G systemd cap was never approached. The FP8 converter peaked at 10 GB RSS on bs2
(streaming).

## 5. KV accounting checked against the allocator

Per-token bytes, FP8 E4M3 (1 B/element):

| pool | expected | allocator (r03) | verdict |
|---|---|---|---|
| full, K | 5 layers x 2 heads x 512 = 5,120 B | 0.43 GiB / 89,571 = 5,155 B | match |
| full, V | 0 if V aliased K (`attention_k_eq_v`), else 5,120 B | 0.43 GiB, same as K | **V is stored separately: full-layer K is stored twice** (5,120 B/token wasted) |
| sliding, K + V | 25 layers x 8 heads x (256 + 256) = 102,400 B | 2 x 1.28 GiB / 26,871 = 102,300 B | match |

What the sliding pool actually reserves per request:
- Pool size = `swa_full_tokens_ratio` x full-pool tokens (default ratio 0.8), not a per-session window
  cap. With 5K-token sessions the default gives the sliding pool 89% of the KV bytes while the full
  pool runs out first (r01: full 91% used, sliding 34%).
- Held per decoding request (probe peaks): 1,570 (r01), 1,485 (r02), 1,414 (r03) sliding tokens per
  running request, against a 1,024-token window. The excess is `SGLANG_SWA_EVICTION_INTERVAL` = 128
  (out-of-window tokens are freed only after 128 have accumulated), page rounding, and prefill-time
  slots of the chunked 5,000-token prompt that are released only once decode starts evicting.
- Full pool per request: 5,281 tokens at peak (5,000 prompt + ~281 decoded).

Per 5K-in / 300-out session in r03: full 5,281 x 10,240 B = 54 MB; sliding ~1,414 x 102,400 B =
145 MB; **~199 MB per session** (the planning estimate of ~156 MB assumed a sliding cost of exactly
1,024 tokens). Sliding KV is 73% of a session's KV.

## 6. Memory levers (estimates unless marked measured)

Session cost ~199 MB (0.185 GiB) at r03. "Sessions" means concurrent 5K/300 requests in flight.
Gains are relative to r03 (16 clean) unless noted. They are not additive past a pool balance:
re-balance `--swa-full-tokens-ratio` after each change.

| # | lever | kind | memory effect | sessions | status / risk |
|---|---|---|---|---|---|
| L1 | `--swa-full-tokens-ratio` 0.8 -> 0.3 | flag | moves 1.2 GiB from the idle sliding pool to the full pool | 5 -> 12 | **measured** (r01 -> r02) |
| L2 | `--disable-prefill-cuda-graph` + `--mem-fraction-static` 0.90 -> 0.93 | flag | +0.92 GiB KV | 12 -> 16 | **measured** (r02 -> r03); prefill latency effect not measured yet |
| L3 | text-only checkpoint (no vision tower) | checkpoint | 1.10 GiB | +5-6 vs the mm checkpoint | **measured** weight delta (r04); already in r01-r03 |
| L4 | fine-tune the ratio to the measured mix (1,414 / 5,281 = 0.27) | flag | balances pools (r03 left 11% of sliding unused) | +1 | estimate |
| L5 | tighter sliding eviction: `SGLANG_SWA_EVICTION_INTERVAL` 128 -> 16-32, free out-of-window prompt slots at prefill end | env + small code | ~1,414 -> ~1,100 sliding tokens/session, ~32 MB/session | +2-3 | estimate; interval is config-only, prefill-end freeing needs code; more frequent eviction calls cost CPU |
| L6 | store full-layer V as an alias of K (`attention_k_eq_v`) | SGLang code (default-off switch) | 5,120 B/token = 27 MB/session; 0.43 GiB of the r03 pool | +2-3 | estimate; touches pool + Triton attention V reads |
| L7 | use the 1.72 GiB activation slack: `--mem-fraction-static` ~0.95, possibly with `--chunked-prefill-size 2048` to cap the activation peak | flag | +0.6 GiB KV | +3 | estimate; needs a measured activation peak at the workload's longest prompt (10K) |
| L8 | FP8 embedding + LM head (row-scaled; gather-dequant for lookup) | SGLang code | 0.69 GiB | +3-4 | estimate; logit quality must pass the gate |
| L9 | size RoPE caches to `--context-length` instead of 262,144 positions | SGLang code | up to ~0.4 GiB (unverified part of the 0.87 GiB residual) | +2 | estimate; verify the residual first |
| L10 | cap `--max-running-requests` (2,799 -> 64) | flag | req_to_token 2,799 x 16,384 x 4 B = 0.17 GiB | +1 | estimate |
| L11 | FP4 (NVFP4) KV cache | SGLang kernels + quality gate | ~-47% KV bytes/session | 16 -> ~30 | estimate; SM120 Triton-attention support unchecked; quality risk on long context |

Rough stack: flags and env only (L4, L5-interval, L7, L10) ~16 -> ~21; plus the code levers L5-full,
L6, L8, L9 ~26-28; plus L11 roughly x1.9 on top. Each step must be gated (quality anchor below)
and measured with the probe, because sliding/full balance shifts with every change.

Workload levers that are not single-request memory (PB's scope; noted for completeness): cross-turn
prefix reuse in the radix cache (the full-layer KV of earlier turns survives; out-of-window sliding KV
does not), host offload of idle sessions' KV between turns (HiCache), and prefill/decode split. At the
fleet's ~2,200 sessions, the number that matters is concurrently *active* requests, which multi-turn
think time keeps well below the session count.

## 7. Quality anchor

The official BF16 model (51.6 GB) cannot run on one 32 GB RTX 5090. So **there is no on-hardware
BF16 reference: the FP8 checkpoint's own scores are the study's quality baseline**, and later
candidates are judged by paired comparison against them (`gate quality-compare`), not against BF16.
The FP8-vs-BF16 gap itself is unmeasured here (it would need two GPUs, a larger GPU, or a CPU run that
bs3's 60 GB RAM cannot hold next to 51.6 GB of weights).

`baseline/quality_anchor.py` against r01 (pool sizing does not change numerics), greedy:

| set | n | result |
|---|---|---|
| GSM8K test (gemma4nv gate prompt, `####` answer) | 1,319 (full split) | **96.21%** |
| tool-call JSON (gate set) | 40 | **100.0%** |
| multilingual role-play sanity (ko x2, ja, zh, en x2, es, pt, fr, de, id, th, vi, ru, ar, hi; persona system prompt + 1-2 earlier turns) | 16 | **16 / 16 pass**: reply in the user's language, 4-gram repetition < 0.2 (max 0.009), terminated before 300 tokens (90-190 tokens) |

Outputs are in `baseline/runs/r01-m090/quality.json` for human review; persona and tone were in
character on a skim of all 16. The role-play checks are sanity gates (language, loops, termination),
not a quality score.

## 8. Reproduce

```bash
# bs3 (CPU) or bs2 (streaming from the Hub)
python baseline/make_fp8_ckpt.py --src <bf16 dir or meta-only dir> --out <out> \
    [--remote-repo google/gemma-4-26B-A4B-it --revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52]
# bs2: server under host.lock + systemd MemoryMax=24G + watchdog (queues behind other engines)
baseline/serve.sh r03 <out>/text --context-length 16384 --mem-fraction-static 0.93 \
    --swa-full-tokens-ratio 0.3 --disable-prefill-cuda-graph
python baseline/probe.py --n 32 --input-len 5000 --output-len 300 --log runs/r03/server.log
PYTHONPATH=<tree>/experiments/gemma4-nvfp4-5090 python baseline/quality_anchor.py --model <out>/text --out q.json
```

Tests (CPU): `python baseline/test_make_fp8_ckpt.py` (6), `PYTHONPATH=<gate> python
baseline/test_quality_anchor.py` (12).

Notes for whoever runs next:
- Stop the server with `pkill -f "^[^ ]*python -m sglang.launch_server"`. A looser pattern also
  matches the hostwatch wrapper's argv and kills it before it writes `hostmem.summary.json` (r01 lost
  its summary that way; its peaks come from `hostmem.csv`).
- hostwatch's phase regex predates this tree's "Capture target ... CUDA graph" log text, so graph
  capture is counted inside the `weight_load` phase.
- bs3 was rebooted at 14:11 KST by another session (`nvidia-smi -r` then `systemctl reboot`, from the
  journal); its GPU now enumerates, but this task used bs3 for CPU/disk only.
- World model: `wm consult` was a cold start for this model (no measured trials); no trial was
  preregistered because this task only establishes the baseline.
