# Gemma-4-26B-A4B-it FP8 on one RTX 5090: deployment guide (g4poc ship branch)

This branch (`jumanzii/g4poc-ship` on fractalyze/sglang) is upstream SGLang at
[`a9871012ac`](https://github.com/sgl-project/sglang/commit/a9871012acb768dc94a43a6542cc32626c7b7b0b) (sgl-project main,
2026-09-21) plus nine lever commits and this guide. It packages what the g4poc study (2026-10-04 to 10-06) measured and
adopted for one workload:

- **Model:** `google/gemma-4-26B-A4B-it`, the official checkpoint (revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`),
  converted offline to FP8: compressed-tensors, E4M3 weights with one scale per output channel, dynamic per-token
  activations, text-only (section 2).
- **Hardware:** one RTX 5090 (32 GB, SM120) per server.
- **Traffic:** non-streaming, multi-turn, multilingual role-play: ~5K-token prompts (at most 10K), replies of at most
  300 tokens, every turn resending the whole history.

All costs are per 1M **output** tokens at an illustrative $0.70 per GPU-hour.

## 1. TL;DR

| traffic | config (section 3) | operating point, one RTX 5090 | measured |
|---|---|---|---|
| requests always in flight | (a) in-flight: HiCache + chunk 2048 + lpm + all levers | **28 in flight** | 30-min soak: p90 7.33 s, p99 8.95 s, 1,127 output tok/s, **$0.173**, 0 failed, 0.87% retracted |
| same, 6 s p90 SLO | (b) = (a) | **16 in flight** | 240 s point: p90 5.05 s, 941 tok/s, **$0.207** |
| chat, mean think time >= 30 s, ~12 GB host RAM per GPU | (c) chat: device prefix cache only, default chunking | **~88 live sessions per GPU** at a 10 s p90 SLO | 84.4 live sessions meet it (p90 8.77 s), 97.7 miss (11.02 s); **~$0.36** |

Against the study's FP8 baseline ($0.343 at 12 in flight, 10 s SLO), the in-flight config costs about half. Against
round 1 of the study (the same configs without the decode-glue fusion and the dense FP8 tiles), in-flight cost falls
15% ($0.204 -> $0.173) and chat capacity rises from ~70 to ~88 sessions per GPU.

Every lever is an opt-in switch, default off, except two tuned config files that apply on an RTX 5090 by default
(section 4). The switches are set in the launch commands below.

## 2. Prepare the FP8 checkpoint

SGLang's online `--quantization fp8` loads the full 51.6 GB BF16 model onto the GPU first, which does not fit 32 GB,
and 128x128 block FP8 does not fit the shapes (`moe_intermediate_size` 704, `intermediate_size` 2112). The study
converts offline, on CPU, with
[`experiments/g4poc/baseline/make_fp8_ckpt.py`](https://github.com/fractalyze/sglang/blob/jumanzii/g4poc/experiments/g4poc/baseline/make_fp8_ckpt.py)
on the `jumanzii/g4poc` branch:

```bash
git clone --branch jumanzii/g4poc --depth 1 https://github.com/fractalyze/sglang.git g4poc-harness
huggingface-cli download google/gemma-4-26B-A4B-it \
    --revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 --local-dir <bf16-dir>
python g4poc-harness/experiments/g4poc/baseline/make_fp8_ckpt.py --src <bf16-dir> --out <fp8-dir>
# Without disk for the BF16 copy: put only config, index and tokenizer files in <meta-dir> and range-read the tensors:
#   --src <meta-dir> --remote-repo google/gemma-4-26B-A4B-it --revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52
```

- **Output:** `<fp8-dir>/text` (`Gemma4ForCausalLM`, no vision tower; **serve this one**) and `<fp8-dir>/mm`
  (`Gemma4ForConditionalGeneration`), sharing the shards. The text dir is 24.2 GiB on disk.
- **Format:** compressed-tensors `float-quantized` (the scheme of `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic`). Weights
  are E4M3 with a BF16 per-output-channel min-max scale and no calibration data; every language-model linear is
  quantized (q/k/v/o, dense MLP, all 128 experts). Embeddings (tied LM head), `router.proj`, norms and the vision
  tower stay BF16.
- **Text config:** the converter writes it in SGLang's attention-dim convention (SGLang remaps Gemma-4's sliding and
  full attention dims only for `model_type == "gemma4"`), so no SGLang change is needed for the text-only dir.
- **Reproducible:** the study produced it twice, from a local BF16 copy and by range-reading the Hub; all 8 shard
  sha256 matched.
- Needs `torch` and `safetensors` (CPU only).

## 3. Launch commands

Every command below is exactly what the study measured, with the flags deduplicated (the study's refs stacked some
flags twice; argparse keeps the last). The `systemd-run` scope caps the server's host memory the way the study did;
any equivalent cap works (a container memory limit, a systemd service's `MemoryMax`).

Three settings apply to every config:

- **Attention backend:** Triton. SGLang picks it for Gemma-4 on SM120, so no flag sets it.
- **`SGLANG_LOGPROB_CHUNK_SIZE=128`:** only clients that request input logprobs need it. At `--mem-fraction-static
  0.955` SGLang's default 2,048-row chunk of fp32 logits over the 262,144-token vocabulary does not fit. The study set
  it on every server; it does not touch generation.
- **`--port` and `--decode-log-interval 1`:** the study's harness passed both for its own measurement. They are not
  needed in production.

### (a) In-flight traffic: requests always in flight, operating point 28 per GPU

```bash
# HiCache's start check counts the server cgroup's page cache: read the weights from outside it first.
cat <fp8-dir>/text/*.safetensors > /dev/null

SGLANG_OPT_GEMMA4_FP8_VOCAB_TABLE=1 \
SGLANG_OPT_TRITON_EXTEND_SM120_FP8_KV_TILES=1 \
SGLANG_OPT_GEMMA4_FUSED_GLUE=2 \
SGLANG_OPT_HICACHE_PIN_LOAD_BACK_WINDOW=1 \
SGLANG_OPT_HICACHE_FENCE_WRITE_THROUGH=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
SGLANG_LOGPROB_CHUNK_SIZE=128 \
systemd-run --user --scope -p MemoryMax=32G -p MemorySwapMax=0 \
python -m sglang.launch_server --model-path <fp8-dir>/text \
  --kv-cache-dtype fp8_e4m3 --context-length 16384 \
  --json-model-override-args '{"max_position_embeddings": 16384}' \
  --disable-prefill-cuda-graph --max-running-requests 64 \
  --mem-fraction-static 0.955 --swa-full-tokens-ratio 0.268 --cuda-graph-max-bs-decode 48 \
  --enable-hierarchical-cache --hicache-size 12 --hicache-write-policy write_through \
  --hicache-io-backend kernel --hicache-mem-layout page_first \
  --chunked-prefill-size 2048 --schedule-policy lpm
```

- **Memory scope:** the study ran this config in a 28 GB scope (`MemoryMax=28G`). 32 GB or more is recommended;
  see the HiCache start check in section 6.
- **Host RAM:** the server pins a 12 GB host pool and peaks at ~18 GB RSS.
- **Admission:** cap admission at **28 requests in flight per GPU** in the router or client. SGLang's
  `--max-running-requests 64` sizes its request table (L10 below); it does not set the operating point.
- **Routing:** route a session's turns to the same server (sticky routing), so its history stays in that server's
  prefix cache.

### (b) 6 s p90 SLO: the same server at 16 in flight

Launch exactly as (a) and cap admission at **16 requests in flight per GPU**. At 20 in flight p90 sits on the 6 s edge
(6.00 s, $0.195); at 24 it misses (6.43 s).

### (c) Chat with think time (mean >= 30 s) and ~12 GB of host RAM per GPU: device-only, ~88 sessions per GPU

```bash
SGLANG_OPT_GEMMA4_FP8_VOCAB_TABLE=1 \
SGLANG_OPT_TRITON_EXTEND_SM120_FP8_KV_TILES=1 \
SGLANG_OPT_GEMMA4_FUSED_GLUE=2 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
SGLANG_LOGPROB_CHUNK_SIZE=128 \
systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0 \
python -m sglang.launch_server --model-path <fp8-dir>/text \
  --kv-cache-dtype fp8_e4m3 --context-length 16384 \
  --json-model-override-args '{"max_position_embeddings": 16384}' \
  --disable-prefill-cuda-graph --max-running-requests 64 \
  --mem-fraction-static 0.955 --swa-full-tokens-ratio 0.268 --cuda-graph-max-bs-decode 48
```

- **Sizing:** ~88 live sessions per GPU at 30 s of mean think time and a 10 s p90 SLO, so 2,200 concurrent
  sessions need ~25 GPUs. Use sticky routing per session.
- **Why no HiCache:** at these think times a 12 GB HiCache host pool adds nothing. Under write-through it is an
  inclusive mirror of the device pool, and idle histories outlive it.
- **Why default chunking:** chunk 2048 + lpm were 14-16% worse at the chat SLO edge.
- **Longer think time:** round 1 measured >= 118 sessions per GPU at 60 s of think time. That was before the
  decode-glue fusion and the dense FP8 tiles; round 2 did not re-measure 60 s.

### What each server flag is for

| flag | why | study step |
|---|---|---|
| `--kv-cache-dtype fp8_e4m3` | FP8 KV cache: half the bytes per token | FP8 baseline |
| `--context-length 16384` | the workload's longest prompt + reply fits; bounds the request table | FP8 baseline |
| `--json-model-override-args '{"max_position_embeddings": 16384}'` | sizes the RoPE tables to the served context instead of the model's maximum: weights 25.12 -> 24.43 GiB | L9 |
| `--max-running-requests 64` | sizes the request-to-token table to 64 rows, freeing activation memory for the KV pool | L10 |
| `--mem-fraction-static 0.955` | gives the freed memory to the KV pool. GPU peak stays below CUDA capacity - 512 MiB (31,642 MiB on the RTX 5090) only with expandable segments | L7 |
| `--swa-full-tokens-ratio 0.268` | re-balances the sliding-window and full-attention pools for ~5K-token sessions (Gemma-4: 25 sliding layers with window 1,024, 5 full layers) | L4 |
| `--disable-prefill-cuda-graph` | the prefill graphs' memory goes to the KV pool | FP8 baseline |
| `--cuda-graph-max-bs-decode 48` | decode CUDA graphs up to batch 48; above the largest captured size decode runs eagerly | memory step |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | keeps the allocator's peak flat (31,556-31,570 MiB vs 31,590-31,768 without), inside the 512 MiB rule | memory step |
| `--enable-hierarchical-cache --hicache-size 12 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first` | (a) only. A 12 GB host-RAM copy of evicted prefixes keeps the prefix hit rate at ~0.77 past the device pool's cliff. `kernel` io + `page_first` because the `direct` path copies token by token at page size 1; `write_through` because `write_back` never saves internal sliding-window nodes | HiCache step |
| `--chunked-prefill-size 2048 --schedule-policy lpm` | (a) only: smaller prefill chunks raise the device hit rate and smooth decode; lpm admits queued turns with cached prefixes first | C3 |

## 4. The levers

Commits are listed in order on this branch. "Both" means configs (a) and (c).

| # | lever | commit | switch, default | in config | measured effect (one RTX 5090) |
|---|---|---|---|---|---|
| 1 | **K1** Gemma-4 decode-glue fusion: q/k/v RMSNorm + RoPE + FP8 KV store in one kernel, norm pairs fused | `perf(gemma4): fuse the per-layer decode glue` | `SGLANG_OPT_GEMMA4_FUSED_GLUE=2`, default 0 (off) | both | 1,061 -> 791 kernels per decode step; 28 in flight p90 -4.4%, tok/s +3.9%; 12 in flight -2.2% / +2.5%; chat at 30 s think p90 -8.1% |
| 2 | **L8** one FP8 table for the tied embedding and LM head, with its Triton FP8 vocab-head kernel | `perf(gemma4): optional FP8 vocab table` | `SGLANG_OPT_GEMMA4_FP8_VOCAB_TABLE=1`, default off | both | weights 24.43 -> 23.74 GiB, freed memory goes to the KV pool: +4 concurrent 5K-token sessions in the burst probe (29 -> 33, on a memory stack that also had L5) |
| 3 | **C2-A** SM120 FP8-KV tiles for Triton extend (prefill) attention | `perf(attention): sm120 FP8-KV Triton extend tiles` | `SGLANG_OPT_TRITON_EXTEND_SM120_FP8_KV_TILES=1`, default off | both | 12 in flight p90 -10.3%, tok/s +10.6%; prefill attention kernels 1.7-3.8x faster |
| 4 | **C1** tuned Triton fused_moe config, RTX 5090, E=128, N=704, FP8 W8A8 per-channel | `perf(moe): tuned fused_moe config` | config file, **on** for an RTX 5090 (kill: `SGLANG_MOE_CONFIG_DIR=<dir with an empty configs/>`) | both | 12 in flight p90 -1.2%, tok/s +1.1% |
| 5 | **HiCache fix 1**: pin the load-back window during SWA prefill admission | `fix(hicache): pin the load-back window` | `SGLANG_OPT_HICACHE_PIN_LOAD_BACK_WINDOW=1`, default off | (a) | without it the scheduler crashes at 28 in flight with HiCache (SWA out-of-memory on a load-back); with it, 0 failures from 20 to 32 |
| 6 | **HiCache fix 2**: fence write-through copies behind the forward stream | `fix(hicache): fence write-through copies` | `SGLANG_OPT_HICACHE_FENCE_WRITE_THROUGH=1`, default off | (a) | multi-turn load-back exactness 5/12 -> 12/12 (stock HiCache restores a stale token per turn); cost -1.0..-1.8% tok/s |
| 7 | **c1** tuned `triton_scaled_mm` tiles for the six dense FP8 shapes at decode M 1-48, RTX 5090 | `perf(fp8): tuned Triton tiles for Gemma-4-26B-A4B dense shapes` | config files on upstream's tuned-tile route, **on** for an RTX 5090 (kill: `SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE=0`) | both | dense GEMMs 4.86 -> 1.40-1.46 ms per decode step; 28 in flight p90 -11.2%, tok/s +14.1%; 12 in flight -17.8% / +23%; chat ~73 -> ~88 sessions per GPU |
| 8 | **c2** (optional) SWA window-id translate without a host sync in the Triton decode replay | `perf(attention): opt-in SWA window translate without a host sync` | `SGLANG_OPT_SWA_DECODE_NO_HOST_SYNC=1`, default off | neither | exact; 28 in flight p90 -1.2%, tok/s +0.4% (under the study's 1% bar) |
| 9 | **SWA margin** (optional) keeps extra SWA tokens behind a cached prompt, so a chat's next turn, which matches 4 tokens short under Gemma-4's template, reuses it | `fix(swa): opt-in SWA margin` | `SGLANG_OPT_SWA_PREFILL_WINDOW_MARGIN=128`, default 0 | neither | HiCache config with default chunking: +7.1% tok/s, p90 -7.3%, hit 0.70 -> 0.80; on config (a): +1.3% tok/s but retractions 0.69% -> 2.67% in a 30-min soak, so it is not in (a) |

Notes on the levers:

- **K1 is not in the base.** The study's trees carried it from earlier work (gemma4nv T4); here it is commit 1.
- **Both config files apply by default, but only on an RTX 5090.** c1's dense tiles and C1's MoE config are keyed by
  device name; every other GPU keeps upstream's kernels. The tile route itself is upstream (sgl-project #34331).
- **The margin's known cost.** Its retain floor holds window + margin sliding-window KV for every decoding request,
  not only for prompts whose insert stopped at a branch point; that is the likely source of the retraction rise.
  The upstream version of the fix (sgl-project/sglang#42703) narrows it.
- **Server flags count too.** These steps are flags, not code: the memory flags (L4, L7, L9, L10, decode graphs to
  48, expandable segments), HiCache with a 12 GB host pool and chunk 2048 + lpm, all in section 3.
  - Burst capacity: the memory flags with L8 take it from 17 to 29 sessions.
  - HiCache at 28 in flight: 610 -> 927 output tok/s.
  - Chunk 2048 + lpm at 32 in flight: p90 -9%, tok/s +3%.
- **Upstream PRs for the fixes:** fix 1 is sgl-project #42700, fix 2 #42701, the margin #42703.

### Not shipped

- **The HiCache retraction-backup fence** (`jumanzii/hicache-retraction-fence`, 7d0ba713da). It fenced
  `retraction_backup` in PD-disaggregated decode. Later analysis found that path is not racy: the backup copies
  `seqlen - 1` tokens, all written by forwards whose results the scheduler has already processed. The race is on the
  restore side, fixed differently in sgl-project #42702. It also rewired the scheduler outside its switch, and no
  shipped config runs PD disaggregation.
- **Rejected or debug code from the study's trees:**
  - L5, the sliding-window lock release (`SGLANG_OPT_SWA_RELEASE_SLID_WINDOW`), with its NaN-poison debug switch;
  - the gemma4nv NVFP4 and speculative-decoding levers (MTP FP8 head, split-KV verify on CUDA, spec replay, small-M
    BF16 / FP8 weight-only GEMM routes, the FP8 LM-head copy);
  - every `experiments/` directory (harness, runs, reports; they stay on `jumanzii/g4poc`).

## 5. Results

### Cost and capacity, round 1 -> round 2 (this branch)

| traffic | FP8 base | round 1 | round 2 (this branch) |
|---|---|---|---|
| in flight, 30-min soak at 28 (bs3) | (10 s SLO: 12 in flight, $0.343) | p90 8.59 s, p99 11.13 s, 952 tok/s, **$0.204** | p90 7.33 s, p99 8.95 s, 1,127 tok/s, **$0.173** |
| 6 s p90 SLO | 8 in flight, $0.415 | 12 in flight, **$0.300** | 16 in flight, **$0.207** |
| chat, 30 s mean think, 10 s p90 SLO | | **~70 sessions**, $0.45-0.47 | **~88 sessions**, ~$0.36 |
| FP8 base, 10 s SLO | **$0.343 at 12 in flight** | | |

- **Hosts:** the in-flight soaks and the 6 s points ran on build-server-3. Round 1's in-flight soak on build-server-2
  agreed: p90 8.52 s, $0.203.
- **Retractions:** 0.69% in round 1's soak, 0.87% in round 2's. There were 0 failed requests.
- **Chat cost per token is higher** than in flight because idle sessions' histories do not fit any cache at 30 s of
  think time, so most turns re-prefill their history. Host RAM per GPU is the lever. A model (not measured) puts a
  48 GB host pool at ~107 sessions per GPU with round 1's levers.

### Quality

Every arm is greedy and paired per item against the FP8 base anchor run on the same host.

| check | round 1 final | + K1 (decode glue) | + c1 (dense tiles) | budget |
|---|---|---|---|---|
| GSM8K, all 1,319 | 96.36% vs 96.13% (McNemar p 0.58) | 96.44% vs 96.13% (p 0.45) | 96.29% vs 96.21% (p 1.0) | no significant drop |
| tool-call JSON | 40/40 | 40/40 | 40/40 | 40/40 |
| role-play reference NLL rise | +0.0003 nats/token | +0.0030 | +0.0016 | <= 0.02 |
| role-play language adherence (80 items) | net 0 flips | 68 -> 66 (2 flips, both mixed-language detector items) | 67 vs 68 (1 flip, the item every neutral change flips) | inside the paired re-roll band |
| long role-play KL vs A/A (8 prompts, teacher-forced) | 0.029 vs 0.022 | 0.0475 vs 0.0328 (limit 0.066) | 0.034 vs 0.033 (limit 0.066) | under the limit |
| multi-turn exactness, concurrency 1 (12 later turns, each a HiCache load-back) | 12/12 | 12/12 | 12/12 alone and stacked with K1 | 12/12 |

- **Which configs:** K1 and c1 were judged one at a time on the in-flight final. The full stack (final + K1 + c1)
  passed multi-turn exactness 12/12. This branch reproduces that stack token for token (section 9).
- **KL per earlier lever:** C2-A 0.032 against a 0.041 limit; L8 within budget on GSM8K, tool-JSON and role-play.

## 6. Deployment caveats

- **HiCache's host-memory start check (config (a)).** SGLang wants 10 GiB of headroom in the server's memory cgroup
  beyond the 12 GB host pool it pins. It counts the cgroup's whole usage at that moment, including page cache charged
  to it. In a 28 GB cgroup 2-3 of every 7-9 starts failed with "Not enough host memory available". Three steps make
  starts reliable:
  - give the cgroup >= 32 GB;
  - read the weight files outside the server's cgroup just before launch (page cache stays charged to whoever read it
    first);
  - retry the start on that error.
- **Client keep-alive below 5 s.** SGLang's HTTP server closes an idle keep-alive connection after 5 s
  (`SGLANG_TIMEOUT_KEEP_ALIVE`). A chat client idles between turns, so a pooled connection can be reused just as the
  server closes it, and the request fails at once with a disconnect before any byte is processed.
  - Set the client's keep-alive below 5 s (the study used 2 s).
  - Retry once on a disconnect that arrives before any response bytes; that request never ran.
  - Raising the server's keep-alive also works, at the cost of more idle sockets on the server.
- **On a shared GPU, use `--mem-fraction-static 0.94`.** At 0.955 the server's peak is ~31.35-31.5 GB, 85-290 MiB
  under CUDA capacity - 512 MiB. Another process's CUDA context (even a ~500 MiB one) can then run either one out of
  memory. 0.94 gives back ~470 MiB of KV pool; it was not measured for throughput.
- **With HiCache, batched outputs are not reproducible run to run.** Which path serves a shared prefix (device hit,
  host load-back or recompute) depends on the timing of asynchronous host copies, and the paths produce different FP8
  bytes for the same prefix.
  - A/A: 66/80 role-play replies are token-identical across two runs, against 80/80 without HiCache.
  - Load-back itself is exact: at concurrency 1, 12/12 multi-turn turns are token-identical to device hits.
  - Config (c) has no HiCache and reproduces token for token at a fixed config.
- **Retraction tail above 28 in flight.** Past 28 the device pool overcommits and SGLang retracts requests.
  - At 32 in flight (round 1, 30 min) 1.2% were retracted and p99 reached 37 s.
  - At 28: 0.69% (round 1) and 0.87% (this branch), with p99 under 9 s.
  - Keep admission at 28 per GPU. The 4-minute sweeps do not show this tail; 30-minute soaks do.
- **Chunk 2048 + lpm is for in-flight traffic only.** Under think time they lowered the prefix hit rate and raised
  p90 at the SLO edge by 14-16%, so config (c) keeps default chunking.
- **Sliding-window caching and chat templates.** Gemma-4's template re-renders a past assistant turn 4 tokens shorter
  than the generation prompt it was decoded under. So without the margin (lever 9) every chat's second turn
  re-prefills its history. The measured numbers include this.

## 7. Hardware dependence

Tuned for the RTX 5090 (SM120, 32 GB, ~99 KB shared memory per SM):

- **C2-A extend tiles.** Selected on any SM120 GPU (CUDA capability 12.x) with an FP8 KV cache, and swept on an RTX
  5090. On another SM120 part, re-sweep with `experiments/g4poc/compute/extend_attn_bench.py`.
- **c1 dense FP8 tiles.** Keyed by device name `NVIDIA GeForce RTX 5090` and the six (N, K) shapes. Elsewhere the
  route finds no file and keeps CUTLASS. Re-tune with `compute/k2_dense_gemm_bench.py` and
  `compute/k2_channelwise_configs.py`.
- **C1 fused_moe config.** Keyed by device name and found under `triton_3_7_1`. Under another Triton version SGLang
  falls back to this file with a warning (possibly slower). Re-tune with SGLang's
  `benchmark/kernels/fused_moe_triton` tuner or the study's `compute/moe_tune.py`.
- **L8 vocab-head tile.** Keyed by shape only (262144 x 2816), so it runs on any CUDA GPU; its speed was measured
  only on the 5090.
- **K1.** No hardware tuning. Its Triton kernels need an FP8 E4M3 KV cache (SM 8.9+), the Triton backend and the
  static NHD pool; anything else keeps the unfused path.
- **Memory settings.** `--mem-fraction-static 0.955`, `--swa-full-tokens-ratio 0.268`, decode graphs to 48 and the
  12 GB host pool fit a 32 GB card with 31,642 MiB usable, and the 30-minute soak plateau was 31,354 MiB. Any other
  card needs these re-derived: the study's `memory/capacity.py` probe finds the largest clean session count per
  setting.

On other hardware:

- **RTX 4090 (24 GB, SM89).** The FP8 weights alone are 23.7 GiB, so the model does not fit one 4090 with any KV
  pool. It needs two GPUs (TP2) or smaller weights.
  - Off by device: C2-A (capability check), c1 and C1 (device-name keys).
  - K1 still applies.
  - Re-tune: the dense tiles, the MoE config and every memory setting.
- **TP2 (any GPU).** Per-rank shapes change: qkv and gate_up N halves, o and down K halves, MoE N = 352 per rank. So
  the c1 and C1 files no longer match, and both need re-tuning.
  - L8 refuses to load under TP2 (ValueError): the vocab-parallel embedding is 131072 x 2816 per rank, which has no
    tuned tile. Leave L8 off or add a tile for that shape.
  - K1 and the HiCache fixes are shape-generic, but they have not been run under TP.

## 8. Reproduce

- **Harness:** `jumanzii/g4poc` on fractalyze/sglang, `experiments/g4poc/`.
  - `gate/` holds the load generator, sweeps, soaks, A-B-B-A, quality, KL and the host-safety protocol.
  - `*/refs.json` name every measured config.
  - `WORKLOAD.md` describes the session file and the loads.
- **Ship refs:** this branch's two configs, written out flag by flag, are `ship-inflight` and `ship-chat` in
  `experiments/g4poc/ship/refs.json` on `jumanzii/g4poc-ship-verify`. That file also has the exactness and
  upstream-equivalence pairs; `ship/ship_unit.sh` runs every check in section 9. For example, from
  `experiments/g4poc` on a host with the harness's `gate/env.sh` set up:

  ```bash
  source gate/env.sh
  G4POC_SERVER_MEMORY_MAX=28G python -m gate sweep --ref ship-inflight --load inflight --concurrency 16,28
  G4POC_SERVER_MEMORY_MAX=28G python -m gate sweep --ref ship-inflight --load soak --concurrency 28   # 30 min
  G4POC_SERVER_MEMORY_MAX=24G python -m gate sweep --ref ship-chat --load pthink30 --concurrency 92
  ```

- **Reports on `jumanzii/g4poc`:**
  - `COMPUTE.md`:
    - section 2: C1, C2-A, chunk 2048, lpm;
    - section 3: the round-1 final, its soaks and the margin evaluation;
    - section 5: chat with think time;
    - section 7: K1;
    - section 8: c1 and c2;
    - sections 9-10: the MoE and decode-attention measurements that closed round 2.
  - `memory/REPORT.md`: the memory flags, L8 and the HiCache deployment notes.
  - `hicache/REPORT.md` and `hicache/UPSTREAM.md`: the two HiCache bugs.
  - `BASELINE-FP8.md`: the checkpoint and the FP8 baseline.
- **Round-2 doc branches:**
  - `jumanzii/g4poc-r2-k1-doc` (K1, COMPUTE.md section 7);
  - `jumanzii/g4poc-k2` (c1, c2 and the closing measurements, sections 8-10).
  Both are merged into `jumanzii/g4poc`.
- **The study's code trees:**
  - `jumanzii/g4poc-r2-k2` (cfc12c0bac) is the tree the round-2 numbers ran on;
  - `jumanzii/g4poc-final-hicache` (a0491db764) is round 1's;
  - `jumanzii/g4poc-swa-margin` is the margin's.

## 9. How this branch was verified

This branch's code, `python/`, against the measured tree cfc12c0bac:

- **Identical files:** `gemma4_causal.py` (apart from one import reordered by isort), `gemma4_fused_ops.py`,
  `schedule_policy.py`, `cache_controller.py`, `triton_backend.py`'s decode path and the tile configs.
- **Removed:** only code no shipped config runs, every piece behind a switch that was off:
  - L5 and its debug switch;
  - the gemma4nv MTP and speculative code;
  - split-KV verify on CUDA;
  - the small-M BF16 / FP8 weight-only routes and their kernels' unused shapes;
  - the FP8 LM-head copy of the multimodal class.
- **Added or moved:**
  - the margin (default 0);
  - the C1 MoE config, moved from `SGLANG_MOE_CONFIG_DIR` into the default config directory;
  - two comment edits.

Checks on build-server-3's RTX 5090, 2026-10-06. They ran on the ship head `e652f30b1d` (`python/` tree
`bd956697e2`); this guide's commit changes no other file. Refs, unit script and records are in
`experiments/g4poc/ship/` on `jumanzii/g4poc-ship-verify`.

| check | result |
|---|---|
| unit tests | every commit's own tests at that commit, GPU tests included: pass. All ship test files at the head: 1,375 passed, 0 failed |
| identity, in-flight config (a) vs cfc12c0bac: greedy, concurrency 1, 4 role-play sessions x 3 turns, 128 tokens each | **12/12 token-identical**, identical `cached_tokens` |
| identity, chat config (c) vs cfc12c0bac | **12/12** |
| HiCache multi-turn exactness: (a) on a 16K-token device pool, so every later turn loads back from host, vs the same flags without HiCache | **12/12**. Both arms are also 12/12 identical to round 2's runs of the same pair on cfc12c0bac |
| upstream equivalence: this branch with every switch unset, the dense tile route killed and the MoE config dir emptied, vs upstream `a9871012ac`, under (a)'s server flags | **12/12** |
| 28 in flight, A-B-B-A vs cfc12c0bac (240 s points) | this branch p90 7.43 / 7.36 s vs 7.34 / 7.33 s (gain 0.992); 1,125 / 1,120 vs 1,134 / 1,128 output tok/s (0.992); p99 gain 1.023; retracted 5 + 8 vs 11 + 12; 0 failed; control drift <= 0.5%. Round 2's 30-min soak: p90 7.33 s, 1,127 tok/s |
| GPU memory during that A-B-B-A (1 s samples, whole GPU) | plateau 31.3K MiB, under the 31,642 MiB rule. 4 of 2,138 samples above it: 2 are the host's periodic ~500 MiB canary (hh:m3:10); 2 consecutive ones at 19:04:18-19 read 32,097 MiB (+743 MiB for 2 s) during one of this branch's sweeps, process not recorded. Round 2's 30-min soak of the same code showed no such sample |
| chat (c), pthink30 at 72 | 63.3 live sessions, 2.46 turns/s, p50 / p90 / p99 3.06 / 5.97 / 9.07 s, 432 tok/s, 0 failed, 0 retracted. Round 2 on the same seeded plan: 63.3, 2.47, 3.16 / 6.06 / 9.14 s, 432 |
| server starts | both configs start. In the 28 GB scope, SGLang's HiCache host-memory check failed 4 of 14 in-flight starts (1 on this branch, 3 on cfc12c0bac); every one passed on retry or on the rerun pair (section 6) |
