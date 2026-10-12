# Qwen3.8-27B on one RTX 5090: the decode-mk switches against stock SGLang

Every arm was served in turn in one session on one idle RTX 5090 by
[`ab.py`](ab.py), with the batch-1 protocol of fractalyze/decode-mk's
`control/qwen38` baseline:

- **Stock:** SGLang's cookbook RTX 5090 recipe at fractalyze/sglang
  `1d59ce7`, on the int4 (`cyankiwi/Qwen3.8-27B-AWQ-INT4` at `6e134ba`) and
  NVFP4 (`nvidia/Qwen3.8-27B-NVFP4` at `482ca0f`) checkpoints, with and
  without its EAGLE MTP at three drafts a cycle.
- **Ours:** this branch's `SGLANG_DECODE_MK_DECODE` and `_PREFILL` switches
  on the int4 checkpoint (decode-mk `bc571b0` vendored), without MTP and with
  `SGLANG_DECODE_MK_MTP` at one to three drafts a cycle, the most the switch
  takes.

## Speed, batch 1

Prompts of 30, 1,053 and 8,509 tokens; each cell is the median of five
replies, with their range.

| Arm | TTFT, 30 tok (ms) | TTFT, 1,053 tok (ms) | TTFT, 8,509 tok (ms) |
|---|---:|---:|---:|
| int4 | 21 (21–21) | 304 (304–305) | 2,124 (2,121–2,136) |
| int4 + MTP | 30 (29–30) | 318 (317–319) | 2,176 (2,174–2,178) |
| NVFP4 | 23 (23–30) | **95** (94–115) | **756** (755–771) |
| NVFP4 + MTP | 28 (28–30) | 104 (104–111) | 800 (799–804) |
| ours | 28 (28–29) | 325 (324–328) | 2,649 (2,643–2,651) |
| ours + MTP, 1 draft | 28 (28–28) | 327 (327–337) | 2,672 (2,665–2,674) |
| ours + MTP, 2 drafts | 28 (28–28) | 327 (327–328) | 2,668 (2,659–2,673) |
| ours + MTP, 3 drafts | 28 (28–28) | 327 (327–329) | 2,669 (2,659–2,675) |

| Arm | Decode tok/s, 30 tok | Decode tok/s, 1,053 tok | Decode tok/s, 8,509 tok | Accepted tokens per step |
|---|---:|---:|---:|---:|
| int4 | 82.3 (82.3–82.3) | 82.2 (82.1–82.2) | 80.6 (80.6–80.6) | — |
| int4 + MTP | 130.6 (130.3–130.6) | 136.9 (136.7–137.0) | 137.4 (137.3–137.4) | 2.93 |
| NVFP4 | 77.5 (77.5–77.6) | 77.4 (77.4–77.4) | 76.1 (76.0–76.2) | — |
| **NVFP4 + MTP** | **149.2** (149.2–149.2) | **175.6** (175.5–175.6) | **155.5** (155.3–155.5) | 2.81 |
| ours | 77.0 (76.9–77.0) | 76.1 (76.1–76.1) | 70.4 (70.4–70.4) | — |
| ours + MTP, 1 draft | 115.3 (115.3–115.4) | 117.9 (117.9–118.1) | 107.9 (107.8–108.0) | 1.84 |
| ours + MTP, 2 drafts | 122.5 (122.2–122.7) | 136.7 (136.6–136.7) | 124.3 (124.1–124.3) | 2.50 |
| ours + MTP, 3 drafts | 121.5 (121.3–121.6) | 138.7 (138.3–138.8) | 123.2 (123.2–123.3) | 2.91 |

- Measured on 2026-10-12: RTX 5090, driver 595.84, CUDA 13.0.
  `results/<arm>.json` holds every reply and the server's launch command.
- Every reply is greedy, has thinking off, and runs to the 512-token cap.
  The prefix cache is flushed before each, so every prefill runs in full.
- TTFT is the client's time to the first streamed token. Decode tok/s is the
  tokens after the first over the time from the first to the last.
- Accepted tokens per step is the server's mean over the whole run, warm-up
  included.

**We do not beat the fastest stock arm.** NVFP4 + MTP decodes faster than
every one of our arms at every prompt length, 175.6 tok/s against our best
138.7 at 1,053 tokens, and prefills 3.3 times faster at 8,509 tokens. Against
stock int4 + MTP, the same checkpoint, ours decodes 1% faster at 1,053 tokens
and 7–10% slower at 30 and 8,509, and prefills 23% slower at 8,509.

- **The served MTP runs the checkpoint's bf16 LM head, at most three
  drafts.** decode-mk's own MTP benchmark
  (`harness/results/qwen38-mtp.md`) passes NVFP4 + MTP only with its int4 LM
  head (`Qwen38Weights.with_int4_lm_head`), at three drafts or four; with the
  bf16 head its best is about ours here. This branch has no switch for that
  head, and `DECODE_MK_MTP` refuses a fourth draft though the verify step
  checks five tokens. Both are
  [fractalyze/decode-mk#105](https://github.com/fractalyze/decode-mk/issues/105),
  along with the hub-id launch below.
- **Our prefill is slower than stock int4's** at 1,053 and 8,509 tokens, as
  decode-mk's own prefill benchmark also measures it.

## Quality

Ours (MTP, three drafts) against stock int4, on the same int4 checkpoint and
the same questions: all 1,319 of GSM8K, and the 1,000 MMLU questions sgl-eval
samples at its seed 0. Every reply is greedy with thinking off, capped at 2,048
tokens, one request at a time, under `ab.py eval` with sgl-eval 0.1.2.

| Benchmark | ours + MTP, 3 drafts | int4 | Ours − int4, 95% CI | Right in ours only / int4 only | Truncated, ours / int4 |
|---|---:|---:|---:|---:|---:|
| GSM8K (1,319) | 96.44% | 96.13% | +0.30 pts (−0.17, +0.77) | 7 / 3 | 0.99% / 1.29% |
| MMLU (1,000) | 87.50% | 88.20% | −0.70 pts (−1.76, +0.36) | 11 / 18 | 3.5% / 3.6% |

**Neither difference is significant.** The interval is the normal
approximation on the per-question paired difference, and both intervals hold
zero. One arm is right and the other wrong on 0.8% of GSM8K and 2.9% of MMLU.
No request errored.
`results/eval/<arm>-<benchmark>/` holds sgl-eval's `metrics.json` and the
server's launch command; the per-question predictions are not committed.

There is no bf16 row: the bf16 checkpoint does not fit the card, and serving
it with `--cpu-offload-gb` on this host ran out of host memory while loading
its weights.

## Running

`ab.py` serves every arm from the checkpoint's directory in the Hugging Face
cache, which must already hold it: this branch's switches read
`--model-path` as a directory, so a hub id does not serve them
([fractalyze/decode-mk#105](https://github.com/fractalyze/decode-mk/issues/105)).

```bash
# A venv with stock SGLang 1d59ce7, from decode-mk's control/qwen38/setup.sh,
# and sgl-eval 0.1.2 in any other.
PYTHON=<venv>/bin/python
$PYTHON ab.py time int4 int4-mtp nvfp4 nvfp4-mtp ours ours-mtp1 ours-mtp2 ours-mtp3 \
  --out-dir results
SGL_EVAL=<sgl-eval venv>/bin/sgl-eval $PYTHON ab.py eval ours-mtp3 gsm8k --out-dir results/eval
SGL_EVAL=<sgl-eval venv>/bin/sgl-eval $PYTHON ab.py eval ours-mtp3 mmlu --num-examples 1000 \
  --out-dir results/eval
```

Each arm needs the GPU to itself; `ab.py` waits for it to be idle before
serving one.
