# DPCache on Qwen-Image 2.1: measured study

Preliminary, single-GPU, single-checkpoint evidence for the DPCache adapter that
this branch stacks on top of. It is not a general ranking of caching methods and
not a production claim.

## What this branch is

`experiment/qwen-image21-dpcache` stacks on `feat/qwen-image21-dpcache-core`,
which is the upstream-facing change: the runtime adapter, the planner, the
schedule artifact and its tests. Nothing here is required to serve DPCache.

This branch adds only research material:

- `python/sglang/multimodal_gen/benchmarks/bench_qwen_image21_dpcache.py` --
  corpus freezing, feature capture, PACT scoring, planning, sweeps, scoring,
  selection freezing, contact sheets.
- `python/sglang/multimodal_gen/benchmarks/bench_qwen_image21_cache_comparison.py`
  -- the comparison against the caching strategies SGLang already ships, and the
  endpoint-matched uniform placement control.
- The tests for both, plus the uniform control's own tests.
- This directory: the frozen artifacts and the report.

The uniform control's *generator* (`uniform_full_steps`, `UNIFORM_GENERATOR`)
lives in the comparator, not in the runtime cache module: SGLang ships no way to
produce a uniform schedule, and the generator exists only to separate what the
calibrated placement buys from what the budget buys. What it writes is an
ordinary DPCache schedule artifact, and the comparator serves it through the
same generic runtime as a DP schedule -- same `dpcache_schedule` request field,
same validation, same predicted-step behaviour. That shared path is what makes
the control endpoint-matched rather than a second implementation.

Regenerating `uniform-K12/K20/K40` from the comparator reproduces the frozen
artifacts' full step sequences and their endpoints -- mandatory prefix, budget
and last key. That is the claim; a byte-for-byte claim is not established
against the copies published here, because every schedule in this directory is
marked `rewritten` in `MANIFEST.json` (producing-machine calibration paths were
relativised) and so has a different digest from its raw original by design.

## Where this lives

- **Upstream draft PR** for the runtime change this stacks on:
  [sgl-project/sglang#40848](https://github.com/sgl-project/sglang/pull/40848).
- **Original prototype archive**:
  [`archive/qwen-image21-dpcache-study`](https://github.com/fractalyze/sglang/tree/archive/qwen-image21-dpcache-study),
  tip `38877d1b2` -- a snapshot of the completed prototype as it stood before
  the upstream split, kept for provenance rather than for use. The measurement
  provenance is finer-grained than that snapshot: each run artifact records the
  working-tree hash it was produced from, and those are the hashes a number
  should be traced to.

## Method and attribution

The method adapts **DPCache** (https://arxiv.org/abs/2602.22654). The adaptation
is ours and differs from the paper and its reference implementation on purpose:

- The predictor is **first order** -- `h_j + (h_j - h_i) / (j - i) * (t - j)`
  from the two latest full steps -- in the feature dtype, with the same
  operation order in calibration and inference. Predictions never become
  anchors.
- Because a prediction depends on exactly the two retained anchors, the cost of
  a key triple is well defined and the planner is an **exact dynamic program
  over (previous, current) key pairs**. The reference planner keeps one
  predecessor per (budget, current key), which is not optimal for
  triple-dependent costs.
- **Only steps the runtime actually predicts are scored.** The terminal sentinel
  is an endpoint; no synthetic feature is invented or scored there. The paper's
  proxy includes the endpoint.

No training, distillation, quantization or weight change is involved. This is
not a reproduction of the paper's order-2 headline results.

## Provenance of the measurements

Every number in this report was measured on the **original tested source**,
commit `a9871012acb768dc94a43a6542cc32626c7b7b0b` plus the working tree that
became this change. It was not re-measured after the upstream rebase or the
cleanup, and nothing here is relabelled as a measurement of the current tree.

What was re-checked after the rebase onto upstream `1d59ce7c9` is narrow, and
should be read narrowly. On **8 validation pairs** -- the validation split of
`corpus/corpus-v1.json`, arms `native` and `K=20` only -- the rebased tree
reproduces the original tested source bit-for-bit (`torch.equal` on decoded
samples and final latents, plus identical PNG bytes). Separately, the rebased
tree reproduces pristine upstream bit-for-bit with DPCache disabled and with an
all-full schedule.

That is the whole of the re-validation, and **the held-out table below was not
rerun on this tree**. Its 20 pairs are the fresh comparator held-out corpus,
which shares no prompt with the validation split; it also covers `K=12`, both
uniform controls and both Cache-DiT presets, none of which was re-measured on
any split. So **every LPIPS figure and every timing figure in that table belongs
to the original study on `a9871012`**, and none of them is restated here as a
measurement of the current tree. The parity evidence supports exactly one
thing: on those 8 pairs, for those 2 arms, the rebase changed no output bit.

| | |
| --- | --- |
| Checkpoint | `Qwen/Qwen-Image-2.1`, revision `790c92633540aa0cb11d9abf19eb46d861714758` |
| GPU | one NVIDIA GeForce RTX 5090 (32 GiB), driver 610.43.02 |
| Software | torch 2.13.0, diffusers 0.37.0, transformers 5.12.1, python 3.12.3 |
| Recipe | 1024x1024, 40 scheduler updates, guidance 1.0, no CFG, BF16, `torch_sdpa`, DiT layerwise offload, eager (no compile, no CUDA graphs), CPU generator, 1 image per request |

The recipe is fixed across every arm. **BF16 weights and DiT layerwise offload
are part of the measured scope**, not incidental: a predicted step runs no
transformer block, so under layerwise offload it also streams no block weights,
which is a large part of why wall time tracks `K` so closely. Results are not
expected to transfer to a resident-weight configuration unchanged.

## Corpus and splits

There are **two corpora and two separate held-out studies**. They are never
pooled, and no number here averages across them.

`corpus/corpus-v1.json` -- the original DPCache study. 10 calibration prompts,
8 validation pairs, and a held-out set of **20 pairs from 10 prompts x 2
seeds**; all three splits are prompt-disjoint. Calibration produced the DP
schedules, the validation split alone chose the DP budgets
(`results/selection-dpcache-v1.json`, rule: "validation split only"), and its
held-out 20 were then evaluated once.

`corpus/corpus-comparator-v1.json` -- the comparison study, and the source of
the table below. 3 control prompts (reused calibration prompts, exactness only)
plus a **fresh** held-out set of **20 pairs from 10 new prompts x seeds
7001/7002**, prompt-disjoint from every earlier split. It reuses the earlier
work rather than redoing it: the DP and uniform arms are frozen exactly as the
calibration and validation splits of `corpus-v1` left them, and those same 8
validation pairs -- already seen -- were used only to pick the two Cache-DiT
presets. `results/selection-comparator-v1.json` records both sides of that:
`validation_corpus_sha256` is the original corpus, and
`expected_heldout_corpus_sha256` is the comparator corpus.

Each held-out set was evaluated once, after its own selection was frozen
(`results/selection-*.json`, written before the held-out run). The two 20-pair
held-out sets are disjoint and cover different arm lists. Prompt categories
across the two cover text rendering and typography, counting and composition,
people, faces and hands, complex scenes, illustration, transparency, natural
scenes and varied colour.

## Quality policy

LPIPS (AlexNet) against the native output of the **same prompt and seed**,
computed on **deterministically white-composited RGBA**. Alpha-dropped LPIPS and
maximum absolute alpha error are reported separately and are **not gated**; the
RGBA policy is provisional. Two readings are used, both this study's policy and
neither a threshold published by any paper:

- provisional: mean <= 0.18 and max <= 0.30
- strict: mean <= 0.05 and max <= 0.10

LPIPS is evaluation arithmetic in CPU FP32; it is not the model's inference
precision.

## Held-out results (comparator corpus, 20 fresh pairs)

These are the 20 fresh comparator held-out pairs, measured once on `a9871012`.
Neither the quality columns nor the timing columns were revalidated on the
rebased tree; see "Provenance of the measurements" above. The older study's
held-out 20 are a different corpus and are not shown here.

Time is allocated GPU-seconds per image: warm end-to-end client wall time on an
otherwise idle GPU, in grouped same-config repeats (3 repeats x 4 prompts), which
is what a steady single-config service sees. "Blocks" counts real transformer
block invocations per image; native runs 40 x 32 = 1280.

| Arm | Blocks | LPIPS mean | LPIPS max | alpha max | provisional / strict | GPU s/image | vs native |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native | 1280 | 0 | 0 | 0 | reference | 13.457 | 1.00x |
| dp-K12 | 384 | 0.0848 | 0.2200 | 0.071 | pass / fail | 4.197 | 3.21x |
| uniform-K12 (control) | 384 | 0.1548 | 0.2799 | 0.290 | pass / fail | 4.196 | 3.21x |
| cachedit-stock | 443 | 0.1023 | 0.1759 | 0.278 | pass / fail | 4.979 | 2.70x |
| dp-K20 | 640 | 0.0076 | 0.0350 | 0.043 | pass / **pass** | 6.845 | 1.97x |
| uniform-K20 (control) | 640 | 0.0701 | 0.2079 | 0.059 | pass / fail | 6.844 | 1.97x |
| cachedit-conservative | 710 | 0.0159 | 0.0425 | 0.094 | pass / **pass** | 7.678 | 1.75x |

Every arm measured 20/20 planned pairs and 12/12 grouped timings; grouped
standard deviation was 0.002-0.005 s.

### What the numbers say

- **At the strict gate**, `dp-K20` and `cachedit-conservative` both qualify.
  `dp-K20` used **10.8 % less GPU time** (6.845 vs 7.678 s) and was lower on both
  mean and max LPIPS. This is the stronger of the two comparisons.
- **At the provisional gate**, `dp-K12` used **15.7 % less GPU time** than
  `cachedit-stock` (4.197 vs 4.979 s) and had the lower mean LPIPS, but a
  **worse worst case** (0.220 vs 0.176). Which is preferable depends on whether
  mean or worst-case fidelity matters more for the workload.
- **Placement, not budget.** The uniform controls share their DP arm's budget,
  mandatory prefix, last key step and measured time to within a millisecond, and
  are clearly further from native. `uniform-K20` fails the strict gate that
  `dp-K20` passes. The offline PACT proxy predicted this ordering before any
  image was generated.
- **Equal scheduler updates are not equal compute.** The Cache-DiT arms ran
  15.4 % and 10.9 % more transformer blocks than the DP arms they are compared
  against, which exceeds the +/-5 % tolerance declared in advance. GPU-seconds is
  therefore the comparison, and no equal-compute claim is made.
- **TeaCache is unsupported here.** Qwen-Image 2.1 has no TeaCache adapter:
  `enable_teacache=True` is accepted, runs all 1280 blocks and reproduces native
  output bit-exactly. It appears as a no-op diagnostic and in no ranking. No
  coefficients from other models were reused.

### Exactness

| Control | Result |
| --- | --- |
| native repeated, native after each mode switch, native after an intentionally failed request | bit-exact |
| DPCache all-full (`K=40`) | bit-exact |
| uniform all-full (`K=40`) | bit-exact |
| Cache-DiT forced-full (warmup 40, threshold 0) | bit-exact, 1280 blocks |
| stock TeaCache flag | bit-exact, 1280 blocks (no-op) |
| **any accelerated arm** (`K<40`, or Cache-DiT caching) | **not** bit-exact |

Bit-exact means `torch.equal` on final latents and decoded tensors plus identical
PNG bytes, against a read-only export of the unmodified source commit --
28 of 28 rows on all three checks
(`results/crossrun-exactness-audit.json`). That is what makes the differences
above attributable to caching rather than to the integration.

**No accelerated arm is bit-identical to native.** Passing an LPIPS gate is not
equality, and this report does not claim it is.

### Cost

The DPCache schedules cost 155.7 GPU-seconds of offline calibration (feature
capture plus PACT scoring) and 0.04 CPU-seconds of planning; Cache-DiT presets
need none. Charged against the measured per-image savings, incremental
break-even is about **199 images** for `K=12` vs stock Cache-DiT and **187
images** for `K=20` vs conservative Cache-DiT. The comparator's own preset tuning
is a separate search cost and is not amortized into any per-image number.

### Figures

Small review aids, not data. No tensors and no full-resolution outputs are
published here.

- [`figures/heldout-quality-vs-gpu-seconds.png`](figures/heldout-quality-vs-gpu-seconds.png)
  -- mean and max white-composite LPIPS against allocated GPU-seconds per image,
  one point per arm, with both gates drawn.
- [`figures/heldout-worst-by-dpK12.jpg`](figures/heldout-worst-by-dpK12.jpg) --
  native / `dp-K12` / `uniform-K12` / `cachedit-stock`, rows ordered worst-first
  by `dp-K12` LPIPS.
- [`figures/heldout-worst-by-dpK20.jpg`](figures/heldout-worst-by-dpK20.jpg) --
  native / `dp-K20` / `uniform-K20` / `cachedit-conservative`, rows ordered
  worst-first by `dp-K20` LPIPS.

## Limits

- 20 held-out pairs from 10 prompts, 4 timing prompts, one GPU, one checkpoint,
  one resolution, one step count. Small samples; treat as preliminary.
- The two LPIPS readings are this study's policy, not published thresholds.
- The RGBA policy is provisional; alpha is reported, not gated. `uniform-K12`
  (0.290) and `cachedit-stock` (0.278) showed the largest alpha deviation.
- A passing mean does not guarantee every image. On this comparator corpus the
  worst `dp-K12` rows are the counting/spatial prompt (LPIPS 0.220 and 0.207,
  where the chairs around the table are arranged and counted differently) and
  the typography prompt (0.182 and 0.149, where the enamel sign's lettering and
  line breaks change). On those same rows `dp-K20` stays at or below 0.035 and
  is visually close, while `uniform-K20` reaches 0.208 on the chairs and 0.167
  on the sign. Inspect the two galleries linked under "Figures" -- each ordered
  worst-first by its own DP arm -- before choosing a budget.
- The **older** held-out study, on `corpus/corpus-v1.json`, recorded a different
  `K=12` worst case: a layout change on an aerial river scene, and one of four
  coloured cups becoming a vase-like object
  (`results/dpcache-results.json`, `heldout_20_pairs.K12.visual_note`). That is
  a separate corpus, a separate run and a separate arm list; its contact sheets
  are not published here, and it says nothing about the table above.
- Timing was measured on `a9871012` and is not restated for the rebased tree.
  Neither is held-out quality: see "Provenance of the measurements".

## Reproducing this

What is published here is enough to **re-run inference on the published
schedules** and to **redo a selection from scratch**. It is not enough to
re-derive the numbers above from artifacts alone: the raw feature captures, the
PACT error `.npz` tensors, per-image tensors, decoded outputs, run plans and
logs are not published, so calibration and scoring cannot be replayed -- they
have to be re-run.

In particular, `results/selection-*.json` cannot be handed to a harness as
`--selection`. Both records point at raw run directories that are not published
(`validation-02/scores.json`, `runs/validation-c01`), and both record the
**raw** schedule digests, which differ from the rewritten copies in this
directory by design. A held-out run needs a selection you froze yourself.

Both harnesses are run by path, and every subcommand has its own `--help`:

```bash
SRC=/path/to/sglang/python     # --source-dir, the tree under test
BENCH=$SRC/sglang/multimodal_gen/benchmarks
STUDY=benchmark/experiments/qwen_image21_dpcache
MODEL=/path/to/Qwen-Image-2.1/snapshots/790c92633540aa0cb11d9abf19eb46d861714758

python $BENCH/bench_qwen_image21_dpcache.py launch --help
python $BENCH/bench_qwen_image21_cache_comparison.py launch --help
```

**Repeat inference on a published schedule.** `--schedule-dir` resolves
`<arm>.json`, so `schedules/dp/` serves the eight published DP arms by name:
`K12`, `K16`, `K20`, `K24`, `K28`, `K32`, `K36`, `K39`.
`--timing-repeats 0` makes the run evidence-only: it generates images and makes
no timing claim.

```bash
python $BENCH/bench_qwen_image21_dpcache.py launch sweep \
  --corpus $STUDY/corpus/corpus-v1.json \
  --split validation --arms K20 \
  --schedule-dir $STUDY/schedules/dp \
  --model-path $MODEL --source-dir $SRC \
  --out runs/repro-k20 --timing-repeats 0 \
  --gpu-lock /tmp/qwen-image21.lock
```

That is the shape of the run behind the 8-pair parity check above. The
comparator wants one flat schedule directory holding both families, which
`schedules/comparator/` is (`K12/K20/K40` plus `uniform-K12/K20/K40`); its
`validation` phase runs any split without demanding a frozen selection, so it
is the ungated way to re-generate the comparator's held-out images:

```bash
python $BENCH/bench_qwen_image21_cache_comparison.py launch validation \
  --corpus $STUDY/corpus/corpus-comparator-v1.json --split heldout \
  --arms dp-K20 uniform-K20 cachedit-conservative \
  --schedule-dir $STUDY/schedules/comparator \
  --model-path $MODEL --source-dir $SRC --out runs/repro-cmp
```

**Score, report, inspect.** Scoring is CPU-only, and the comparator reuses the
DPCache harness's `score` and `contact-sheet` unchanged:

```bash
python $BENCH/bench_qwen_image21_dpcache.py score \
  --run runs/repro-cmp --out runs/repro-cmp/scores.json
python $BENCH/bench_qwen_image21_cache_comparison.py report \
  --run runs/repro-cmp --scores runs/repro-cmp/scores.json
python $BENCH/bench_qwen_image21_dpcache.py contact-sheet \
  --run runs/repro-cmp --scores runs/repro-cmp/scores.json \
  --arms native dp-K20 uniform-K20 cachedit-conservative \
  --worst-by dp-K20 --out worst-by-dpK20.jpg
```

**Freeze before the held-out set.** The selection is what keeps the held-out
split honest, and the harnesses enforce the order rather than trusting it.
`select` reads a validation run only, refuses to overwrite an existing
selection ("a selection is frozen once") and writes the file read-only:

```bash
python $BENCH/bench_qwen_image21_dpcache.py select \
  --run runs/repro-validation --scores runs/repro-validation/scores.json \
  --out runs/selection.json

python $BENCH/bench_qwen_image21_dpcache.py launch sweep --split heldout \
  --corpus $STUDY/corpus/corpus-v1.json --arms K12 K20 \
  --schedule-dir $STUDY/schedules/dp --selection runs/selection.json \
  --model-path $MODEL --source-dir $SRC --out runs/repro-heldout
```

The held-out launch then refuses to start unless the arm set is exactly the
frozen one, the corpus digest matches the one the selection was made on, and
every selected schedule still hashes to what was frozen. The comparator's
`select` additionally takes `--schedule-dir` and `--heldout-corpus` and records
the held-out corpus digest, and its `heldout` phase re-digests the validation
run and its scores before running anything. Freezing your own selection is the
supported path; the published records above are evidence of what was frozen,
not reusable inputs.

## Contents

| Path | What it is |
| --- | --- |
| `MANIFEST.json` | every published file, its raw digest, its published digest, and which fields (if any) were rewritten |
| `prepare_publish_artifacts.py` | the tool that produced this directory from the raw run artifacts |
| `corpus/` | the two frozen prompt/seed corpora (`corpus-v1` for the original study, `corpus-comparator-v1` for the comparison), byte-identical to the ones the runs used |
| `schedules/dp/` | the frozen DP schedules: `K` = 12, 16, 20, 24, 28, 32, 36, 39 |
| `schedules/comparator/` | the endpoint-matched uniform controls and the DP schedules they were derived from |
| `results/` | the machine-readable results, the frozen selections, and the cross-run exactness audit. The selections record what was frozen; they are not reusable as `--selection` inputs -- see "Reproducing this" |
| `figures/` | quality vs GPU-seconds, and worst-case galleries ordered by each DP arm; linked individually under "Figures" |

### On the published digests

The raw run artifacts are immutable and authoritative. A file here is byte
identical to its raw original unless `MANIFEST.json` marks it `rewritten`, in
which case the only change is that paths on the producing machine were
relativised, and **both** digests are recorded. A rewritten file therefore has a
different hash from the raw one by design; the raw hash is preserved in the
manifest so the two can always be matched up. Rewriting touches no result value,
and the schedules remain usable as-is: what the runtime validates is the
artifact's `request` block, which was not changed.

Per-image tensors, decoded outputs, run plans and logs are **not** published
here. They stay with the raw artifacts on the machine that produced them.
