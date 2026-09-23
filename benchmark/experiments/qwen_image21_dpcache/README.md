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

The uniform control (`uniform_full_steps`, `UNIFORM_GENERATOR`) lives in the
comparator, not in the runtime cache module. Nothing serves a uniform schedule;
it exists to separate what the calibrated placement buys from what the budget
buys. Regenerating the frozen `uniform-K12/K20/K40` artifacts from the
comparator reproduces them byte for byte.

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

What was re-checked after the rebase onto upstream `1d59ce7c9`: image outputs.
The rebased tree reproduces the original tested source bit-for-bit on the native
path and on the `K=20` path (8 validation pairs, `torch.equal` on decoded
samples and final latents plus identical PNG bytes), and reproduces pristine
upstream bit-for-bit with DPCache disabled and with an all-full schedule. So the
**quality** numbers below describe the current tree's outputs. **Timing was not
re-measured**, so every second and every speedup below belongs to `a9871012`.

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

10 calibration prompts, 8 validation pairs, and a held-out set of **20 pairs
from 10 prompts x 2 seeds**. All three splits are prompt-disjoint. The held-out
set is evaluated once, after the arm selection was frozen
(`results/selection-*.json`, written before the held-out run). Prompt categories
cover text rendering, counting and composition, people, complex scenes,
illustration and varied colour.

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

## Held-out results (20 pairs)

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

<!-- Figures are small review aids, not data: two worst-case galleries and one
     scatter. No tensors and no full-resolution outputs are published here. -->

## Limits

- 20 held-out pairs from 10 prompts, 4 timing prompts, one GPU, one checkpoint,
  one resolution, one step count. Small samples; treat as preliminary.
- The two LPIPS readings are this study's policy, not published thresholds.
- The RGBA policy is provisional; alpha is reported, not gated. `uniform-K12`
  (0.290) and `cachedit-stock` (0.278) showed the largest alpha deviation.
- A passing mean does not guarantee every image. On the held-out set `K=12`
  visibly changed the layout of some scenes and turned one of four coloured cups
  into a different object; `K=20` stayed visually close on every pair. Inspect
  `figures/` before choosing a budget.
- Timing was measured on `a9871012` and is not restated for the rebased tree.

## Contents

| Path | What it is |
| --- | --- |
| `MANIFEST.json` | every published file, its raw digest, its published digest, and which fields (if any) were rewritten |
| `prepare_publish_artifacts.py` | the tool that produced this directory from the raw run artifacts |
| `corpus/` | the frozen prompt/seed corpora, byte-identical to the ones the runs used |
| `schedules/dp/` | the frozen DP schedules, `K=12..39` |
| `schedules/comparator/` | the endpoint-matched uniform controls and the DP schedules they were derived from |
| `results/` | the machine-readable results, the frozen selections, and the cross-run exactness audit |
| `figures/` | quality vs GPU-seconds, and worst-case galleries ordered by each DP arm |

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
