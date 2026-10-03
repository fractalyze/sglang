# W5 report: gate integrity fixes, base2 and base3 pinned, T2 and T3 confirmed (build-server-3)

**Outcome: done.** The gate fixes are committed with tests. base2 and then base3 are pinned,
each with a 6-pair A/A. T2 was re-evaluated on bs2 and kept, and both stacked trials (T2S, T3S)
were kept on bs3. Full numbers are in `BASELINE.md` (base3 and base2 sections).

## Gate fixes (branch `jumanzii/gemma4nv-gate`, 50 unit tests pass locally and on bs2/bs3)

| commit | fix |
|---|---|
| 888cc1ba6 | Server info is compared path by path. Each declared flag also declares the paths the server resolves from it: chunk size covers the prefill cuda-graph shape, pool fraction covers `max_running_requests`, and graph max bs covers the graph shapes. `launch_command` is checked flag by flag: a flag only one arm passes must be declared. This fixes T1's `cuda_graph_config` and T2's `launch_command` false flags. |
| 5522d7d6d | Timed-output agreement is a hard check only for A/A runs and for refs that declare `numerics_unchanged` (never inherited through `extends`). For every other ref it is reported, and the teacher-forced fidelity gate decides. The threshold is calibrated by `set-noise` from the A/A per-pair agreement: lowest pair minus max(3 sigma, 0.02). bs3 measured 1.0 in all 12 A/A pairs, so the threshold is 0.98. A new hard check requires every timed stream to reach its decode length. |
| 6c5581181 + a7ece6d73 | `bin/deploy` refuses a dirty tree, rsyncs the tree and writes `DEPLOY.json` with the commit and a content hash. `harness_commit` comes from git when `experiments/` is tracked and clean, then from the stamp while the tree still hashes to it, and otherwise is `tree-sha256:<hash>`. Before this, bs2's ledger recorded a9871012, the SGLang checkout's HEAD. A re-evaluated report also names its evaluating harness. |
| d18d74700 | `gate run/reevaluate --decide-on <metric>`, on the coordinator's call. The metric the frozen prediction names must clear its bar. Every other workload metric, W32 included, is a no-regression guard. The default stays the W8 composite with the W1 guard, so older verdicts keep their meaning. |

## Results

| run | control -> candidate | deciding metric | result | verdict |
|---|---|---|---|---|
| `T2-20261003-085810-build-server-2-22b5a6` (bs2, `reevaluate`) | base -> splits16 | W1 TPOT | gain 1.0199, W8 0.998, W32 1.001; integrity OK, fidelity pass | **promote** (was parked; numbers unchanged) |
| `AA-base2-20261003-092146-build-server-3-dce578` | base2 A/A | - | sigma 0.04%, agreement 1.0 | pinned base2 |
| `gemma4nv-b3-t2s-20261003-093801-build-server-3-0e5824` | base2 -> base2-splits16 | W1 TPOT | **-1.94%** (6.070 -> 5.952 ms, pairs 1.018-1.021); W8 0.9998, W32 1.005 | **kept** (predicted [-2.5, -1.3]%) |
| `gemma4nv-b3-t3s-20261003-095540-build-server-3-fe1492` | base2-splits16 -> base2-splits16-smallm | W8 composite | **1.0457** (decode 1.061, pairs 1.044-1.048); W1 -2.1%, W32 +1.7% | **kept** (predicted [+3.0, +5.5]%) |
| `AA-base3-20261003-101154-build-server-3-5644da` | base3 A/A | - | sigma <= 0.15%, agreement 1.0 | pinned base3 |

Against base, base3's A/A control legs show:

- **W8 decode step:** 9.29 -> 8.76 ms, with sol_fraction 0.60 -> 0.63.
- **W1 TPOT:** 6.075 -> 5.833 ms, with sol_fraction 0.52 -> 0.55.
- **W32:** 1021 -> 1551 tok/s.

These are separate A/As on one host; the gated deltas are the paired runs above.

## Findings

- **Teacher-forced fidelity is blind to decode-only kernels.** The forced pass is a prefill
  (M > 32). Its numbers are identical to four digits across base2, T2S and T3S, and only the
  decode-path KL check saw those changes. The gate already gates decode KL, so this is not a
  hole, but a decode-only trial should not be described as passing on the forced check.
- **Host safety:** 48 legs and 2 prebuilds ran under the protocol (peaks below are from the two A/As and the prebuilds).
  - Minimum MemAvailable was 49.2 GB, and swap never moved past its 0.13 GB starting value.
  - Peak load1 was 5.9.
  - Weight load is the RSS peak, at 10.9 GB.
  - With a warm FlashInfer cache no nvcc ran. New Triton kernels compiled at 0.35 GB.

## Vault (`$WORLD_MODEL_PATH`, my paths only, not pushed)

- Stacks:
  - `stack-a9871012a-gemma4nv-base2` (parent `stack-a9871012a-gemma4nv`), now with its A/A profile.
  - `stack-1fd77e64b-gemma4nv-base3` (parent base2).
- Trials:
  - `gemma4nv-b3-t2s` (variant of `gemma4nv-b2-t2`) and `gemma4nv-b3-t3s` (variant of
    `gemma4nv-b2-t3`): preregistered before their runs, recorded kept, with prose and claim
    evidence.
  - `gemma4nv-b2-t2`: re-recorded as kept, with a reason that names the gate fix.
  - `gemma4nv-b3-w32b`: result stack set to base2.
- `meta/ledgers.yaml` maps the T2S and T3S refs.
- Two raw imports of `gemma4nv-bs3`.
- `git status` was clean before each import.

## Notes

- I pushed the study branch `jumanzii/gemma4nv-gate` to `fractalyze` (fast-forward), so the
  registered prediction URLs resolve, as the earlier trials' URLs do. The vault was not pushed.
- The deployed harness lives at `/data/jooman/gemma4nv/src-w5/gemma4-nvfp4-5090` on bs3 and
  bs2. W3/W6's `src-gate` deploys were left untouched.
- On bs3, `reference/noise.json` is now base3's; base's and base2's are kept as
  `noise.base.json` and `noise.base2.json`. The fidelity reference stays base's, so stacked
  trials are judged against the original model.

## Left

- bs2's noise file has no agreement calibration yet, so its hard threshold falls back to 0.5
  until a bs2 A/A runs through `set-noise`.
- The split count could follow the batch instead of one global flag (see
  `c-gemma4nv-b1-decode-attention-needs-more-kv-splits`).
