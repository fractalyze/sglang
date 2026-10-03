# T2S (gemma4nv-b3-t2s): split-KV 16 stacked on base2, build-server-3

Registered before the run (W5, 2026-10-03). Variant of `gemma4nv-b2-t2` (bs2, against `base`).

- **Change:** gate ref `base2-splits16` = `base2` (base + `--mem-fraction-static 0.76`) plus
  `--triton-attention-num-kv-splits 16` (tree default 8). Control `base2`.
- **Deciding metric:** W1 TPOT (`gate run --decide-on w1_tpot_gain`); W8 composite and W32
  throughput are no-regression guards at their bars.
- **Prediction:** W1 TPOT -1.9%, interval [-2.5, -1.3]%. W8 composite within +-1%, W32
  throughput within +-1%.
- **Basis:** prior trial. On bs2 against `base`, T2 measured W1 TPOT -1.95% (gain 1.0199, per-pair
  sigma 0.026%). A single decoding sequence's Triton stage-1 grid is batch x heads x splits, and
  base2 changes only the KV pool size, which leaves the B=1 kernel grid and its sequence length
  as they were. bs3 and bs2 have the same GPU model; hosts compare by delta only.
- **Falsified if:** W1 TPOT improves by less than 1.0% (the bar), or W8 composite / W32
  throughput regress past their bars.
- **Numerics:** reorders the attention reduction (the ref is not numerics_unchanged); the
  teacher-forced fidelity gate decides fidelity.
