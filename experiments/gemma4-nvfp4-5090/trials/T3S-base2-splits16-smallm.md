# T3S (gemma4nv-b3-t3s): Triton small-M BF16 GEMM stacked on base2 + split-KV 16, build-server-3

Registered before the run (W5, 2026-10-03). Variant of `gemma4nv-b2-t3` (bs2, kept against `base2`).

- **Change:** candidate `base2-splits16-smallm` = control `base2-splits16` on SGLang commit
  1fd77e64b0 (a9871012 + `triton_small_m_bf16_gemm.py`, routed in `UnquantizedLinearMethod.apply`)
  with `SGLANG_OPT_USE_TRITON_SMALL_M_BF16_GEMM=1`.
- **Deciding metric:** W8 composite (gate default); W1 TPOT is the guard.
- **Prediction:** W8 composite gain +4.3%, interval [+3.0, +5.5]%. W1 TPOT -1.9% (guard; no
  regression). W32 +1 to +2% (ungated).
- **Basis:** prior trial. On bs2 against base2, T3 measured W8 composite 1.0445 (decode 1.060),
  W1 -1.98% and W32 +1.7%. Split-KV 16 changes only decode attention, and T3 changes only the
  M <= 32 o_proj and dense-MLP GEMMs, so the two touch disjoint kernels. The interval is
  widened for the host change (same GPU model).
- **Falsified if:** the W8 composite gain is below +1.0% (the bar), or W1 TPOT regresses past
  its bar.
- **Numerics:** reorders the GEMM reduction (fp32 accumulation in K order); the ref is not
  numerics_unchanged, so the teacher-forced fidelity gate decides.
