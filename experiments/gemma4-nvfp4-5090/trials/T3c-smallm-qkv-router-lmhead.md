# T3c (gemma4nv-b2-t3c): extend the small-M BF16 GEMM to qkv_proj, the router and lm_head

**Status: retired (W7, 2026-10-03).** Gate `T3c-20261003-095735-build-server-2-fb82c2`: W1 TPOT -0.92% (bar 1%), W8 composite +0.83%, W32 flat, fidelity pass. Code removed again in `ed0aefcd40`; see `REPORT-bs2-w7.md`. Below is the frozen registration.

Registered 2026-10-03 by W7 (bs2) before any gate run. Code: `826d472504` on
`jumanzii/gemma4nv-analysis`, behind the existing switch `SGLANG_OPT_USE_TRITON_SMALL_M_BF16_GEMM`
(T3's allowlist plus three shapes, so T3c is a commit change, not a new switch).

## Control and candidate

- Control: gate ref `smallm-gemm`, which is base2 + T3 on `1fd77e64b0` with the switch on.
- Candidate: the same flags and switch on `826d472504`.

## Evidence (W7 microbench, `t3/gemm_microbench_w7.py`)

Results: `t3/results/w7-t3c*.json`. The method is T3's: a CUDA graph of 64 calls, best of 7 replays, and
weights rotated over at least 256 MB. Three repeat runs agree within 0.25% on every cell, so
that is the noise.

| shape (N x K) | M=1 | M=8 | M=16 | M=32 | chosen config |
|---|---:|---:|---:|---:|---|
| qkv sliding (8192x2816), cuBLAS µs | 32.2 | 31.4 | 31.3 | 31.5 | |
| Triton single-pass | 29.4 (-8.7%) | 29.6 (-5.7%) | 29.7 (-5.3%) | 30.4 (-3.6%) | BN32 BK128 s4 |
| qkv full (10240x2816), cuBLAS | 39.4 | 37.9 | 37.8 | 37.8 | |
| Triton single-pass | 35.9 (-8.9%) | 36.2 (-4.3%) | 36.4 (-3.8%) | 37.0 (-2.1%) | BN32 BK128 s4 |
| router (128x2816), cuBLAS | 2.45 | 3.57 | 3.73 | 3.72 | |
| Triton split-K 8 + reduce | 2.26 (-7.8%) | 2.57 (-28.2%) | 2.66 (-28.8%) | 2.87 (-22.9%) | BN16 BK128 s3 sk8 |
| lm_head (262144x2816), cuBLAS | 933.2 | 881.4 | 884.9 | 912.9 | |
| Triton single-pass | 872.6 (-6.5%) | 877.8 (-0.4%) | 881.3 (-0.4%) | 886.8 (-2.9%) | BN32 BK256 s4 |

- **Router.** Single-pass Triton loses to cuBLAS by 30-110%: 4-8 N tiles cannot fill 170 SMs.
  Split-K with an fp32 partial and a reduce launch wins.
- **lm_head.** cuBLAS is at SOL at M=8 and M=16. It drops at M=1 and M=32, and those two are where Triton gains.

## Prediction (frozen)

Per-step savings are the microbench deltas times the launch counts: 25 sliding + 5 full qkv, 30 routers, 1 lm_head.

| metric (bs2, gate ratio of sums) | microbench sum | predicted |
|---|---|---|
| **W1 TPOT (deciding)** | 154 µs of 5.874 ms (qkv 87, router 6, lm_head 61) = 2.6% | **-1.5 … -3.5%** |
| W8 composite (guard) | 87 µs of ~8.8 ms decode = ~1.0% decode | +0.3 … +1.2% (below the 1% bar by design) |
| W32 tok/s (guard) | 84 µs per M=32 step | 0 … +2% |
| Fidelity | | reorder tier: the router's split-K changes its fp32 summation order, and qkv and lm_head match cuBLAS up to bf16 rounding |

Decision rule: run the gate with `--decide-on w1_tpot_gain`. W8 composite and W32 are no-regression guards at their bars.

The W8 composite gain is predicted to sit under the bar, so it cannot decide. W1 is where the shapes T3c touches move most:
- M=1 qkv and lm_head are where cuBLAS loses most.
- The router saves under 1% at M=1.

**Falsified** if any of these holds:
- W1 TPOT improves by less than its bar (1%);
- the W8 composite or W32 regresses past its bar;
- fidelity fails.
