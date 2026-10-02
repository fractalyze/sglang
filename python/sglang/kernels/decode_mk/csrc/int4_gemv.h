// Copyright 2026 Fractalyze Inc. All rights reserved.

#ifndef S2MK_CSRC_INT4_GEMV_H_
#define S2MK_CSRC_INT4_GEMV_H_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace s2mk {

// The width of the int4 GEMV LaunchInt4Gemv runs, Qwen3.8's residual.
constexpr int kInt4GemvK = 5120;
// Most rows a CTA of LaunchInt4Gemv may own.
constexpr int kInt4GemvMaxRowsPerCta = 64;

// y = W x in fp32 for an n × kInt4GemvK int4 W (int4_gemv_core.cuh): through
// Int4GemvRows when `zeros` is null, else through Int4ZpGemvRows, so the two
// cores can be held to each other. x is bf16 [kInt4GemvK]; each of the
// `num_ctas` CTAs owns at most kInt4GemvMaxRowsPerCta rows.
cudaError_t LaunchInt4Gemv(const int32_t* packed, const __nv_bfloat16* scales,
                           const uint32_t* zeros, const __nv_bfloat16* x,
                           float* y, int n, int num_ctas, cudaStream_t stream);

// Tokens LaunchInt4Mma takes at most: 8 n-tiles of mma.m16n8k16.
constexpr int kInt4MmaMaxTokens = 64;

// y = X W^T in fp32 for `tokens` ≤ kInt4MmaMaxTokens bf16 rows X ([tokens,
// kInt4GemvK]) and an n × kInt4GemvK W, n a multiple of 16, on the prefills'
// tensor-core tiles (int4_mma_core.cuh), a warp per 16 rows over all of k:
// W int4 through Int4MmaRows when `zeros` is null, else through
// Int4ZpMmaRows; or, with `packed` null, W bf16 (`bf16`, [n, kInt4GemvK])
// through Bf16MmaRows. y is [tokens, n].
cudaError_t LaunchInt4Mma(const int32_t* packed, const __nv_bfloat16* scales,
                          const uint32_t* zeros, const __nv_bfloat16* bf16,
                          const __nv_bfloat16* x, float* y, int n, int tokens,
                          cudaStream_t stream);

}  // namespace s2mk

#endif  // S2MK_CSRC_INT4_GEMV_H_
