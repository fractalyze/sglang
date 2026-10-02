// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// Qwen3.8's LM head, which the decode, verify, prefill and MTP kernels share:
// each CTA's rows of the vocabulary over the normalized residual, read by the
// head's format tag (weight.h).

#ifndef S2MK_CSRC_QWEN38_LM_HEAD_CUH_
#define S2MK_CSRC_QWEN38_LM_HEAD_CUH_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "gemv_core.cuh"
#include "int4_gemv_core.cuh"
#include "qwen38_layer.h"
#include "weight.h"

namespace s2mk {
namespace qwen38 {

constexpr int kLmHeadGroupsPerLane = kQwen38Dim / (32 * kInt4Group);
static_assert(kLmHeadGroupsPerLane * 32 * kInt4Group == kQwen38Dim);

// The int4 row loop kept out of line on its own, as Bf16Rows keeps the bf16
// one, so neither format's live registers raise the other's or the kernel's.
template <int kM, typename Emit>
__device__ __noinline__ void Int4LmHeadRows(const Weight& w, int begin,
                                            int end, const uint4* xs,
                                            const Emit& emit) {
  Int4ZpWarpRowsM<kLmHeadGroupsPerLane, kM>(static_cast<const uint4*>(w.data),
                                            w.scales, w.zeros, begin, end, xs,
                                            emit);
}

// emit(row, m, lm_head[row] · x_m) for this CTA's rows of a `vocab`-row LM
// head and each of kM normalized residuals held one after another at xs
// (kQwen38Dim bf16 each, in shared memory). The head is kBf16 or kInt4Zp; a
// logit is the same whichever CTA or warp owns its row, and for input m the
// one kM = 1 gives on x_m alone. Called by every thread, with no barrier: the
// caller makes xs visible first.
template <int kM, typename Emit>
__device__ __forceinline__ void LmHeadRows(const Weight& w, int vocab,
                                           const uint4* xs, const Emit& emit) {
  const int begin = RowBegin(vocab, blockIdx.x);
  const int end = RowBegin(vocab, blockIdx.x + 1);
  switch (w.format) {
    case WeightFormat::kBf16:
      Bf16Rows<kQwen38Dim, kM>(static_cast<const __nv_bfloat16*>(w.data),
                               begin, end, xs, emit);
      return;
    case WeightFormat::kInt4Zp:
      Int4LmHeadRows<kM>(w, begin, end, xs, emit);
      return;
  }
}

}  // namespace qwen38
}  // namespace s2mk

#endif  // S2MK_CSRC_QWEN38_LM_HEAD_CUH_
