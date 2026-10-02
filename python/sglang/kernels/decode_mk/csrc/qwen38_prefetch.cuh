// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// The L2 prefetch of the next phase's weights a CTA of a Qwen3.8 layer stack
// (qwen38_decode.h) starts on arriving at a grid barrier, which the decode
// step and the verify step share.

#ifndef S2MK_CSRC_QWEN38_PREFETCH_CUH_
#define S2MK_CSRC_QWEN38_PREFETCH_CUH_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "gemv_core.cuh"
#include "qwen38_decode.h"
#include "qwen38_layer.cuh"
#include "thinker_layer.cuh"

namespace s2mk {
namespace qwen38 {

// Bytes of each CTA's slice of the next phase's weights fetched into L2 on
// arriving at a barrier, so the phase starts on warm rows while the barrier
// waits on the slowest CTA. A whole slice would not fit: one MLP projection
// alone outgrows the L2.
constexpr int64_t kPrefetchBytes = 128 << 10;

__device__ inline void PrefetchWeight(const Weight& w, int k, int begin,
                                      int rows) {
  if (w.format == WeightFormat::kBf16) {
    const int64_t row_bytes = int64_t{k} * sizeof(__nv_bfloat16);
    const int n = min(rows, static_cast<int>(kPrefetchBytes / row_bytes));
    if (n > 0) {
      thinker::PrefetchL2(
          static_cast<const __nv_bfloat16*>(w.data) + int64_t{begin} * k,
          n * row_bytes);
    }
    return;
  }
  // A kInt4Zp row's words, scales and zero points.
  const int row_bytes = k / 2 + k / kInt4Group * 2 + k / 64;
  PrefetchRows(w, k, begin,
                       min(rows, static_cast<int>(kPrefetchBytes / row_bytes)));
}

// This CTA's rows of an n-row weight of k columns.
__device__ inline void PrefetchShare(const Weight& w, int n, int k) {
  const int begin = RowBegin(n, blockIdx.x);
  PrefetchWeight(w, k, begin, RowBegin(n, blockIdx.x + 1) - begin);
}

// The first phase of layer `layer`: its input projection.
__device__ inline void PrefetchLayer(const Qwen38DecodeParams& p, int layer) {
  if (layer >= p.num_layers) return;
  if (Qwen38IsFull(layer)) {
    PrefetchShare(p.full[Qwen38FullIndex(layer)].wqkv, kQwen38QkvRows,
                  kDim);
  } else {
    PrefetchShare(p.linear[Qwen38LinearIndex(layer)].in_proj, kGdnInt4Rows,
                  kDim);
  }
}

// On arriving at barrier `index`, starts this CTA's share of the next phase's
// weights into L2. Barrier 0 follows the embedding; then each layer's five.
__device__ inline void PrefetchNext(const Qwen38DecodeParams& p, int index) {
  if (threadIdx.x != 0) return;
  if (index == 0) return PrefetchLayer(p, 0);
  const int layer = (index - 1) / kQwen38BarriersPerLayer;
  const bool full = Qwen38IsFull(layer);
  const Qwen38MlpParams& mlp =
      full ? p.full[Qwen38FullIndex(layer)].mlp
           : p.linear_mlp[Qwen38LinearIndex(layer)];
  // The full-attention block prefetches its own O rows, and attention and
  // the delta rule read little of the weights.
  switch ((index - 1) % kQwen38BarriersPerLayer) {
    case 1:  // next: the linear-attention block's output projection
      if (!full) {
        PrefetchShare(p.linear[Qwen38LinearIndex(layer)].out_proj,
                      kDim, kGdnValueDim);
      }
      break;
    case 2: {  // next: the gate-up pairs
      const int begin = RowBegin(kQwen38Ffn, blockIdx.x);
      PrefetchWeight(mlp.w13, kDim, 2 * begin,
                     2 * (RowBegin(kQwen38Ffn, blockIdx.x + 1) - begin));
      break;
    }
    case 3:  // next: the down rows
      PrefetchShare(mlp.w2, kDim, kQwen38Ffn);
      break;
    case 4:  // next: the next layer's input projection
      PrefetchLayer(p, layer + 1);
      break;
    default:
      break;
  }
}

}  // namespace qwen38
}  // namespace s2mk

#endif  // S2MK_CSRC_QWEN38_PREFETCH_CUH_
