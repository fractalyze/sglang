// Copyright 2026 Fractalyze Inc. All rights reserved.

#ifndef S2MK_CSRC_SLOW_AR_H_
#define S2MK_CSRC_SLOW_AR_H_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "barrier.h"
#include "kv_cache.h"
#include "layer.h"

namespace s2mk {

// A profiled launch writes one row of int64 stamps per CTA: the
// kProfileHeader words, then clock64 on arriving at and on leaving each
// barrier, then %globaltimer at the same moments. The two clocks' ratio over a
// phase is the SM clock the CTA ran that phase at.
constexpr int ProfileWords(int num_layers) {
  return kProfileHeader + 4 * kBarriersPerLayer * num_layers;
}

// One Slow AR step: `num_layers` layers on the token at `pos`, the final
// norm, and a `head_rows`-row LM head.
struct SlowArParams {
  const LayerWeights* layers;  // [num_layers], on the device
  int num_layers;
  const __nv_bfloat16* final_norm;  // [kDim]
  const __nv_bfloat16* head;  // [head_rows, kDim]
  int head_rows;
  const __nv_bfloat16* rope;  // [max positions, kHeadDim / 2, (cos, sin)]
  KvLayout kv;
  float eps;
  int pos;
  int step;  // reported by the watchdog
  // Bytes of each CTA's next GEMV slice to prefetch into L2; a multiple of 16.
  int prefetch_bytes;
  int64_t timeout_ns;

  // Workspace.
  float* residual;  // [kDim]: the decoder input in, the last layer's output out
  float* qkv;  // [kQkvRows]
  float* partial_ml;  // [attention items, 2]: running max and sum
  float* partial_o;  // [attention items, kHeadDim]: unnormalised output
  __nv_bfloat16* act;  // [kFfn]: SiLU(gate) · up
  float* logits;  // [head_rows]
  __nv_bfloat16* final_hidden;  // [kDim]: the final norm's output
  float* dump;  // [num_layers + 1, kDim] or null; the launch fills rows 1..
  int64_t* profile;  // [num_ctas, ProfileWords(num_layers)] or null
  unsigned* sync;  // kSyncWords words
  ErrorRecord* error;  // device view of the host-mapped record
};

// Attention work items for `num_ctas` CTAs: every query head split over the
// same number of sequence chunks.
inline int AttentionItems(int num_ctas) {
  return num_ctas / kQHeads * kQHeads;
}

cudaError_t LaunchSlowAr(const SlowArParams& params, int num_ctas,
                         cudaStream_t stream);

}  // namespace s2mk

#endif  // S2MK_CSRC_SLOW_AR_H_
