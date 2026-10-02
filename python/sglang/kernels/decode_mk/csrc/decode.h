// Copyright 2026 Fractalyze Inc. All rights reserved.

#ifndef S2MK_CSRC_DECODE_H_
#define S2MK_CSRC_DECODE_H_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "barrier.h"
#include "kv_cache.h"
#include "layer.h"
#include "sampler.h"

namespace s2mk {

constexpr int kCodebooks = 10;
constexpr int kCodebookSize = 4096;
// Fast AR passes that end in a codebook head: passes 1..9.
constexpr int kCodebookHeads = kCodebooks - 1;

// A profiled decode launch writes one row of int64 words per CTA: the
// kProfileHeader words, then kDecodeStepWords per step: clock64 at the step's
// start, when the Slow AR and LM head's last barrier releases, when the
// sampler finishes and at the step's end; then the clock64 cycles spent
// waiting at barriers in the Slow AR (with the LM head) and in the Fast AR;
// then the barriers each of those two sections passed.
constexpr int kDecodeStepWords = 8;

constexpr int DecodeProfileWords(int num_steps) {
  return kProfileHeader + kDecodeStepWords * num_steps;
}

// The most grid barriers one decode step passes.
constexpr int DecodeBarriers(int slow_layers, int fast_layers) {
  return kBarriersPerLayer * slow_layers + 1 +
         kCodebooks * kShortBarriersPerLayer * fast_layers + kCodebookHeads;
}

// `num_steps` decode steps of S2 Pro in one launch: step s embeds its input
// column (the token and its 10 codes), runs the Slow AR at position
// pos0 + s, the trimmed LM head and the sampler, then the 10 Fast AR passes.
// Step s + 1's input is step s's token and codes: the recorded ones when
// teacher-forced, else the kernel's own, and a free-running launch stops
// after the step that draws <|im_end|>.
struct DecodeParams {
  // The Slow AR and its LM head.
  const LayerWeights* slow_layers;  // [num_slow_layers], on the device
  int num_slow_layers;
  const __nv_bfloat16* final_norm;  // [kDim]
  const __nv_bfloat16* head;  // [kNumSampleable, kDim]
  const __nv_bfloat16* rope;  // [max positions, kHeadDim / 2, (cos, sin)]
  KvLayout kv;
  const __nv_bfloat16* embeddings;  // [vocab, kDim]
  // [kCodebooks × kCodebookSize, kDim]: codebook c's rows start at c × 4096.
  const __nv_bfloat16* codebook_embeddings;

  // The Fast AR: its layers have no QK-norm.
  const LayerWeights* fast_layers;  // [num_fast_layers], on the device
  int num_fast_layers;
  const __nv_bfloat16* fast_norm;  // [kDim]
  const __nv_bfloat16* fast_head;  // [kCodebookSize, kDim]
  const __nv_bfloat16* fast_embeddings;  // [kCodebookSize, kDim]
  const __nv_bfloat16* fast_rope;  // [kCodebooks, kHeadDim / 2, (cos, sin)]
  KvLayout fast_kv;  // max_seq = kCodebooks
  // [kCodebookSize, kQkvRows]: the first layer's q, k and v for each code, as
  // LaunchBuildFastQkvTable builds them for this launch's grid; or null to
  // run that layer's QKV GEMV in every pass.
  const float* fast_qkv_table;

  float eps;
  SamplingParams sampling;
  // [history_count]: the sampler's history, oldest first.
  const int64_t* history;
  int history_count;
  int sample_step;  // the first step's draw index
  int pos0;
  int num_steps;
  const int64_t* first_column;  // [1 + kCodebooks]: step 0's input
  // [num_steps] and [num_steps, kCodebooks]: the recorded tokens and codes,
  // or null to run free.
  const int64_t* forced_semantic;
  const int64_t* forced_codes;
  // Bytes of each CTA's next GEMV slice to prefetch into L2; a multiple of 16.
  int prefetch_bytes;
  int64_t timeout_ns;

  // Workspace.
  float* residual;  // [kDim]
  float* qkv;  // [kQkvRows]
  float* partial_ml;  // [attention items, 2]
  float* partial_o;  // [attention items, kHeadDim]
  __nv_bfloat16* act;  // [kFfn]
  uint64_t* argmax;  // [num_ctas]: each CTA's best codebook row

  // Outputs, per step.
  float* semantic_logits;  // [num_steps, kNumSampleable]
  float* codebook_logits;  // [num_steps, kCodebookHeads, kCodebookSize]
  __nv_bfloat16* final_hidden;  // [num_steps, kDim]: the final norm's output
  int64_t* semantic;  // [num_steps]: the drawn token
  int64_t* codes;  // [num_steps, kCodebookHeads]: the codebook argmaxes
  int* steps_run;  // [1]
  // [num_steps, num_slow_layers + 1, kDim]: the decoder input, then each
  // layer's output; and [num_steps, kCodebooks, num_fast_layers, kDim]; or
  // null.
  float* slow_dump;
  float* fast_dump;
  int64_t* profile;  // [num_ctas, DecodeProfileWords(num_steps)] or null
  unsigned* sync;  // kSyncWords words
  ErrorRecord* error;  // device view of the host-mapped record
};

// Part of a weight the decode launch keeps in persisting L2 lines: `bytes`
// of the `window_bytes` from `base`, spread evenly over them, so every CTA's
// slice of the GEMV reading them is partly resident. Every Fast AR pass
// reads the same layers, so a pinned slice of them is read from HBM once a
// step. bytes = 0 pins nothing.
struct L2Pin {
  const void* base;
  size_t window_bytes;
  size_t bytes;
};

// Fills `table` ([kCodebookSize, kQkvRows] fp32) with the first Fast AR
// layer's q, k and v for each code's fast embedding, before RoPE. GEMV rows
// sum in an order set by how the grid splits them, so the table matches the
// decode's own QKV GEMV bit for bit only for a decode of `num_ctas` CTAs.
cudaError_t LaunchBuildFastQkvTable(const LayerWeights* fast_layers,
                                    const __nv_bfloat16* fast_embeddings,
                                    float eps, float* table, int num_ctas,
                                    cudaStream_t stream);

// Launches the decode. A pin reserves `pin.bytes` of the device's L2 for
// persisting lines (cudaLimitPersistingL2CacheSize), a device-wide limit
// that stays set after the launch.
cudaError_t LaunchDecode(const DecodeParams& params, const L2Pin& pin,
                         int num_ctas, cudaStream_t stream);

}  // namespace s2mk

#endif  // S2MK_CSRC_DECODE_H_
