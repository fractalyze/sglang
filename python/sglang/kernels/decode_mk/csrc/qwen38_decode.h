// Copyright 2026 Fractalyze Inc. All rights reserved.

#ifndef S2MK_CSRC_QWEN38_DECODE_H_
#define S2MK_CSRC_QWEN38_DECODE_H_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "barrier.h"
#include "gdn.h"
#include "qwen38_layer.h"
#include "weight.h"

namespace s2mk {

// Layer i of Qwen3.8-27B's language model is full attention when
// i mod kQwen38FullInterval = kQwen38FullInterval − 1, else linear attention
// (gated delta rule); every layer ends with the dense MLP.
constexpr int kQwen38FullInterval = 4;

__host__ __device__ constexpr bool Qwen38IsFull(int layer) {
  return layer % kQwen38FullInterval == kQwen38FullInterval - 1;
}

// A full layer's index among the full layers.
__host__ __device__ constexpr int Qwen38FullIndex(int layer) {
  return layer / kQwen38FullInterval;
}

// A linear layer's index among the linear layers.
__host__ __device__ constexpr int Qwen38LinearIndex(int layer) {
  return layer - layer / kQwen38FullInterval;
}

// Grid barriers a layer: the two inside either attention block, the one
// after it, the one inside the MLP and the one after it.
constexpr int kQwen38BarriersPerLayer = 5;

// One decode step of Qwen3.8-27B's language model in one launch: the token's
// embedding, every layer (gdn_block.cuh's linear-attention block or
// qwen38_layer.cuh's full-attention block, then the MLP), the final norm and
// the LM head.
//
// Each layer's blocks come as their own params, built once for the layer's
// weights, state, cache and workspace, and stacked on the device: layer i
// takes full[Qwen38FullIndex(i)] when it is full attention, else linear[j]
// and linear_mlp[j] for j = Qwen38LinearIndex(i). The step
// supplies what changes per step, read from device memory so a CUDA graph can
// replay the launch, and the kernel points every block's residual at
// `residual`, ignoring the blocks' own residual and position fields.
//
// No value is summed across CTAs except in a fixed order, so the step is
// bitwise deterministic, and the same at every CTA count for a given number
// of attention chunks.
struct Qwen38DecodeParams {
  const GdnParams* linear;  // [linear layers], on the device
  const Qwen38MlpParams* linear_mlp;  // [linear layers], on the device
  const Qwen38LayerParams* full;  // [full layers], on the device
  int num_layers;

  // The token, the cache position it sits at (it attends to [0, pos]), and
  // its M-RoPE positions (temporal, height, width).
  const int32_t* token;  // [1]
  const int32_t* pos;  // [1]
  const int32_t* positions;  // [3]

  const __nv_bfloat16* embed;  // [vocab, kQwen38Dim]
  const __nv_bfloat16* final_norm;  // [kQwen38Dim], applied as 1 + w
  Weight lm_head;  // kBf16 or kInt4Zp [vocab, kQwen38Dim]
  int vocab;
  float eps;
  int64_t timeout_ns;

  float* residual;  // [kQwen38Dim]: the last layer's output
  float* logits;  // [vocab]
  // [num_layers + 1, kQwen38Dim]: the residual entering each layer and the
  // last layer's output; null skips it.
  float* hidden;
  unsigned* sync;  // kSyncWords words
  ErrorRecord* error;  // device view of the host-mapped record
};

// A plain launch of one CTA per SM (see LaunchCodePredictor on MPS), at
// least kGdnMinCtas of them.
cudaError_t LaunchQwen38Decode(const Qwen38DecodeParams& params, int num_ctas,
                               cudaStream_t stream);

// The most tokens one verify step takes: shared memory holds that many
// tokens' attention outputs beside the attention block's own buffers.
constexpr int kQwen38MaxVerifyTokens = 4;

// num_tokens consecutive text tokens through the decode step's layers in one
// launch, as speculative decoding verifies a token and its drafts: every
// projection streams its weights once for all of them, and each token's
// logits, residual and states are bitwise those of decoding the tokens one
// step at a time.
//
// `step` carries the layers, the embedding, the final norm, the LM head and
// the barrier as a decode step takes them, with these differences:
//   - step.token holds num_tokens tokens, token t at cache position
//     *step.pos + t, at M-RoPE positions all *step.pos + t; step.positions
//     and step.hidden are unused.
//   - step.residual is [num_tokens, kQwen38Dim] and step.logits
//     [num_tokens, vocab].
//   - Each linear layer keeps num_slots copies of its states, one after
//     another from GdnParams::conv_state and GdnParams::state. The step reads
//     slot *slot and writes token t's states to slot
//     (*slot + 1 + t) mod num_slots, leaving slot *slot as it was, so a caller
//     that accepts the first n tokens continues from slot
//     (*slot + n) mod num_slots without recomputing anything.
//   - The layers' workspace fields are unused; every layer shares the
//     workspace below, sized for num_tokens.
// Full layers write each token's key and value at its position; a caller that
// rejects a token overwrites them with the next step.
struct Qwen38VerifyParams {
  Qwen38DecodeParams step;
  int num_tokens;  // in [1, kQwen38MaxVerifyTokens]
  int num_slots;  // more than num_tokens
  const int32_t* slot;  // [1]

  // Workspace, token-major.
  float* mixed;  // [num_tokens, kGdnConvDim]
  float* z;  // [num_tokens, kGdnValueDim]
  float* beta;  // [num_tokens, kGdnVHeads]
  float* decay;  // [num_tokens, kGdnVHeads]
  float* core;  // [num_tokens, kGdnValueDim]
  float* qkv;  // [num_tokens, kQwen38QkvRows]
  // [num_tokens, kQwen38QHeads × splits, 2], splits as the full layers' own.
  float* partial_ml;
  float* partial_o;  // [num_tokens, kQwen38QHeads × splits, kQwen38HeadDim]
  __nv_bfloat16* act;  // [num_tokens, kQwen38Ffn]

  // [num_tokens, kQwen38Dim]: each token's final-normed residual, the LM
  // head's input, as the MTP head reads the model's hidden state.
  __nv_bfloat16* final_hidden;
};

// LaunchQwen38Decode's launch for a verify step.
cudaError_t LaunchQwen38Verify(const Qwen38VerifyParams& params, int num_ctas,
                               cudaStream_t stream);

}  // namespace s2mk

#endif  // S2MK_CSRC_QWEN38_DECODE_H_
