// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// One decode token through a full-attention layer of Qwen3.8-27B's language
// model: the attention block, then the dense MLP every layer shares. Every
// projection is asymmetric compressed-tensors W4A16 (kInt4Zp, weight.h); norms
// are bf16. s2mk/qwen38_layer.py holds the same widths.

#ifndef S2MK_CSRC_QWEN38_LAYER_H_
#define S2MK_CSRC_QWEN38_LAYER_H_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "barrier.h"
#include "thinker_attention.h"
#include "weight.h"

namespace s2mk {

constexpr int kQwen38Dim = 5120;
constexpr int kQwen38QHeads = 24;
constexpr int kQwen38KvHeads = 4;
constexpr int kQwen38HeadDim = 256;
// partial_rotary_factor 0.25: RoPE turns a head's first 64 dims.
constexpr int kQwen38Rotary = 64;
constexpr int kQwen38Ffn = 17408;
constexpr int kQwen38QDim = kQwen38QHeads * kQwen38HeadDim;
constexpr int kQwen38KvDim = kQwen38KvHeads * kQwen38HeadDim;
// q rows, k rows, v rows, then the output gate's rows: q_proj's per-head
// [q, gate] halves, regrouped so q, k, v keep the thinker's layout.
constexpr int kQwen38QkvRows = 2 * kQwen38QDim + 2 * kQwen38KvDim;

// The attention shape (thinker_attention.h): GQA 6:1 at 256 dims, partial
// rotate-half RoPE with interleaved M-RoPE sections 11/11/10, Gemma
// RMSNorm (1 + w) on q and k, and a sigmoid output gate.
struct Qwen38AttentionDims {
  static constexpr int kQHeads = kQwen38QHeads;
  static constexpr int kKvHeads = kQwen38KvHeads;
  static constexpr int kHead = kQwen38HeadDim;
  static constexpr int kRotary = kQwen38Rotary;
  static constexpr int kMropeHeight = 3 * 11;
  static constexpr int kMropeWidth = 3 * 10;
  static constexpr bool kZeroCenteredNorm = true;
  static constexpr bool kOutputGate = true;
};

// residual = h + down(SiLU(gate(n)) × up(n)), n = RMSNorm(h) × (1 + post_norm):
// the dense MLP every layer of the model ends with, in two phases and one
// grid barrier: the (gate, up) pairs, then the down rows. The last phase
// writes each residual row from the same row of h, so the two may be one
// buffer.
struct Qwen38MlpParams {
  const __nv_bfloat16* post_norm;  // [kQwen38Dim]: post_attention_layernorm
  // kInt4Zp [2 × kQwen38Ffn, kQwen38Dim]: gate and up rows interleaved, so
  // pair j is rows 2j and 2j + 1.
  Weight w13;
  Weight w2;  // kInt4Zp [kQwen38Dim, kQwen38Ffn]
  float eps;
  __nv_bfloat16* act;  // workspace, [kQwen38Ffn]: SiLU(gate) × up
};

// residual = MLP(h) after h = residual_in + o_proj(sigmoid(g) × attention(q,
// k, v)) over RMSNorm(residual_in) × (1 + input_norm). The token sits at cache
// position `pos` and attends to positions [0, pos]. Five phases and four grid
// barriers: QKV, then (query head, chunk) items, then the merge and O rows,
// then the MLP's two. Each output is one CTA's fixed-order sum, with no
// atomics, so the result is deterministic.
struct Qwen38LayerParams {
  const float* residual_in;  // [kQwen38Dim]
  const __nv_bfloat16* input_norm;  // [kQwen38Dim]: input_layernorm
  Weight wqkv;  // kInt4Zp [kQwen38QkvRows, kQwen38Dim]
  const __nv_bfloat16* q_norm;  // [kQwen38HeadDim]
  const __nv_bfloat16* k_norm;  // [kQwen38HeadDim]
  Weight wo;  // kInt4Zp [kQwen38Dim, kQwen38QDim]
  // [positions, kQwen38Rotary]: each position's 32 cosines, then its 32
  // sines.
  const __nv_bfloat16* cos_sin;
  const int32_t* positions;  // [3]: temporal, height, width
  PagedKv kv;
  int pos;
  int splits;  // chunks per query head
  Qwen38MlpParams mlp;
  float eps;
  int64_t timeout_ns;

  // Workspace.
  float* qkv;  // [kQwen38QkvRows]
  float* partial_ml;  // [kQwen38QHeads × splits, 2]: running max and sum
  float* partial_o;  // [kQwen38QHeads × splits, kQwen38HeadDim]

  // Outputs.
  float* hidden;  // [kQwen38Dim]: h, the residual after attention
  float* residual;  // [kQwen38Dim]
  unsigned* sync;  // kSyncWords words
  ErrorRecord* error;  // device view of the host-mapped record
};

// A plain launch of one CTA per SM (see LaunchCodePredictor on MPS).
cudaError_t LaunchQwen38Layer(const Qwen38LayerParams& params, int num_ctas,
                              cudaStream_t stream);

}  // namespace s2mk

#endif  // S2MK_CSRC_QWEN38_LAYER_H_
