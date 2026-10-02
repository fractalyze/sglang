// Copyright 2026 Fractalyze Inc. All rights reserved.

#ifndef S2MK_CSRC_GDN_H_
#define S2MK_CSRC_GDN_H_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "barrier.h"
#include "weight.h"

namespace s2mk {

// Qwen3.8's linear-attention (gated delta rule) layers: the residual width,
// 16 key heads shared by 48 value heads, and a causal depthwise conv of width
// 4 over q, k and v.
constexpr int kGdnDim = 5120;
constexpr int kGdnKHeads = 16;
constexpr int kGdnVHeads = 48;
constexpr int kGdnHeadDim = 128;
constexpr int kGdnConvWidth = 4;
constexpr int kGdnKeyDim = kGdnKHeads * kGdnHeadDim;
constexpr int kGdnValueDim = kGdnVHeads * kGdnHeadDim;
// The conv's channels: q, then k, then v.
constexpr int kGdnConvDim = 2 * kGdnKeyDim + kGdnValueDim;
// The input projection's rows: q, k, v (the conv's channels), then z, which
// the checkpoint quantizes, then the gates b and a, one per value head, which
// it keeps in bf16.
constexpr int kGdnInt4Rows = kGdnConvDim + kGdnValueDim;
constexpr int kGdnGateRows = 2 * kGdnVHeads;

// Fewest CTAs a launch may use: below it a CTA owns more state columns than
// its warps hold at once (gdn_block.cuh).
constexpr int kGdnMinCtas = 96;

// One token through a linear-attention layer in one launch:
// residual = residual_in + out_proj(GatedRmsNorm(DeltaRule(...), z)) over
// h = RMSNorm(residual_in) × (1 + norm). The projections are stored as
// Qwen3.8's int4 checkpoint keeps them (weight.h). Three phases and two grid
// barriers: the input projection with each channel's conv and gates, then the
// delta rule on each CTA's state columns, then the output projection. No
// atomics, and no value is summed across CTAs, so the result is deterministic
// and the same at every CTA count.
// The last phase writes each residual row from the same row of residual_in, so
// the two may be one buffer.
struct GdnParams {
  const float* residual_in;  // [kGdnDim]
  const __nv_bfloat16* norm;  // [kGdnDim]: input_layernorm, applied as 1 + norm
  Weight in_proj;  // kInt4Zp [kGdnInt4Rows, kGdnDim]: q, k, v, then z
  Weight gates;  // kBf16 [kGdnGateRows, kGdnDim]: b, then a
  const __nv_bfloat16* conv;  // [kGdnConvDim, kGdnConvWidth], oldest tap first
  const float* a_log;  // [kGdnVHeads]
  const float* dt_bias;  // [kGdnVHeads]
  const __nv_bfloat16* out_norm;  // [kGdnHeadDim]: the gated RMSNorm's weight
  Weight out_proj;  // kInt4Zp or kBf16 [kGdnDim, kGdnValueDim]
  float eps;
  int64_t timeout_ns;

  // State, read and written in place.
  // [kGdnConvDim, kGdnConvWidth - 1]: past inputs, oldest first.
  float* conv_state;
  // [kGdnVHeads, dv, dk], both kGdnHeadDim: S of each value head stored by
  // column, so a column's dk values, and a run of columns, are contiguous.
  float* state;

  // Workspace.
  float* mixed;  // [kGdnConvDim]: q, k and v after the conv and SiLU
  float* z;  // [kGdnValueDim]: the output gate
  float* beta;  // [kGdnVHeads]: sigmoid(b)
  float* decay;  // [kGdnVHeads]: exp(g)
  float* core;  // [kGdnValueDim]: the delta rule's readout

  // Outputs.
  float* residual;  // [kGdnDim]
  unsigned* sync;  // kSyncWords words
  ErrorRecord* error;  // device view of the host-mapped record
};

// A cooperative launch of `num_ctas` CTAs, kGdnMinCtas or more.
cudaError_t LaunchGdn(const GdnParams& params, int num_ctas,
                      cudaStream_t stream);

}  // namespace s2mk

#endif  // S2MK_CSRC_GDN_H_
