// Copyright 2026 Fractalyze Inc. All rights reserved.

#ifndef S2MK_CSRC_QWEN38_PREFILL_H_
#define S2MK_CSRC_QWEN38_PREFILL_H_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "barrier.h"
#include "gdn.h"
#include "qwen38_decode.h"
#include "qwen38_layer.h"
#include "weight.h"

namespace s2mk {

// Tokens a prefill launch takes at most: 8 n-tiles of mma.m16n8k16, and the
// delta rule's chunk.
constexpr int kQwen38PrefillMaxTokens = 64;

// Grid barriers a linear-attention layer and a full-attention layer take, one
// after each of their phases, the MLP's included.
constexpr int kQwen38PrefillLinearBarriers = 9;
constexpr int kQwen38PrefillFullBarriers = 9;

// Cache positions an attention partial covers: a full layer splits each
// token's keys into spans of this many, merged after.
constexpr int kQwen38PrefillAttnSpan = 512;

// Grid barriers a prefill launch over `num_layers` layers takes.
__host__ __device__ constexpr int Qwen38PrefillBarriers(int num_layers) {
  const int full = num_layers / kQwen38FullInterval;
  return full * kQwen38PrefillFullBarriers +
         (num_layers - full) * kQwen38PrefillLinearBarriers;
}

// One prefill chunk of Qwen3.8-27B's language model in one launch: `tokens`
// prompt tokens through every layer on tensor cores (prefill_dense.cuh), then,
// when `logits` is set, the final norm and the LM head for the last token.
//
// The chunk's tokens sit at cache positions pos0[0] on. Each linear-attention
// layer runs the chunked gated delta rule from its conv and recurrent states
// and leaves them where token-by-token decode would; each full-attention
// layer writes the chunk's keys and values to its paged cache and attends
// causally over the cache. So a Qwen38DecodeParams over the same layer params
// decodes on from the state the chunk leaves.
//
// The layers' weights, states and caches come from the decode's block params
// (qwen38_decode.h); their workspace and residual fields are unused.
//
// Every sum has one order, whichever CTA takes it, so the result is bitwise
// deterministic and the same at every CTA count.
struct Qwen38PrefillParams {
  const GdnParams* linear;  // [linear layers], on the device
  const Qwen38MlpParams* linear_mlp;  // [linear layers], on the device
  const Qwen38LayerParams* full;  // [full layers], on the device
  int num_layers;
  int tokens;  // ≤ kQwen38PrefillMaxTokens
  const int32_t* pos0;  // [1]: the first token's cache position
  // [3, tokens]: each token's M-RoPE positions, temporal, height, width.
  const int32_t* positions;

  const __nv_bfloat16* final_norm;  // [kQwen38Dim], applied as 1 + w
  Weight lm_head;  // kBf16 or kInt4Zp [vocab, kQwen38Dim]
  int vocab;
  float eps;
  int64_t timeout_ns;

  // The chunk's embeddings in, as fp32; each layer's residual after.
  float* residual;  // [tokens, kQwen38Dim]
  // Workspace.
  __nv_bfloat16* h;  // [tokens, kQwen38Dim]: a norm's output
  // [tokens, kGdnInt4Rows]: the linear layers' q, k, v (before the conv)
  // and z; the full layers' q, k, v and gate rows (kQwen38QkvRows).
  float* proj;
  float* beta;  // [tokens, kGdnVHeads]: sigmoid(b)
  float* gate;  // [tokens, kGdnVHeads]: g, the log decay
  // [tokens, kGdnKHeads, kGdnHeadDim]: after the conv, SiLU and L2 norm, the
  // query also scaled by kGdnHeadDim^(-1/2).
  float* query;
  float* key;
  float* value;  // [tokens, kGdnValueDim]: after the conv and SiLU
  float* core;  // [tokens, kGdnValueDim]: the delta rule's readout
  // [tokens, kQwen38QDim]: the gated RMSNorm's output (linear layers), or
  // the gated attention output (full layers).
  __nv_bfloat16* attn;
  __nv_bfloat16* act;  // [tokens, kQwen38Ffn]: SiLU(gate) × up
  // Each (token, query head)'s attention partial over each span of
  // kQwen38PrefillAttnSpan positions: [tokens, kQwen38QHeads, max_spans] of
  // (running max, sum), and of kQwen38HeadDim unnormalized outputs. max_spans
  // spans cover every position the chunk reaches.
  float* partial_ml;
  float* partial_o;
  int max_spans;

  // Outputs.
  float* logits;  // [vocab]: the last token's; null skips the LM head
  // [num_layers + 1, tokens, kQwen38Dim]: the residual entering each layer
  // and the last layer's output; null skips it.
  float* hidden;
  // [num_ctas, Qwen38PrefillBarriers(num_layers), 2]: each CTA's globaltimer
  // on arriving at and leaving every grid barrier; null skips it.
  int64_t* profile;
  unsigned* sync;  // kSyncWords words
  ErrorRecord* error;  // device view of the host-mapped record
};

// The fewest CTAs a prefill launch may use: one per (value head, half of its
// state columns) for the delta rule.
constexpr int kQwen38PrefillMinCtas = 2 * kGdnVHeads;

// A plain launch of `num_ctas` CTAs, one per SM at most and at least
// kQwen38PrefillMinCtas.
cudaError_t LaunchQwen38Prefill(const Qwen38PrefillParams& params,
                                int num_ctas, cudaStream_t stream);

// One of the prefill's int4 projections alone, on the dense phase the prefill
// runs it with: y = x W^T in fp32 for `tokens` ≤ kQwen38PrefillMaxTokens
// bf16 rows x ([tokens, k]) and an n-row kInt4Zp W of width k, kQwen38Dim,
// kQwen38QDim or kQwen38Ffn, n a multiple of 16. Each row's k splits into
// the prefill's slices for k, or with `whole_k` into one, so a row's sum runs
// over its groups in order as a lone tile's does (int4_gemv.h). y is [tokens,
// n].
cudaError_t LaunchQwen38Projection(const Weight& w, int n, int k,
                                   const __nv_bfloat16* x, int tokens,
                                   bool whole_k, float* y, int num_ctas,
                                   cudaStream_t stream);

}  // namespace s2mk

#endif  // S2MK_CSRC_QWEN38_PREFILL_H_
