// Copyright 2026 Fractalyze Inc. All rights reserved.

#ifndef S2MK_CSRC_QWEN38_MTP_H_
#define S2MK_CSRC_QWEN38_MTP_H_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "barrier.h"
#include "qwen38_layer.h"
#include "qwen38_prefill.h"
#include "weight.h"

namespace s2mk {

// One step of Qwen3.8-27B's multi-token-prediction head in one launch, as
// SGLang's Qwen3_5ForCausalLMMTP drafts: at position pos it takes the hidden
// state h of the token at pos (the model's final-normed residual, or the
// head's own output at the previous step) and the token x at pos + 1, and
// predicts the token at pos + 2:
//
//   r = fc([RMSNorm(embed[x]) × (1 + embed_norm), RMSNorm(h) × (1 + hidden_norm)])
//   r = the full-attention layer with its MLP, at position pos
//   hidden_out = RMSNorm(r) × (1 + final_norm);  logits = lm_head · hidden_out
//
// The checkpoint keeps the head in bf16, so every projection is kBf16. The
// layer's KV cache is its own, written at pos. Six phases and five grid
// barriers: the fc rows, the attention block's three phases, the MLP's two,
// then the final norm and the LM head. Every value is one CTA's fixed-order
// sum, so the step is deterministic.
struct Qwen38MtpParams {
  const int32_t* token;  // [1]: x
  // [1]: the cache position and the M-RoPE positions (a text token's).
  const int32_t* pos;
  const __nv_bfloat16* hidden_in;  // [kQwen38Dim]: h
  const __nv_bfloat16* embed;  // [vocab, kQwen38Dim]: the model's
  const __nv_bfloat16* embed_norm;  // [kQwen38Dim]: pre_fc_norm_embedding
  const __nv_bfloat16* hidden_norm;  // [kQwen38Dim]: pre_fc_norm_hidden
  Weight fc;  // kBf16 [kQwen38Dim, 2 × kQwen38Dim]
  // The layer, its weights kBf16; its residual, hidden, pos, positions and
  // barrier fields are unused.
  Qwen38LayerParams layer;
  const __nv_bfloat16* final_norm;  // [kQwen38Dim]: mtp.norm
  Weight lm_head;  // kBf16 or kInt4Zp [vocab, kQwen38Dim]: the model's
  int vocab;
  float eps;
  int64_t timeout_ns;

  float* residual;  // [kQwen38Dim]: workspace
  __nv_bfloat16* hidden_out;  // [kQwen38Dim]
  float* logits;  // [vocab], or null to skip the LM head
  unsigned* sync;  // kSyncWords words
  ErrorRecord* error;  // device view of the host-mapped record
};

// A plain launch of one CTA per SM (see LaunchCodePredictor on MPS).
cudaError_t LaunchQwen38Mtp(const Qwen38MtpParams& params, int num_ctas,
                            cudaStream_t stream);

// The head over num_tokens consecutive positions in one launch, as
// speculative decoding's draft extend runs it over a cycle's accepted tokens:
// token t takes h_t and x_t at position pos + t through Qwen38MtpParams's
// steps, writing its key and value to the head's cache, and the last token
// alone writes hidden_out and logits. The projections run on tensor cores
// (qwen38_tokens.cuh), every token through each weight at once, so a token's
// results are the same bits on every run, at any CTA count with the same
// attention chunks, and whatever tokens follow it, and match its one-token
// step's within bf16's error, not bitwise. Eight phases and seven grid
// barriers: the fc inputs' norms, the fc rows, the attention block's three
// phases, the MLP's two, then the final norm and the LM head.
struct Qwen38MtpExtendParams {
  // The head; its token, hidden_in and residual are unused, and pos is the
  // first token's.
  Qwen38MtpParams head;
  int num_tokens;  // in [1, kQwen38MaxVerifyTokens] (qwen38_decode.h)
  const int32_t* tokens;  // [num_tokens]: x_t
  const __nv_bfloat16* hidden_in;  // [num_tokens, kQwen38Dim]: h_t

  // Workspace, token-major.
  __nv_bfloat16* fc_in;  // [num_tokens, 2 × kQwen38Dim]
  float* residual;  // [num_tokens, kQwen38Dim]
  float* qkv;  // [num_tokens, kQwen38QkvRows]
  // [num_tokens, kQwen38QHeads × splits, 2], splits as the layer's own.
  float* partial_ml;
  float* partial_o;  // [num_tokens, kQwen38QHeads × splits, kQwen38HeadDim]
  __nv_bfloat16* act;  // [num_tokens, kQwen38Ffn]
};

// LaunchQwen38Mtp's launch for an extend.
cudaError_t LaunchQwen38MtpExtend(const Qwen38MtpExtendParams& params,
                                  int num_ctas, cudaStream_t stream);

// The head over a chunk of prompt positions in one launch: the key and value
// a step at each position pos0[0] + t writes to the layer's cache, from h_t,
// the model's final-normed hidden state there, and x_t, the token after it.
// In the prompt the head drafts nothing, so the cache is all a step leaves
// that is kept; with one layer, a position's key and value need no other
// position's, and the attention, the MLP and the final norm are skipped:
//
//   1. Per token (a CTA each): the two inputs' RMSNorms, side by side.
//   2. The fc rows of every token into the residual.
//   3. Per token: the layer's input RMSNorm into h.
//   4. The k and v rows of every token.
//   5. A warp per (token, KV head): the key's QK-norm and M-RoPE; key and
//      value, bf16, to the cache.
//
// The projections run on tensor cores (prefill_dense.cuh's FixedDense) and
// phases 3 and 5 are the prefill's (qwen38_prefill_per_token.cuh). Each sum
// has one order, so the result is bitwise the same at every CTA count; it
// differs from the step's, whose sums run in another order, in the last bits.
struct Qwen38MtpPromptParams {
  const int32_t* tokens;  // [chunk.tokens]: x
  const __nv_bfloat16* hidden;  // [chunk.tokens, kQwen38Dim]: h
  const __nv_bfloat16* embed;  // [vocab, kQwen38Dim]: the model's
  const __nv_bfloat16* embed_norm;  // [kQwen38Dim]: pre_fc_norm_embedding
  const __nv_bfloat16* hidden_norm;  // [kQwen38Dim]: pre_fc_norm_hidden
  Weight fc;  // kBf16 [kQwen38Dim, 2 × kQwen38Dim]
  // The layer, its weights kBf16; of the rest only its input_norm, k_norm,
  // eps, cos_sin and kv are read.
  Qwen38LayerParams layer;
  float eps;
  // [kQwen38PrefillChunk, 2 × kQwen38Dim]: fc's inputs, workspace.
  __nv_bfloat16* fc_in;
  // The chunk: tokens, pos0 and positions; residual, h and proj (only its k
  // and v rows) as workspace; timeout_ns, sync and error. Nothing else is
  // read.
  Qwen38PrefillParams chunk;
};

// A plain launch of `num_ctas` CTAs, one per SM at most.
cudaError_t LaunchQwen38MtpPrompt(const Qwen38MtpPromptParams& params,
                                  int num_ctas, cudaStream_t stream);

}  // namespace s2mk

#endif  // S2MK_CSRC_QWEN38_MTP_H_
