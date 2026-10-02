// Copyright 2026 Fractalyze Inc. All rights reserved.

#ifndef S2MK_CSRC_QWEN38_MTP_H_
#define S2MK_CSRC_QWEN38_MTP_H_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "barrier.h"
#include "qwen38_layer.h"
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

}  // namespace s2mk

#endif  // S2MK_CSRC_QWEN38_MTP_H_
