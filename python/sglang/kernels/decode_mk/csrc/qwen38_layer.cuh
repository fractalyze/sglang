// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// Qwen3.8's full-attention block and dense MLP block (qwen38_layer.h) as
// device functions every CTA of a persistent launch calls.
//
// Attention, three phases, on thinker_layer.cuh's attention helpers at
// Qwen38AttentionDims:
//   1. Every CTA normalizes the residual itself and computes its share of the
//      q, k, v and gate rows. Barrier.
//   2. Query head h and chunk s form item h × S + s, one per CTA: QK-norm and
//      partial M-RoPE, then attention over its chunk. Barrier.
//   3. Every CTA merges all partials, gates them, and computes its share of
//      the O rows, added to the residual.
//
// MLP, two phases:
//   1. Every CTA normalizes the residual itself; CTA c computes the (gate, up)
//      pairs [RowBegin(kQwen38Ffn, c), RowBegin(kQwen38Ffn, c + 1)) and
//      writes SiLU(gate) × up. Barrier.
//   2. Every CTA stages all of SiLU(gate) × up and computes its share of the
//      down rows, added to the residual.
//
// Each block's last phase writes the residual rows RowBegin(kQwen38Dim, cta)
// from the same rows of its input, so the two may be one buffer.

#ifndef S2MK_CSRC_QWEN38_LAYER_CUH_
#define S2MK_CSRC_QWEN38_LAYER_CUH_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "decoder_layer.cuh"
#include "gemv_core.cuh"
#include "int4_gemv_core.cuh"
#include "qwen38_layer.h"
#include "thinker_layer.cuh"

namespace s2mk {
namespace qwen38 {

constexpr int kDim = kQwen38Dim;
constexpr int kFfn = kQwen38Ffn;
constexpr int kQkvRows = kQwen38QkvRows;
// Int4Rows's groups per lane, with 32 lanes a row, at each input width.
constexpr int kDimGroupsPerLane = kDim / (32 * kInt4Group);
constexpr int kQGroupsPerLane = kQwen38QDim / (32 * kInt4Group);
constexpr int kFfnGroupsPerLane = kFfn / (32 * kInt4Group);
static_assert(kDimGroupsPerLane * 32 * kInt4Group == kDim);
static_assert(kQGroupsPerLane * 32 * kInt4Group == kQwen38QDim);
static_assert(kFfnGroupsPerLane * 32 * kInt4Group == kFfn);
static_assert(kDim <= kQwen38QDim, "AttentionShared::xs holds the residual");

// The residual's RMSNorms: Gemma's, scaling by 1 + w.
struct NormDims {
  static constexpr int kDim = kQwen38Dim;
  static constexpr bool kZeroCenteredNorm = true;
};

using AttentionShared = thinker::AttentionShared<Qwen38AttentionDims>;

struct MlpShared {
  uint4 x[kFfn / kVecElems];  // the normalized residual, then SiLU(gate) × up
  float ys[2 * kMaxRowsPerCta];
  float red[kWarps];
};

union Shared {
  AttentionShared attention;
  MlpShared mlp;
};

// Starts fetching `rows` rows from `begin` of a kInt4Zp weight of k columns
// into L2: its words, scales and zero points. Every Qwen3.8 width keeps a
// row's zero points, k / 64 bytes, a 16-byte multiple as the bulk prefetch
// requires. One thread calls it.
__device__ __forceinline__ void PrefetchRows(const Weight& w, int k, int begin,
                                             int rows) {
  if (rows <= 0) return;
  thinker::PrefetchInt4Rows(static_cast<const int32_t*>(w.data), w.scales, k,
                            begin, rows);
  thinker::PrefetchL2(w.zeros + int64_t{begin} * (k / 256),
                      int64_t{rows} * (k / 64));
}

// The kInt4Zp GEMV rows [begin, begin + rows) of `w` over xs into ys, its k
// kGroupsPerLane × 32 groups.
template <int kGroupsPerLane>
__device__ __forceinline__ void Int4Rows(const Weight& w, int begin, int rows,
                                         const uint4* xs, float* ys) {
  Int4ZpGemvRows<1, 32, kGroupsPerLane>(static_cast<const uint4*>(w.data),
                                        w.scales, w.zeros, begin, rows, xs,
                                        ys);
}

// The attention block on every CTA, calling sync() for its two barriers.
template <typename Sync>
__device__ void AttentionBlock(const Qwen38LayerParams& p,
                               const thinker::BlockStep& step, Sync& sync,
                               AttentionShared& sh) {
  const int cta = blockIdx.x;
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(sh.xs);
  const int o_begin = RowBegin(kDim, cta);
  const int o_rows = RowBegin(kDim, cta + 1) - o_begin;

  // 1. RMSNorm, then this CTA's q, k, v and gate rows.
  RmsNorm<NormDims>(step.residual_in, nullptr, p.input_norm, p.eps, xs,
                    nullptr, sh.red);
  {
    const int begin = RowBegin(kQkvRows, cta);
    const int rows = RowBegin(kQkvRows, cta + 1) - begin;
    Int4Rows<kDimGroupsPerLane>(p.wqkv, begin, rows, sh.xs, sh.ys);
    for (int i = threadIdx.x; i < rows; i += kThreads) {
      p.qkv[begin + i] = sh.ys[i];
    }
  }
  // Attention reads little of HBM, so O's rows stream in under it.
  if (threadIdx.x == 0) {
    PrefetchRows(p.wo, kQwen38QDim, o_begin, o_rows);
  }
  sync();

  // 2. This CTA's (head, chunk) item.
  thinker::Attend(p, step, sh);
  sync();

  // 3. The merged, gated attention output, then this CTA's O rows.
  thinker::Merge(p, sh, xs);
  Int4Rows<kQGroupsPerLane>(p.wo, o_begin, o_rows, sh.xs, sh.ys);
  for (int i = threadIdx.x; i < o_rows; i += kThreads) {
    step.residual[o_begin + i] =
        __ldcg(step.residual_in + o_begin + i) + sh.ys[i];
  }
}

// The MLP block on every CTA, reading `residual_in` and writing `residual`,
// calling sync() for its barrier.
template <typename Sync>
__device__ void MlpBlock(const Qwen38MlpParams& p, const float* residual_in,
                         float* residual, Sync& sync, MlpShared& sh) {
  const int cta = blockIdx.x;

  // 1. RMSNorm, then this CTA's (gate, up) pairs: rows 2j and 2j + 1.
  RmsNorm<NormDims>(residual_in, nullptr, p.post_norm, p.eps,
                    reinterpret_cast<__nv_bfloat16*>(sh.x), nullptr, sh.red);
  {
    const int begin = RowBegin(kFfn, cta);
    const int pairs = RowBegin(kFfn, cta + 1) - begin;
    Int4Rows<kDimGroupsPerLane>(p.w13, 2 * begin, 2 * pairs, sh.x, sh.ys);
    for (int i = threadIdx.x; i < pairs; i += kThreads) {
      p.act[begin + i] = __float2bfloat16(Silu(sh.ys[2 * i]) * sh.ys[2 * i + 1]);
    }
  }
  sync();

  // 2. Every pair's SiLU(gate) × up, then this CTA's down rows.
  for (int i = threadIdx.x; i < kFfn / kVecElems; i += kThreads) {
    sh.x[i] = __ldcg(reinterpret_cast<const uint4*>(p.act) + i);
  }
  {
    const int begin = RowBegin(kDim, cta);
    const int rows = RowBegin(kDim, cta + 1) - begin;
    Int4Rows<kFfnGroupsPerLane>(p.w2, begin, rows, sh.x, sh.ys);
    for (int i = threadIdx.x; i < rows; i += kThreads) {
      residual[begin + i] = __ldcg(residual_in + begin + i) + sh.ys[i];
    }
  }
}

}  // namespace qwen38
}  // namespace s2mk

#endif  // S2MK_CSRC_QWEN38_LAYER_CUH_
