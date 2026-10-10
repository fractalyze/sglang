// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// Qwen3.8's full-attention block and dense MLP block (qwen38_layer.h) over kT
// consecutive tokens in one persistent launch, as device functions every CTA
// calls: the verify step's (qwen38_verify.cu) over the model's kInt4Zp
// layers, and the MTP head's extend (qwen38_mtp.cu) over its kBf16 layer.
// Each projection runs on tensor cores (prefill_dense.cuh's FixedDenseRows),
// the kT tokens one n-tile of mma.m16n8k16, so a CTA streams its 16-row blocks
// of the weight once for every token. A row's sum runs over k's slices in one
// order whichever CTA takes it, and a token's sums are its own column of the
// n-tile, so a token's results are the same bits on every run, at any CTA
// count with the same attention chunks, and whatever tokens follow it.
// Attention splits each token's [0, pos + t] into chunks as a step of that
// token alone would, and a (head, chunk) CTA reads its chunks' keys and values
// once for all the tokens. The tokens' own keys and values come from shared
// memory, rounded to bf16 as a step of one token reads them back from the
// cache.

#ifndef S2MK_CSRC_QWEN38_TOKENS_CUH_
#define S2MK_CSRC_QWEN38_TOKENS_CUH_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "decoder_layer.cuh"
#include "gemv_core.cuh"
#include "int4_mma_core.cuh"
#include "prefill_dense.cuh"
#include "qwen38_layer.cuh"
#include "qwen38_prefetch.cuh"
#include "thinker_layer.cuh"

namespace s2mk {
namespace qwen38 {
namespace tokens {

constexpr int kHd = kQwen38HeadDim;
using prefill::DenseBegin;

// A projection's partials over kK columns: one n-tile at kK's slices.
constexpr int PartialsFor(int k) {
  return prefill::FixedDensePartialFloats(SlicesFor(k), 1);
}

// A projection's kT inputs of kK bf16 side by side, then its partials.
template <int kT, int kK>
struct StagedProjection {
  uint4 xs[kT * kK / kVecElems];
  float partials[PartialsFor(kK)];
};

template <int kT>
struct AttentionShared {
  union {
    StagedProjection<kT, kDim> qkv;
    // One token's head at a time, then every token's query and its key and
    // value for the item's KV group (rounded to bf16, as the cache holds
    // them), as AttendTokens reads them.
    struct {
      qwen38::AttentionShared head;
      float4 query[kT][kHd / 4];
      float4 key[kT][kHd / 4];
      float4 value[kT][kHd / 4];
    } attend;
    // The gated attention outputs; the merge's per-item weights take the
    // place of the O rows' partials until those are written.
    struct {
      uint4 xs[kT * kQwen38QDim / kVecElems];
      union {
        float partials[PartialsFor(kQwen38QDim)];
        struct {
          decltype(qwen38::AttentionShared::merge_weight) weight;
          decltype(qwen38::AttentionShared::merge_sum) sum;
        } merge;
      };
    } o;
  };
  float red[kWarps];
};

template <int kT>
struct MlpShared {
  // Gate-up's; down reads its inputs from L2 and takes these partials.
  StagedProjection<kT, kDim> x;
  float red[kWarps];
};
static_assert(PartialsFor(kFfn) <= PartialsFor(kDim));

// The tiles stream fewer groups at once than the prefill's: they inline into
// kernels held to 128 registers a thread, where a deeper batch pushes the
// kernel's own state into local memory.
__device__ __forceinline__ auto Int4Of(const Weight& w) {
  return prefill::Int4Of<kMmaPrefetch / 2>(w);
}

__device__ __forceinline__ auto Bf16Of(const Weight& w) {
  return prefill::Bf16Of<kMmaPrefetch / 8>(w);
}

// How the blocks read a layer's projections, fixed at compile time so each
// kernel inlines one format's tiles: the model's layers are kInt4Zp, the MTP
// head's kBf16.
struct Int4Projections {
  __device__ __forceinline__ static auto Tile(const Weight& w) {
    return Int4Of(w);
  }
  // Starts fetching `rows` rows from `begin` of w, of k columns, into L2.
  __device__ __forceinline__ static void Prefetch(const Weight& w, int k,
                                                  int begin, int rows) {
    PrefetchRows(w, k, begin, rows);
  }
};

struct Bf16Projections {
  __device__ __forceinline__ static auto Tile(const Weight& w) {
    return Bf16Of(w);
  }
  __device__ __forceinline__ static void Prefetch(const Weight& w, int k,
                                                  int begin, int rows) {
    PrefetchWeight(w, k, begin, rows);
  }
};

// store(row, t, W[row] · x_t) for this CTA's rows DenseBegin(n) of the n-row
// weight `tile` reads, of kK columns, and each of the kT tokens' rows of x
// that `source` reads (SharedX or L2X); with RowPairs, store(2j, t, y0, y1)
// for gate-up pairs. Called by every thread; the caller makes x visible first.
template <int kT, int kK, typename Emit = prefill::EachRow, typename Tile,
          typename Source, typename Store>
__device__ __forceinline__ void Project(const Tile& tile, int n,
                                        const Source& source, float* partials,
                                        const Store& store) {
  prefill::FixedDenseRows<1, SlicesFor(kK), Emit>(
      tile, kK, n, reinterpret_cast<const __nv_bfloat16*>(source.x), kT,
      partials, store, source);
}

// A store adding y to row `row` of token t's residual.
__device__ __forceinline__ auto AddToResidual(float* residual) {
  return [residual](int row, int t, float y) {
    float* r = residual + t * kDim + row;
    *r = __ldcg(r) + y;
  };
}

// The blocks' workspace in global memory, token-major.
struct Workspace {
  float* residual;  // [kT, kQwen38Dim]: read and added to
  float* qkv;  // [kT, kQwen38QkvRows]
  // [kT, kQwen38QHeads × splits, 2], splits as the layer's own.
  float* partial_ml;
  float* partial_o;  // [kT, kQwen38QHeads × splits, kQwen38HeadDim]
  __nv_bfloat16* act;  // [kT, kQwen38Ffn]
};

// --------------------------------------------------------- full attention

// The fields of Qwen38LayerParams thinker::LoadHead and thinker::Merge read,
// at one token's rows of the workspace.
struct TokenView {
  const float* qkv;
  const float* partial_ml;
  const float* partial_o;
  int splits;
  float eps;
  const __nv_bfloat16* q_norm;
  const __nv_bfloat16* k_norm;
  const __nv_bfloat16* cos_sin;
};

__device__ inline TokenView ViewOf(const Qwen38LayerParams& p,
                                   const Workspace& ws, int t) {
  const int items = kQwen38QHeads * p.splits;
  return {ws.qkv + t * kQkvRows,
          ws.partial_ml + int64_t{t} * items * 2,
          ws.partial_o + int64_t{t} * items * kHd,
          p.splits,
          p.eps,
          p.q_norm,
          p.k_norm,
          p.cos_sin};
}

// The positions a warp of AttendTokens takes at a time. The warp holds every
// token's running max, sum and output in registers beside them, and more
// positions at a time push those into local memory.
constexpr int kTokenPosUnroll = 2;

// Where chunk s of token t's len = pos0 + t + 1 positions begins:
// s × len / splits, as thinker::Attend splits a step of that token alone.
__device__ __forceinline__ int ChunkBegin(int pos0, int t, int s, int splits) {
  return s * (pos0 + t + 1) / splits;
}

// thinker::Attend for this CTA's (head, chunk) item and every token in one
// pass: each key and value of the tokens' chunks is read once and scored
// against every token whose chunk holds it. Position `at` goes to warp
// (at / kTokenPosUnroll) mod kWarps, which takes its positions in order, so a
// token's sums depend on its own position alone, as in a step of that token
// alone. The tokens' positions [pos0, pos0 + kT) come from sh.attend.
template <int kT>
__device__ inline void AttendTokens(const Qwen38LayerParams& p,
                                    const Workspace& ws, int pos0,
                                    AttentionShared<kT>& sh) {
  constexpr int kPerLane = kHd / 32;
  constexpr int kRuns = kPerLane / 4;
  constexpr int kU = kTokenPosUnroll;
  const int items = kQwen38QHeads * p.splits;
  const int item = blockIdx.x;
  const int h = item / p.splits;
  const int s = item % p.splits;
  const int g = h / (kQwen38QHeads / kQwen38KvHeads);
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const PagedKv& kv = p.kv;
  const float scale = rsqrtf(static_cast<float>(kHd));

  int begin[kT];
  int end[kT];
#pragma unroll
  for (int t = 0; t < kT; ++t) {
    begin[t] = ChunkBegin(pos0, t, s, p.splits);
    end[t] = ChunkBegin(pos0, t, s + 1, p.splits);
  }
  // Both bounds grow with t: the tokens' chunks lie in [lo, hi).
  const int lo = begin[0];
  const int hi = end[kT - 1];

  float m[kT];
  float l[kT];
  float o[kT][kPerLane];
#pragma unroll
  for (int t = 0; t < kT; ++t) {
    m[t] = -INFINITY;
    l[t] = 0.f;
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) o[t][j] = 0.f;
  }
  // This warp's first run of kU positions at or after lo.
  const int first = lo / kU;
  for (int run = first + (warp - first % kWarps + kWarps) % kWarps;
       run * kU < hi; run += kWarps) {
    const int base = run * kU;
    uint2 kr[kU][kRuns];
    uint2 vr[kU][kRuns];
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      const int at = base + u;
      if (lo <= at && at < hi && at < pos0) {
        const uint2* key =
            reinterpret_cast<const uint2*>(
                thinker::KvAt(kv, kv.key, g, at)) +
            lane * kRuns;
        const uint2* value =
            reinterpret_cast<const uint2*>(
                thinker::KvAt(kv, kv.value, g, at)) +
            lane * kRuns;
#pragma unroll
        for (int r = 0; r < kRuns; ++r) {
          kr[u][r] = __ldcg(key + r);
          vr[u][r] = __ldcg(value + r);
        }
      }
    }
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      const int at = base + u;
      if (at < lo) continue;
      if (at >= hi) break;
      float4 k[kRuns];
      float4 vv[kRuns];
#pragma unroll
      for (int r = 0; r < kRuns; ++r) {
        if (at >= pos0) {
          k[r] = sh.attend.key[at - pos0][lane * kRuns + r];
          vv[r] = sh.attend.value[at - pos0][lane * kRuns + r];
        } else {
          k[r] = Bf16x4(kr[u][r]);
          vv[r] = Bf16x4(vr[u][r]);
        }
      }
      // Every token's score, the warp sums side by side.
      float dot[kT];
#pragma unroll
      for (int t = 0; t < kT; ++t) {
        const float4* q = sh.attend.query[t] + lane * kRuns;
        dot[t] = Dot(q[0], k[0]);
#pragma unroll
        for (int r = 1; r < kRuns; ++r) dot[t] += Dot(q[r], k[r]);
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (int t = 0; t < kT; ++t) {
          dot[t] += __shfl_xor_sync(0xffffffffu, dot[t], offset);
        }
      }
#pragma unroll
      for (int t = 0; t < kT; ++t) {
        if (at < begin[t] || at >= end[t]) continue;
        const float score = dot[t] * scale;
        const float m_new = fmaxf(m[t], score);
        const float correction = expf(m[t] - m_new);
        const float weight = expf(score - m_new);
        l[t] = l[t] * correction + weight;
#pragma unroll
        for (int r = 0; r < kRuns; ++r) {
          float* ot = o[t] + 4 * r;
          ot[0] = fmaf(ot[0], correction, weight * vv[r].x);
          ot[1] = fmaf(ot[1], correction, weight * vv[r].y);
          ot[2] = fmaf(ot[2], correction, weight * vv[r].z);
          ot[3] = fmaf(ot[3], correction, weight * vv[r].w);
        }
        m[t] = m_new;
      }
    }
  }

  // Each token's warps' partials into its item's, one token at a time.
  qwen38::AttentionShared& w = sh.attend.head;
#pragma unroll
  for (int t = 0; t < kT; ++t) {
    if (lane == 0) {
      w.warp_m[warp] = m[t];
      w.warp_l[warp] = l[t];
    }
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) {
      w.warp_o[warp][lane * kPerLane + j] = o[t][j];
    }
    __syncthreads();
    if (threadIdx.x < kHd) {
      // A warp that saw no position has l = 0 and m = -inf; skip it.
      float m_all = -INFINITY;
      for (int wi = 0; wi < kWarps; ++wi) {
        if (w.warp_l[wi] > 0.f) m_all = fmaxf(m_all, w.warp_m[wi]);
      }
      float l_all = 0.f;
      float o_all = 0.f;
      for (int wi = 0; wi < kWarps; ++wi) {
        if (w.warp_l[wi] == 0.f) continue;
        const float f = expf(w.warp_m[wi] - m_all);
        l_all += w.warp_l[wi] * f;
        o_all += w.warp_o[wi][threadIdx.x] * f;
      }
      const int64_t at = int64_t{t} * items + item;
      ws.partial_o[at * kHd + threadIdx.x] = o_all;
      if (threadIdx.x == 0) {
        ws.partial_ml[at * 2] = m_all;
        ws.partial_ml[at * 2 + 1] = l_all;
      }
    }
    __syncthreads();
  }
}

// The attention block over the kT tokens at positions pos0, pos0 + 1, … of
// ws.residual, adding its output to it and writing each token's key and value
// to the layer's cache. Three phases, two grid barriers.
template <int kT, typename Projections, typename Sync>
__device__ void AttentionBlock(const Qwen38LayerParams& p, const Workspace& ws,
                               int pos0, Sync& sync, AttentionShared<kT>& sh) {
  const int cta = blockIdx.x;
  float* residual = ws.residual;
  qwen38::AttentionShared& head = sh.attend.head;

  // 1. Each token's RMSNorm, then this CTA's q, k, v and gate rows.
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(sh.qkv.xs);
  for (int t = 0; t < kT; ++t) {
    RmsNorm<NormDims>(residual + t * kDim, nullptr, p.input_norm, p.eps,
                      xs + t * kDim, nullptr, sh.red);
  }
  __syncthreads();
  Project<kT, kDim>(Projections::Tile(p.wqkv), kQkvRows,
                    SharedX{sh.qkv.xs, kDim}, sh.qkv.partials,
                    [&](int row, int t, float y) {
                      ws.qkv[t * kQkvRows + row] = y;
                    });
  if (threadIdx.x == 0) {
    const int o_begin = DenseBegin(kDim, cta);
    Projections::Prefetch(p.wo, kQwen38QDim, o_begin,
                          DenseBegin(kDim, cta + 1) - o_begin);
  }
  sync();

  // 2. This CTA's (head, chunk) item: each token's head in turn, then every
  // token's attention in one pass.
  if (cta < kQwen38QHeads * p.splits) {
    const int h = cta / p.splits;
    const int s = cta % p.splits;
    const int g = h / (kQwen38QHeads / kQwen38KvHeads);
    const bool writes = h % (kQwen38QHeads / kQwen38KvHeads) == 0 && s == 0;
    for (int t = 0; t < kT; ++t) {
      const int pos = pos0 + t;
      const thinker::BlockStep step{
          pos, -1, {pos, pos, pos}, residual, residual};
      thinker::LoadHead(ViewOf(p, ws, t), step, h, head);
      if (threadIdx.x < kHd) {
        reinterpret_cast<float*>(sh.attend.query[t])[threadIdx.x] =
            head.q[threadIdx.x];
        reinterpret_cast<float*>(sh.attend.key[t])[threadIdx.x] =
            head.k[threadIdx.x];
        reinterpret_cast<float*>(sh.attend.value[t])[threadIdx.x] =
            head.v[threadIdx.x];
        if (writes) {
          const int64_t slot = thinker::SlotOf(p.kv, pos);
          thinker::SlotAt(p.kv, p.kv.key, g, slot)[threadIdx.x] =
              __float2bfloat16(head.k[threadIdx.x]);
          thinker::SlotAt(p.kv, p.kv.value, g, slot)[threadIdx.x] =
              __float2bfloat16(head.v[threadIdx.x]);
        }
      }
      __syncthreads();
    }
    AttendTokens<kT>(p, ws, pos0, sh);
  }
  sync();

  // 3. Each token's merged, gated attention output, then this CTA's O rows.
  xs = reinterpret_cast<__nv_bfloat16*>(sh.o.xs);
  for (int t = 0; t < kT; ++t) {
    thinker::Merge<Qwen38AttentionDims>(ViewOf(p, ws, t), sh.o.merge.weight,
                                        sh.o.merge.sum, xs + t * kQwen38QDim);
    __syncthreads();
  }
  Project<kT, kQwen38QDim>(Projections::Tile(p.wo), kDim,
                           SharedX{sh.o.xs, kQwen38QDim}, sh.o.partials,
                           AddToResidual(residual));
}

// ---------------------------------------------------------------------- MLP

// The MLP block over the kT tokens of ws.residual, adding its output to it.
// Two phases, one grid barrier.
template <int kT, typename Projections, typename Sync>
__device__ void MlpBlock(const Qwen38MlpParams& p, const Workspace& ws,
                         Sync& sync, MlpShared<kT>& sh) {
  __nv_bfloat16* x = reinterpret_cast<__nv_bfloat16*>(sh.x.xs);
  float* residual = ws.residual;

  // 1. Each token's RMSNorm, then this CTA's (gate, up) pairs.
  for (int t = 0; t < kT; ++t) {
    RmsNorm<NormDims>(residual + t * kDim, nullptr, p.post_norm, p.eps,
                      x + t * kDim, nullptr, sh.red);
  }
  __syncthreads();
  Project<kT, kDim, prefill::RowPairs>(
      Projections::Tile(p.w13), 2 * kFfn, SharedX{sh.x.xs, kDim},
      sh.x.partials, [&](int row, int t, float gate, float up) {
        ws.act[t * kFfn + row / 2] = __float2bfloat16(Silu(gate) * up);
      });
  sync();

  // 2. This CTA's down rows, over every token's activations in L2.
  Project<kT, kFfn>(Projections::Tile(p.w2), kDim,
                    L2X{reinterpret_cast<const uint4*>(ws.act), kFfn},
                    sh.x.partials, AddToResidual(residual));
}

}  // namespace tokens
}  // namespace qwen38
}  // namespace s2mk

#endif  // S2MK_CSRC_QWEN38_TOKENS_CUH_
