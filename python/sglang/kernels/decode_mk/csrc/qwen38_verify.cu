// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// The verify step of Qwen3.8-27B (qwen38_decode.h): kT tokens through the
// decode step's phases in one persistent launch, with the same grid barriers.
// Each phase is the decode step's with every token's input staged side by
// side, so a GEMV loads a weight row once and multiplies it into each token
// (Int4ZpGemvRows' kM inputs, Bf16WarpRowsM). A token's arithmetic, and the
// order of every sum, is the one its own decode step performs:
//   - The conv and the delta rule run the tokens in order on the same thread
//     or warp, each token's states going to its own slot.
//   - Attention runs each token over its own chunks of [0, pos + t]. The
//     earlier tokens' keys and values come from shared memory, rounded to
//     bf16 as the decode step reads them back from the cache.
//   - The down projection takes the tokens two at a time, since four tokens'
//     FFN activations do not fit in shared memory. Its second pass reads the
//     CTA's rows again, mostly from L2.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "barrier.cuh"
#include "decoder_layer.cuh"
#include "gdn_block.cuh"
#include "gemv_core.cuh"
#include "int4_gemv_core.cuh"
#include "qwen38_decode.h"
#include "qwen38_layer.cuh"
#include "qwen38_lm_head.cuh"
#include "qwen38_prefetch.cuh"
#include "thinker_layer.cuh"

namespace s2mk {
namespace {

using qwen38::PrefetchNext;

constexpr int kDim = kQwen38Dim;
constexpr int kHd = kQwen38HeadDim;
constexpr int kQkvRows = kQwen38QkvRows;
constexpr int kFfn = kQwen38Ffn;
using AttentionDims = Qwen38AttentionDims;

// Tokens the down projection takes at once.
constexpr int kDownTokens = 2;
// The most (gate, up) pairs and down rows a CTA owns, at kGdnMinCtas CTAs.
constexpr int kMaxPairsPerCta = (kFfn + kGdnMinCtas - 1) / kGdnMinCtas;
constexpr int kMaxDownRowsPerCta = (kDim + kGdnMinCtas - 1) / kGdnMinCtas;

template <int kT>
struct GdnShared {
  uint4 xs[kT * kGdnValueDim / kVecElems];  // either GEMV's inputs, bf16
  float ys[kT * gdn::kMaxRowsPerCta];
  float red[kWarps];
  // [token][head][0]: the query, normalized and scaled; [1]: the key.
  float4 qk[kT][gdn::kMaxHeadsPerCta][2][gdn::kHeadDim / 4];
};

template <int kT>
struct AttentionShared {
  // One token's head at a time, and the merge's per-item weights.
  qwen38::AttentionShared head;
  // Every token's key and value for the item's KV group, rounded to bf16.
  float key[kT][kHd];
  float value[kT][kHd];
  // The normalized residuals, then the gated attention outputs, bf16.
  uint4 xs[kT * kQwen38QDim / kVecElems];
  float ys[kT * kMaxRowsPerCta];
};

template <int kT>
struct MlpShared {
  union {
    uint4 x[kT * kDim / kVecElems];  // the normalized residuals, bf16
    uint4 act[kDownTokens * kFfn / kVecElems];  // SiLU(gate) × up, bf16
  };
  float ys[kT * 2 * kMaxPairsPerCta];
  float red[kWarps];
};
static_assert(kDownTokens * kMaxDownRowsPerCta <= 2 * kMaxPairsPerCta);

template <int kT>
union Shared {
  GdnShared<kT> gdn;
  AttentionShared<kT> attention;
  MlpShared<kT> mlp;
};
// SM120's opt-in limit of dynamic shared memory per CTA.
static_assert(sizeof(Shared<kQwen38MaxVerifyTokens>) <= 99 * 1024);

// The slot token t writes its states to (Qwen38VerifyParams).
__device__ __forceinline__ int SlotOut(const Qwen38VerifyParams& v, int t) {
  return (*v.slot + 1 + t) % v.num_slots;
}

constexpr int64_t kConvSlot = int64_t{kGdnConvDim} * (kGdnConvWidth - 1);
constexpr int64_t kStateSlot =
    int64_t{kGdnVHeads} * kGdnHeadDim * kGdnHeadDim;

// ------------------------------------------------------- linear attention

using gdn::ColumnBegin;
using gdn::Dot4;
using gdn::kColumnsPerWarp;
using gdn::kConvDim;
using gdn::kConvTaps;
using gdn::kGateRows;
using gdn::kHeadsPerKHead;
using gdn::kInt4Rows;
using gdn::kKeyDim;
using gdn::kQueryScale;
using gdn::kValueDim;
using gdn::kVHeads;
using gdn::Softplus;
constexpr int kGdnHd = kGdnHeadDim;

// xs = RMSNorm(x) × (1 + norm) in bf16, as gdn::InputNorm computes it.
__device__ inline void GdnInputNorm(const GdnParams& p, const float* x,
                                    __nv_bfloat16* xs, float* red) {
  float ss = 0.f;
  for (int i = threadIdx.x; i < kDim; i += kThreads) {
    const float xi = __ldcg(x + i);
    ss = fmaf(xi, xi, ss);
  }
  const float inv = rsqrtf(gdn::BlockSum(ss, red) / kDim + p.eps);
  for (int i = threadIdx.x; i < kDim; i += kThreads) {
    const float scale = 1.f + __bfloat162float(p.norm[i]);
    xs[i] = __float2bfloat16(__ldcg(x + i) * inv * scale);
  }
}

// gdn::Route for every token's value of input-projection row `row` below
// the gates: ys[t × rows + i] is token t's.
template <int kT>
__device__ inline void GdnRoute(const GdnParams& p,
                                const Qwen38VerifyParams& v, int row,
                                const float* ys, int rows, int i) {
  if (row < kConvDim) {
    const __nv_bfloat16* w = p.conv + int64_t{row} * kConvTaps;
    const float* past =
        p.conv_state + *v.slot * kConvSlot + int64_t{row} * (kConvTaps - 1);
    float taps[kConvTaps];
#pragma unroll
    for (int t = 0; t < kConvTaps - 1; ++t) taps[t] = past[t];
#pragma unroll
    for (int t = 0; t < kT; ++t) {
      taps[kConvTaps - 1] = ys[t * rows + i];
      float acc = 0.f;
#pragma unroll
      for (int j = 0; j < kConvTaps; ++j) {
        acc = fmaf(__bfloat162float(w[j]), taps[j], acc);
      }
      float* out = p.conv_state + SlotOut(v, t) * kConvSlot +
                   int64_t{row} * (kConvTaps - 1);
#pragma unroll
      for (int j = 0; j < kConvTaps - 1; ++j) {
        taps[j] = taps[j + 1];
        out[j] = taps[j];
      }
      v.mixed[t * kConvDim + row] = gdn::Silu(acc);
    }
  } else {
#pragma unroll
    for (int t = 0; t < kT; ++t) {
      v.z[t * kValueDim + row - kConvDim] = ys[t * rows + i];
    }
  }
}

// gdn::Route for token t's gate row `row` (b, then a).
__device__ inline void GdnRouteGate(const GdnParams& p,
                                    const Qwen38VerifyParams& v, int t,
                                    int row, float x) {
  if (row < kVHeads) {
    v.beta[t * kVHeads + row] = gdn::Sigmoid(x);
  } else {
    const int h = row - kVHeads;
    const float g = -expf(p.a_log[h]) * Softplus(x + p.dt_bias[h]);
    v.decay[t * kVHeads + h] = expf(g);
  }
}

template <int kT>
__device__ inline void GdnInProjection(const GdnParams& p,
                                       const Qwen38VerifyParams& v,
                                       GdnShared<kT>& sh) {
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(sh.xs);
  for (int t = 0; t < kT; ++t) {
    GdnInputNorm(p, v.step.residual + t * kDim, xs + t * kDim, sh.red);
  }
  const int begin = RowBegin(kInt4Rows, blockIdx.x);
  const int rows = RowBegin(kInt4Rows, blockIdx.x + 1) - begin;
  Int4ZpGemvRows<kT, 32, kDim / (32 * kInt4Group)>(
      static_cast<const uint4*>(p.in_proj.data), p.in_proj.scales,
      p.in_proj.zeros, begin, rows, sh.xs, sh.ys);
  for (int i = threadIdx.x; i < rows; i += kThreads) {
    GdnRoute<kT>(p, v, begin + i, sh.ys, rows, i);
  }
  Bf16Rows<kDim, kT>(
      static_cast<const __nv_bfloat16*>(p.gates.data),
      RowBegin(kGateRows, blockIdx.x), RowBegin(kGateRows, blockIdx.x + 1),
      sh.xs, [&](int row, int t, float x) { GdnRouteGate(p, v, t, row, x); });
}

// gdn::DeltaRule for each token in turn on this CTA's state columns, held in
// registers between tokens.
template <int kT>
__device__ inline void GdnDeltaRule(const GdnParams& p,
                                    const Qwen38VerifyParams& v,
                                    GdnShared<kT>& sh) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int begin = ColumnBegin(blockIdx.x);
  const int columns = ColumnBegin(blockIdx.x + 1) - begin;
  if (columns == 0) return;
  const int first_head = begin / kGdnHd;
  const int heads = (begin + columns - 1) / kGdnHd - first_head + 1;
  const float* state_in = p.state + *v.slot * kStateSlot;

  float4 s[kColumnsPerWarp];
#pragma unroll
  for (int u = 0; u < kColumnsPerWarp; ++u) {
    const int c = warp + u * kWarps;
    if (c < columns) {
      s[u] = __ldcg(reinterpret_cast<const float4*>(
                        state_in + int64_t{begin + c} * kGdnHd) + lane);
    }
  }

  // Job (t, 2j + is_key) normalizes token t's query or key of head j.
  for (int job = warp; job < kT * 2 * heads; job += kWarps) {
    const int t = job / (2 * heads);
    const int j = job % (2 * heads) / 2;
    const int is_key = job % 2;
    const int k_head = (first_head + j) / kHeadsPerKHead;
    const float* src =
        v.mixed + t * kConvDim + is_key * kKeyDim + k_head * kGdnHd;
    const float4 x = __ldcg(reinterpret_cast<const float4*>(src) + lane);
    float inv = rsqrtf(WarpSum(Dot4(x, x)) + 1e-6f);
    if (!is_key) inv *= kQueryScale;
    sh.qk[t][j][is_key][lane] =
        make_float4(x.x * inv, x.y * inv, x.z * inv, x.w * inv);
  }
  __syncthreads();

#pragma unroll
  for (int u = 0; u < kColumnsPerWarp; ++u) {
    const int c = warp + u * kWarps;
    if (c >= columns) break;
    const int column = begin + c;
    const int head = column / kGdnHd;
    float4 col = s[u];
#pragma unroll
    for (int t = 0; t < kT; ++t) {
      const float4 q = sh.qk[t][head - first_head][0][lane];
      const float4 k = sh.qk[t][head - first_head][1][lane];
      const float decay = __ldcg(v.decay + t * kVHeads + head);
      col.x *= decay;
      col.y *= decay;
      col.z *= decay;
      col.w *= decay;
      const float delta =
          (__ldcg(v.mixed + t * kConvDim + 2 * kKeyDim + column) -
           WarpSum(Dot4(col, k))) *
          __ldcg(v.beta + t * kVHeads + head);
      col.x = fmaf(k.x, delta, col.x);
      col.y = fmaf(k.y, delta, col.y);
      col.z = fmaf(k.z, delta, col.z);
      col.w = fmaf(k.w, delta, col.w);
      const float o = WarpSum(Dot4(col, q));
      __stcg(reinterpret_cast<float4*>(p.state + SlotOut(v, t) * kStateSlot +
                                       int64_t{column} * kGdnHd) +
                 lane,
             col);
      if (lane == 0) v.core[t * kValueDim + column] = o;
    }
  }
}

template <int kT>
__device__ inline void GdnOutProjection(const GdnParams& p,
                                        const Qwen38VerifyParams& v,
                                        GdnShared<kT>& sh) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(sh.xs);
  const float4 w = make_float4(__bfloat162float(p.out_norm[4 * lane]),
                               __bfloat162float(p.out_norm[4 * lane + 1]),
                               __bfloat162float(p.out_norm[4 * lane + 2]),
                               __bfloat162float(p.out_norm[4 * lane + 3]));
  for (int j = warp; j < kT * kVHeads; j += kWarps) {
    const int t = j / kVHeads;
    const int at = j % kVHeads * kGdnHd + 4 * lane;
    const float4 o =
        __ldcg(reinterpret_cast<const float4*>(v.core + t * kValueDim + at));
    const float4 z =
        __ldcg(reinterpret_cast<const float4*>(v.z + t * kValueDim + at));
    const float inv = rsqrtf(WarpSum(Dot4(o, o)) / kGdnHd + p.eps);
    __nv_bfloat16* x = xs + t * kValueDim + at;
    x[0] = __float2bfloat16(o.x * inv * w.x * gdn::Silu(z.x));
    x[1] = __float2bfloat16(o.y * inv * w.y * gdn::Silu(z.y));
    x[2] = __float2bfloat16(o.z * inv * w.z * gdn::Silu(z.z));
    x[3] = __float2bfloat16(o.w * inv * w.w * gdn::Silu(z.w));
  }
  float* residual = v.step.residual;
  const int begin = RowBegin(kDim, blockIdx.x);
  const int rows = RowBegin(kDim, blockIdx.x + 1) - begin;
  if (p.out_proj.format == WeightFormat::kBf16) {
    __syncthreads();
    Bf16Rows<kValueDim, kT>(
        static_cast<const __nv_bfloat16*>(p.out_proj.data), begin,
        begin + rows, sh.xs, [&](int row, int t, float y) {
          float* r = residual + t * kDim + row;
          *r = __ldcg(r) + y;
        });
    return;
  }
  Int4ZpGemvRows<kT, 32, kValueDim / (32 * kInt4Group)>(
      static_cast<const uint4*>(p.out_proj.data), p.out_proj.scales,
      p.out_proj.zeros, begin, rows, sh.xs, sh.ys);
  for (int i = threadIdx.x; i < kT * rows; i += kThreads) {
    float* r = residual + i / rows * kDim + begin + i % rows;
    *r = __ldcg(r) + sh.ys[i];
  }
}

template <int kT, typename Sync>
__device__ void GdnBlock(const GdnParams& p, const Qwen38VerifyParams& v,
                         Sync& sync, GdnShared<kT>& sh) {
  GdnInProjection<kT>(p, v, sh);
  sync();
  GdnDeltaRule<kT>(p, v, sh);
  sync();
  GdnOutProjection<kT>(p, v, sh);
}

// --------------------------------------------------------- full attention

// The fields of Qwen38LayerParams thinker::LoadHead and thinker::Merge read,
// at one token's rows of the verify workspace.
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
                                   const Qwen38VerifyParams& v, int t) {
  const int items = kQwen38QHeads * p.splits;
  return {v.qkv + t * kQkvRows,
          v.partial_ml + int64_t{t} * items * 2,
          v.partial_o + int64_t{t} * items * kHd,
          p.splits,
          p.eps,
          p.q_norm,
          p.k_norm,
          p.cos_sin};
}

// thinker::Attend for token t of the step, whose positions [pos0, pos0 + t]
// are the step's tokens, read from sh.key and sh.value.
template <int kT>
__device__ inline void AttendToken(const Qwen38LayerParams& p,
                                   const Qwen38VerifyParams& v, int pos0,
                                   int t, AttentionShared<kT>& sh) {
  constexpr int kPerLane = kHd / 32;
  constexpr int kRuns = kPerLane / 4;
  const int items = kQwen38QHeads * p.splits;
  const int item = blockIdx.x;
  const int h = item / p.splits;
  const int s = item % p.splits;
  const int g = h / (kQwen38QHeads / kQwen38KvHeads);
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const PagedKv& kv = p.kv;
  const qwen38::AttentionShared& hs = sh.head;

  const int len = pos0 + t + 1;
  const int begin = s * len / p.splits;
  const int end = (s + 1) * len / p.splits;
  const float scale = rsqrtf(static_cast<float>(kHd));
  const int dim0 = lane * kPerLane;
  float4 q[kRuns];
#pragma unroll
  for (int r = 0; r < kRuns; ++r) {
    const int d = dim0 + 4 * r;
    q[r] = make_float4(hs.q[d], hs.q[d + 1], hs.q[d + 2], hs.q[d + 3]);
  }

  float m = -INFINITY;
  float l = 0.f;
  float o[kPerLane] = {};
  for (int base = begin + warp * kPosUnroll; base < end;
       base += kWarps * kPosUnroll) {
    uint2 kr[kPosUnroll][kRuns];
    uint2 vr[kPosUnroll][kRuns];
#pragma unroll
    for (int u = 0; u < kPosUnroll; ++u) {
      const int at = base + u;
      if (at < end && at < pos0) {
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
    for (int u = 0; u < kPosUnroll; ++u) {
      const int at = base + u;
      if (at >= end) break;
      float4 k[kRuns];
      float4 vv[kRuns];
#pragma unroll
      for (int r = 0; r < kRuns; ++r) {
        if (at >= pos0) {
          const int d = dim0 + 4 * r;
          const float* key = sh.key[at - pos0];
          const float* value = sh.value[at - pos0];
          k[r] = make_float4(key[d], key[d + 1], key[d + 2], key[d + 3]);
          vv[r] = make_float4(value[d], value[d + 1], value[d + 2],
                              value[d + 3]);
        } else {
          k[r] = Bf16x4(kr[u][r]);
          vv[r] = Bf16x4(vr[u][r]);
        }
      }
      float dot = Dot(q[0], k[0]);
#pragma unroll
      for (int r = 1; r < kRuns; ++r) dot += Dot(q[r], k[r]);
      const float score = WarpSum(dot) * scale;
      const float m_new = fmaxf(m, score);
      const float correction = expf(m - m_new);
      const float weight = expf(score - m_new);
      l = l * correction + weight;
#pragma unroll
      for (int r = 0; r < kRuns; ++r) {
        o[4 * r] = fmaf(o[4 * r], correction, weight * vv[r].x);
        o[4 * r + 1] = fmaf(o[4 * r + 1], correction, weight * vv[r].y);
        o[4 * r + 2] = fmaf(o[4 * r + 2], correction, weight * vv[r].z);
        o[4 * r + 3] = fmaf(o[4 * r + 3], correction, weight * vv[r].w);
      }
      m = m_new;
    }
  }

  qwen38::AttentionShared& w = sh.head;
  if (lane == 0) {
    w.warp_m[warp] = m;
    w.warp_l[warp] = l;
  }
#pragma unroll
  for (int j = 0; j < kPerLane; ++j) w.warp_o[warp][dim0 + j] = o[j];
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
    v.partial_o[at * kHd + threadIdx.x] = o_all;
    if (threadIdx.x == 0) {
      v.partial_ml[at * 2] = m_all;
      v.partial_ml[at * 2 + 1] = l_all;
    }
  }
  __syncthreads();
}

template <int kT, typename Sync>
__device__ void AttentionBlock(const Qwen38LayerParams& p,
                               const Qwen38VerifyParams& v, int pos0,
                               Sync& sync, AttentionShared<kT>& sh) {
  const int cta = blockIdx.x;
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(sh.xs);
  float* residual = v.step.residual;
  const int o_begin = RowBegin(kDim, cta);
  const int o_rows = RowBegin(kDim, cta + 1) - o_begin;

  // 1. Each token's RMSNorm, then this CTA's q, k, v and gate rows.
  for (int t = 0; t < kT; ++t) {
    RmsNorm<qwen38::NormDims>(residual + t * kDim, nullptr, p.input_norm,
                              p.eps, xs + t * kDim, nullptr, sh.head.red);
  }
  {
    const int begin = RowBegin(kQkvRows, cta);
    const int rows = RowBegin(kQkvRows, cta + 1) - begin;
    Int4ZpGemvRows<kT, 32, qwen38::kDimGroupsPerLane>(
        static_cast<const uint4*>(p.wqkv.data), p.wqkv.scales, p.wqkv.zeros,
        begin, rows, sh.xs, sh.ys);
    for (int i = threadIdx.x; i < kT * rows; i += kThreads) {
      v.qkv[i / rows * kQkvRows + begin + i % rows] = sh.ys[i];
    }
  }
  if (threadIdx.x == 0) {
    qwen38::PrefetchRows(p.wo, kQwen38QDim, o_begin, o_rows);
  }
  sync();

  // 2. This CTA's (head, chunk) item, for each token in turn.
  if (cta < kQwen38QHeads * p.splits) {
    const int h = cta / p.splits;
    const int s = cta % p.splits;
    const int g = h / (kQwen38QHeads / kQwen38KvHeads);
    const bool writes = h % (kQwen38QHeads / kQwen38KvHeads) == 0 && s == 0;
    for (int t = 0; t < kT; ++t) {
      const int pos = pos0 + t;
      const thinker::BlockStep step{
          pos, -1, {pos, pos, pos}, residual, residual};
      thinker::LoadHead(ViewOf(p, v, t), step, h, sh.head);
      if (threadIdx.x < kHd) {
        sh.key[t][threadIdx.x] = sh.head.k[threadIdx.x];
        sh.value[t][threadIdx.x] = sh.head.v[threadIdx.x];
        if (writes) {
          const int64_t slot = thinker::SlotOf(p.kv, pos);
          thinker::SlotAt(p.kv, p.kv.key, g, slot)[threadIdx.x] =
              __float2bfloat16(sh.head.k[threadIdx.x]);
          thinker::SlotAt(p.kv, p.kv.value, g, slot)[threadIdx.x] =
              __float2bfloat16(sh.head.v[threadIdx.x]);
        }
      }
      __syncthreads();
      AttendToken<kT>(p, v, pos0, t, sh);
    }
  }
  sync();

  // 3. Each token's merged, gated attention output, then this CTA's O rows.
  for (int t = 0; t < kT; ++t) {
    thinker::Merge(ViewOf(p, v, t), sh.head, xs + t * kQwen38QDim);
    __syncthreads();
  }
  Int4ZpGemvRows<kT, 32, qwen38::kQGroupsPerLane>(
      static_cast<const uint4*>(p.wo.data), p.wo.scales, p.wo.zeros, o_begin,
      o_rows, sh.xs, sh.ys);
  for (int i = threadIdx.x; i < kT * o_rows; i += kThreads) {
    float* r = residual + i / o_rows * kDim + o_begin + i % o_rows;
    *r = __ldcg(r) + sh.ys[i];
  }
}

// ---------------------------------------------------------------------- MLP

// The down rows of tokens [t0, t0 + kN), their activations staged first.
template <int kT, int kN>
__device__ inline void DownRows(const Qwen38MlpParams& p,
                                const Qwen38VerifyParams& v, int t0,
                                MlpShared<kT>& sh) {
  const uint4* act = reinterpret_cast<const uint4*>(v.act + t0 * kFfn);
  for (int i = threadIdx.x; i < kN * kFfn / kVecElems; i += kThreads) {
    sh.act[i] = __ldcg(act + i);
  }
  const int begin = RowBegin(kDim, blockIdx.x);
  const int rows = RowBegin(kDim, blockIdx.x + 1) - begin;
  Int4ZpGemvRows<kN, 32, qwen38::kFfnGroupsPerLane>(
      static_cast<const uint4*>(p.w2.data), p.w2.scales, p.w2.zeros, begin,
      rows, sh.act, sh.ys);
  float* residual = v.step.residual;
  for (int i = threadIdx.x; i < kN * rows; i += kThreads) {
    float* r = residual + (t0 + i / rows) * kDim + begin + i % rows;
    *r = __ldcg(r) + sh.ys[i];
  }
}

template <int kT, typename Sync>
__device__ void MlpBlock(const Qwen38MlpParams& p,
                         const Qwen38VerifyParams& v, Sync& sync,
                         MlpShared<kT>& sh) {
  const int cta = blockIdx.x;
  __nv_bfloat16* x = reinterpret_cast<__nv_bfloat16*>(sh.x);
  float* residual = v.step.residual;

  // 1. Each token's RMSNorm, then this CTA's (gate, up) pairs.
  for (int t = 0; t < kT; ++t) {
    RmsNorm<qwen38::NormDims>(residual + t * kDim, nullptr, p.post_norm, p.eps,
                              x + t * kDim, nullptr, sh.red);
  }
  {
    const int begin = RowBegin(kFfn, cta);
    const int pairs = RowBegin(kFfn, cta + 1) - begin;
    Int4ZpGemvRows<kT, 32, qwen38::kDimGroupsPerLane>(
        static_cast<const uint4*>(p.w13.data), p.w13.scales, p.w13.zeros,
        2 * begin, 2 * pairs, sh.x, sh.ys);
    for (int i = threadIdx.x; i < kT * pairs; i += kThreads) {
      const int t = i / pairs;
      const int j = i % pairs;
      const float* y = sh.ys + t * 2 * pairs + 2 * j;
      v.act[t * kFfn + begin + j] = __float2bfloat16(Silu(y[0]) * y[1]);
    }
  }
  sync();

  // 2. The down rows, kDownTokens tokens at a time.
#pragma unroll
  for (int t0 = 0; t0 + kDownTokens <= kT; t0 += kDownTokens) {
    DownRows<kT, kDownTokens>(p, v, t0, sh);
  }
  if constexpr (kT % kDownTokens != 0) {
    DownRows<kT, kT % kDownTokens>(p, v, kT - kT % kDownTokens, sh);
  }
}

// ------------------------------------------------------------------ the step

template <int kT>
__global__ void __launch_bounds__(kThreads, 1)
    Qwen38VerifyKernel(const __grid_constant__ Qwen38VerifyParams v) {
  extern __shared__ uint4 smem[];
  Shared<kT>& sh = *reinterpret_cast<Shared<kT>*>(smem);
  const Qwen38DecodeParams& p = v.step;
  GridBarrier barrier(p.sync, p.error, p.timeout_ns, 0);
  auto sync = [&] {
    PrefetchNext(p, barrier.index());
    barrier.Sync();
  };
  const int pos0 = *p.pos;

  // Every token's embedding: this CTA's rows of each.
  const int end = RowBegin(kDim, blockIdx.x + 1);
  for (int t = 0; t < kT; ++t) {
    const __nv_bfloat16* row = p.embed + int64_t{p.token[t]} * kDim;
    for (int i = RowBegin(kDim, blockIdx.x) + threadIdx.x; i < end;
         i += kThreads) {
      p.residual[t * kDim + i] = __bfloat162float(row[i]);
    }
  }
  sync();
  for (int layer = 0; layer < p.num_layers; ++layer) {
    const Qwen38MlpParams* mlp;
    if (Qwen38IsFull(layer)) {
      const Qwen38LayerParams& full = p.full[Qwen38FullIndex(layer)];
      AttentionBlock<kT>(full, v, pos0, sync, sh.attention);
      mlp = &full.mlp;
    } else {
      GdnBlock<kT>(p.linear[Qwen38LinearIndex(layer)], v, sync, sh.gdn);
      mlp = &p.linear_mlp[Qwen38LinearIndex(layer)];
    }
    sync();
    MlpBlock<kT>(*mlp, v, sync, sh.mlp);
    sync();
  }

  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(sh.gdn.xs);
  for (int t = 0; t < kT; ++t) {
    RmsNorm<qwen38::NormDims>(
        p.residual + t * kDim, nullptr, p.final_norm, p.eps, xs + t * kDim,
        blockIdx.x == 0 ? v.final_hidden + t * kDim : nullptr, sh.gdn.red);
  }
  __syncthreads();
  qwen38::LmHeadRows<kT>(p.lm_head, p.vocab, sh.gdn.xs,
                         [&](int row, int t, float logit) {
                           p.logits[int64_t{t} * p.vocab + row] = logit;
                         });
}

template <int kT>
cudaError_t Launch(const Qwen38VerifyParams& params, int num_ctas,
                   cudaStream_t stream) {
  constexpr int kSmem = sizeof(Shared<kT>);
  cudaError_t err = cudaFuncSetAttribute(
      Qwen38VerifyKernel<kT>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSmem);
  if (err != cudaSuccess) return err;
  Qwen38VerifyKernel<kT><<<num_ctas, kThreads, kSmem, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace

cudaError_t LaunchQwen38Verify(const Qwen38VerifyParams& params, int num_ctas,
                               cudaStream_t stream) {
  cudaError_t err = cudaMemsetAsync(params.step.sync, 0,
                                    kSyncWords * sizeof(unsigned), stream);
  if (err != cudaSuccess) return err;
  static_assert(kQwen38MaxVerifyTokens == 4);
  switch (params.num_tokens) {
    case 1:
      return Launch<1>(params, num_ctas, stream);
    case 2:
      return Launch<2>(params, num_ctas, stream);
    case 3:
      return Launch<3>(params, num_ctas, stream);
    case 4:
      return Launch<4>(params, num_ctas, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace s2mk
