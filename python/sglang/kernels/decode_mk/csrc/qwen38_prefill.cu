// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// One prefill chunk of Qwen3.8-27B (qwen38_prefill.h) as a persistent launch,
// a grid barrier after each phase. A linear-attention layer:
//
//   L1. Per token (a CTA each): RMSNorm into h.
//   L2. The int4 in_proj rows (q, k, v, z) and the bf16 gate rows (b, a) of
//       every token; the gates become β and the log decay g.
//   L3. A warp per (token, head of q, k or v): the causal conv over the conv
//       state and the chunk, SiLU, and for q and k the L2 norm.
//   L4. The chunked gated delta rule, a CTA per (value head, run of its state
//       columns); the conv state takes the chunk's last inputs.
//   L5. A warp per (token, value head): the gated RMSNorm into attn.
//   L6. out_proj rows, added to the residual.
//
// A full-attention layer:
//
//   F1. Per token: RMSNorm into h.
//   F2. q, k, v and gate rows of every token.
//   F3. A warp per (token, KV head): the key's QK-norm and M-RoPE; key and
//       value, bf16, to the paged cache.
//   F4. A CTA per (KV group, 8 tokens, span of 512 cache positions): its
//       queries' QK-norm and M-RoPE, then causal online softmax over the
//       group's keys in the span, staged in shared memory a tile at a time,
//       into a partial.
//   F5. A warp per (token, query head): the partials merged, then gated.
//   F6. O rows, added to the residual.
//
// Then the MLP: M1. RMSNorm. M2. gate and up pairs into SiLU(gate) × up.
// M3. down rows, added to the residual.
//
// The dense phases are FixedDense (prefill_dense.cuh): every 16-row block
// splits k into a fixed number of slices for its shape, so every output is a
// sum in one order, whichever CTA takes it.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "barrier.cuh"
#include "decoder_layer.cuh"
#include "gdn_block.cuh"
#include "gemv_core.cuh"
#include "prefill_dense.cuh"
#include "qwen38_decode.h"
#include "qwen38_layer.cuh"
#include "qwen38_lm_head.cuh"
#include "qwen38_prefill.h"
#include "thinker_layer.cuh"

namespace s2mk {
namespace {

constexpr int kDim = kQwen38Dim;
constexpr int kMaxTokens = kQwen38PrefillMaxTokens;
static_assert(kMaxTokens == prefill::kMaxTokens);
constexpr int kFfn = kQwen38Ffn;
constexpr int kHead = kQwen38HeadDim;
constexpr int kQDim = kQwen38QDim;
constexpr int kKvDim = kQwen38KvDim;
constexpr int kQkvRows = kQwen38QkvRows;
constexpr int kGqa = kQwen38QHeads / kQwen38KvHeads;
constexpr int kRotHalf = kQwen38Rotary / 2;
// The output gate's rows in the fused q, k, v and gate rows.
constexpr int kGateRow = kQDim + 2 * kKvDim;
static_assert(kQkvRows <= kGdnInt4Rows, "proj holds either layer's rows");
static_assert(kQDim == kGdnValueDim, "attn holds either layer's output");

using gdn::kHeadDim;  // a linear layer's head, 128
using gdn::kHeadsPerKHead;
constexpr int kKHeads = kGdnKHeads;
constexpr int kVHeads = kGdnVHeads;
constexpr int kValueDim = kGdnValueDim;
constexpr int kConvDim = kGdnConvDim;
constexpr int kConvTaps = kGdnConvWidth;

__device__ __forceinline__ int GlobalWarp() {
  return blockIdx.x * kWarps + threadIdx.x / 32;
}

__device__ __forceinline__ int GridWarps() { return gridDim.x * kWarps; }

// ------------------------------------------------------------ the dense phases

// Slices of k a block splits into, by width. The projections from the
// residual leave a CTA many blocks, and four slices a block keep its warps
// busy; o_proj's, out_proj's and down's 5120 rows leave a CTA a block or two,
// which take eight, so a CTA's two blocks share each slice's staged x.
constexpr int SlicesFor(int k) { return k == kDim ? 4 : 8; }

__device__ __forceinline__ prefill::Int4ZpTile Int4Of(const Weight& w) {
  return {static_cast<const int32_t*>(w.data), w.scales, w.zeros};
}

// The kInt4Zp weight w of n rows of kK over x, every token's rows to store.
template <int kK, typename Emit = prefill::EachRow, typename Store>
__device__ void Project(const Qwen38PrefillParams& p, const Weight& w, int n,
                        const __nv_bfloat16* x, float* partials,
                        const Store& store) {
  static_assert(kK / kInt4Group % SlicesFor(kK) == 0,
                "every slice holds as many groups (StagedX)");
  prefill::FixedDense<SlicesFor(kK), Emit>(Int4Of(w), kK, n, x, p.tokens,
                                          partials, store);
}

// The kBf16 weight w of n rows of kK over x, every token's rows to store.
template <int kK, typename Store>
__device__ void ProjectBf16(const Qwen38PrefillParams& p, const Weight& w,
                            int n, const __nv_bfloat16* x, float* partials,
                            const Store& store) {
  prefill::FixedDense<SlicesFor(kK)>(
      prefill::Bf16Tile{static_cast<const __nv_bfloat16*>(w.data)}, kK, n, x,
      p.tokens, partials, store);
}

// A store adding row `row` of token t onto the residual.
__device__ __forceinline__ auto AddToResidual(const Qwen38PrefillParams& p) {
  return [&p](int row, int t, float y) {
    float* r = p.residual + int64_t{t} * kDim + row;
    *r = __ldcg(r) + y;
  };
}

// -------------------------------------------------------------- L1, F1, M1

// Per token, a CTA each: the residual to dump (when set), then RMSNorm ×
// (1 + norm) into h, as the decode's RmsNorm<qwen38::NormDims>.
__device__ void TokenNorm(const Qwen38PrefillParams& p,
                          const __nv_bfloat16* norm, float eps, float* dump,
                          float* red) {
  constexpr int kPerThread = kDim / kThreads;
  static_assert(kPerThread * kThreads == kDim);
  for (int t = blockIdx.x; t < p.tokens; t += gridDim.x) {
    const float* row = p.residual + int64_t{t} * kDim;
    float v[kPerThread];
    float squares = 0.f;
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
      const int i = threadIdx.x + j * kThreads;
      v[j] = __ldcg(row + i);
      squares += v[j] * v[j];
      if (dump != nullptr) dump[int64_t{t} * kDim + i] = v[j];
    }
    const float inv = rsqrtf(BlockSum(squares, red) / kDim + eps);
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
      const int i = threadIdx.x + j * kThreads;
      p.h[int64_t{t} * kDim + i] = __float2bfloat16(
          v[j] * inv * NormWeight<qwen38::NormDims>(norm[i]));
    }
  }
}

// -------------------------------------------------------------- L2, L3

__device__ void InProjection(const Qwen38PrefillParams& p, const GdnParams& w,
                             float* partials) {
  Project<kDim>(p, w.in_proj, kGdnInt4Rows, p.h, partials,
                [&](int row, int t, float y) {
                  p.proj[int64_t{t} * kGdnInt4Rows + row] = y;
                });
  ProjectBf16<kDim>(p, w.gates, kGdnGateRows, p.h, partials,
                    [&](int row, int t, float y) {
                      if (row < kVHeads) {
                        p.beta[t * kVHeads + row] = gdn::Sigmoid(y);
                      } else {
                        const int h = row - kVHeads;
                        p.gate[t * kVHeads + h] =
                            -expf(w.a_log[h]) * gdn::Softplus(y + w.dt_bias[h]);
                      }
                    });
}

// Heads of 128 conv channels: q's, then k's, then v's.
constexpr int kConvHeads = kConvDim / kHeadDim;

// A warp per (token, conv head): each lane's four channels through the causal
// conv, whose taps before the chunk come from the conv state, and SiLU; q and
// k heads are then L2-normalized, q also scaled, as the decode's DeltaRule.
__device__ void Conv(const Qwen38PrefillParams& p, const GdnParams& w) {
  const int lane = threadIdx.x % 32;
  for (int item = GlobalWarp(); item < p.tokens * kConvHeads;
       item += GridWarps()) {
    const int t = item / kConvHeads;
    const int head = item % kConvHeads;
    float out[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int c = head * kHeadDim + 4 * lane + r;
      const __nv_bfloat16* tap_w = w.conv + int64_t{c} * kConvTaps;
      float acc = 0.f;
#pragma unroll
      for (int d = 0; d < kConvTaps; ++d) {
        // Input s of the sequence, t − 3 .. t: the chunk's from proj, those
        // before it from the state, oldest first.
        const int s = t - (kConvTaps - 1) + d;
        const float x =
            s >= 0 ? __ldcg(p.proj + int64_t{s} * kGdnInt4Rows + c)
                   : __ldcg(w.conv_state + int64_t{c} * (kConvTaps - 1) +
                            (kConvTaps - 1) + s);
        acc = fmaf(__bfloat162float(tap_w[d]), x, acc);
      }
      out[r] = gdn::Silu(acc);
    }
    float4 y = make_float4(out[0], out[1], out[2], out[3]);
    if (head < 2 * kKHeads) {
      const bool is_key = head >= kKHeads;
      float inv = rsqrtf(WarpSum(gdn::Dot4(y, y)) + 1e-6f);
      if (!is_key) inv *= gdn::kQueryScale;
      y = make_float4(y.x * inv, y.y * inv, y.z * inv, y.w * inv);
      float* dst = (is_key ? p.key : p.query) +
                   (int64_t{t} * kKHeads + head % kKHeads) * kHeadDim;
      reinterpret_cast<float4*>(dst)[lane] = y;
    } else {
      reinterpret_cast<float4*>(p.value + int64_t{t} * kValueDim +
                                (head - 2 * kKHeads) * kHeadDim)[lane] = y;
    }
  }
}

// ---------------------------------------------------------------------- L4
//
// The delta rule over the chunk, for value head h with dk × dv state S₀ at
// the chunk's start. Token t has key kₜ, query qₜ, value vₜ, βₜ and log decay
// gₜ; Gₜ = g₀ + .. + gₜ and γₜ = exp(Gₜ). Token by token, as the decode runs
// it, S ← exp(gₜ) × S then S ← S + kₜ ⊗ δₜ with δₜ = βₜ × (vₜ − Sᵀ kₜ), and
// oₜ = Sᵀ qₜ. Unrolled over the chunk:
//
//   δₜ = βₜ × (vₜ − γₜ × S₀ᵀ kₜ) − Σ_{j<t} Aₜⱼ δⱼ,
//        Aₜⱼ = βₜ × exp(Gₜ − Gⱼ) × (kₜ · kⱼ),
//   oₜ = γₜ × S₀ᵀ qₜ + Σ_{j≤t} Mₜⱼ δⱼ,  Mₜⱼ = exp(Gₜ − Gⱼ) × (qₜ · kⱼ),
//   S  = γ_{T−1} × S₀ + Σⱼ exp(G_{T−1} − Gⱼ) × kⱼ ⊗ δⱼ.
//
// The first is (I + A) δ = rhs, unit lower triangular, solved by forward
// substitution. A state column (dv index) meets only its own v, δ, o and S
// column, so a CTA takes a run of one head's columns, each warp whole columns:
// A and M are the head's, computed alike by every CTA that takes it, and a
// column's arithmetic does not depend on which CTA or warp takes it.

// Row stride of the staged keys and queries: off by one bank a row, so a
// warp reading one dim of 32 rows hits 32 banks.
constexpr int kStageStride = kHeadDim + 1;

struct DeltaShared {
  float key[kMaxTokens * kStageStride];
  float query[kMaxTokens * kStageStride];
  // A, then M once the substitution is done, [t][j].
  float am[kMaxTokens * kMaxTokens];
  float g_sum[kMaxTokens];  // Gₜ
  float beta[kMaxTokens];
  float gamma[kMaxTokens];  // γₜ = exp(Gₜ)
  float to_end[kMaxTokens];  // exp(G_{T−1} − Gₜ)
};

// Columns a warp takes at most: a CTA's run is at most half a head.
constexpr int kColumnsPerWarp = kHeadDim / 2 / kWarps;

// Lane l's share of a dk vector: dims l + 32 r, r < 4.
constexpr int kDkPerLane = kHeadDim / 32;

// Fills sh.am[i][j] for i < T, j < kMaxTokens with value(i, j) when j < i
// (strict) or j ≤ i, else 0. Thread (i, j) reads row i of `a` broadcast and
// row j of `b` across the warp's banks.
template <bool kStrict, typename Value>
__device__ void LowerTriangle(DeltaShared& sh, int tokens, const float* a,
                              const float* b, const Value& value) {
  for (int idx = threadIdx.x; idx < tokens * kMaxTokens; idx += kThreads) {
    const int i = idx / kMaxTokens;
    const int j = idx % kMaxTokens;
    float out = 0.f;
    if (kStrict ? j < i : j <= i) {
      float dot = 0.f;
      for (int d = 0; d < kHeadDim; ++d) {
        dot = fmaf(a[i * kStageStride + d], b[j * kStageStride + d], dot);
      }
      out = value(i, j, dot);
    }
    sh.am[idx] = out;
  }
}

// Warp sum of row t of sh.am against δ (δ₀ for j = lane, δ₁ for j = lane + 32).
__device__ __forceinline__ float RowTimesDelta(const DeltaShared& sh, int t,
                                               int lane, float delta0,
                                               float delta1) {
  return fmaf(sh.am[t * kMaxTokens + lane], delta0,
              sh.am[t * kMaxTokens + lane + 32] * delta1);
}

// Σ over the lane's dims of row t of `staged` times the lane's s.
__device__ __forceinline__ float LaneDot(const float* staged, int t, int lane,
                                         const float (&s)[kDkPerLane]) {
  float acc = 0.f;
#pragma unroll
  for (int r = 0; r < kDkPerLane; ++r) {
    acc = fmaf(staged[t * kStageStride + lane + 32 * r], s[r], acc);
  }
  return acc;
}

__device__ void DeltaRuleUnit(const Qwen38PrefillParams& p, const GdnParams& w,
                              int h, int c0, int c1, DeltaShared& sh) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int tokens = p.tokens;
  const int k_head = h / kHeadsPerKHead;

  for (int idx = threadIdx.x; idx < tokens * kHeadDim; idx += kThreads) {
    const int t = idx / kHeadDim;
    const int d = idx % kHeadDim;
    const int64_t at = (int64_t{t} * kKHeads + k_head) * kHeadDim + d;
    sh.key[t * kStageStride + d] = __ldcg(p.key + at);
    sh.query[t * kStageStride + d] = __ldcg(p.query + at);
  }
  if (threadIdx.x == 0) {
    float g = 0.f;
    for (int t = 0; t < tokens; ++t) {
      g += __ldcg(p.gate + t * kVHeads + h);
      sh.g_sum[t] = g;
      sh.beta[t] = __ldcg(p.beta + t * kVHeads + h);
    }
  }
  __syncthreads();
  if (threadIdx.x < tokens) {
    const int t = threadIdx.x;
    sh.gamma[t] = expf(sh.g_sum[t]);
    sh.to_end[t] = expf(sh.g_sum[tokens - 1] - sh.g_sum[t]);
  }
  LowerTriangle<true>(sh, tokens, sh.key, sh.key, [&](int i, int j, float dot) {
    return sh.beta[i] * expf(sh.g_sum[i] - sh.g_sum[j]) * dot;
  });
  __syncthreads();

  // The warp's columns, all at once so their shuffle chains interleave; a
  // column past c1 runs on zeros and stores nothing.
  float s0[kColumnsPerWarp][kDkPerLane];
  float delta0[kColumnsPerWarp];  // δ of token lane
  float delta1[kColumnsPerWarp];  // δ of token lane + 32
  float v0[kColumnsPerWarp];
  float v1[kColumnsPerWarp];
#pragma unroll
  for (int u = 0; u < kColumnsPerWarp; ++u) {
    const int column = h * kHeadDim + c0 + warp + u * kWarps;
    const bool live = c0 + warp + u * kWarps < c1;
    const float* s = w.state + int64_t{column} * kHeadDim;
#pragma unroll
    for (int r = 0; r < kDkPerLane; ++r) {
      s0[u][r] = live ? s[lane + 32 * r] : 0.f;
    }
    v0[u] = live && lane < tokens
                ? __ldcg(p.value + int64_t{lane} * kValueDim + column)
                : 0.f;
    v1[u] = live && lane + 32 < tokens
                ? __ldcg(p.value + int64_t{lane + 32} * kValueDim + column)
                : 0.f;
    delta0[u] = 0.f;
    delta1[u] = 0.f;
  }

  // rhsₜ = βₜ × (vₜ − γₜ × S₀ᵀ kₜ), held by lane t mod 32 in δ until the
  // substitution replaces it.
  for (int t = 0; t < tokens; ++t) {
#pragma unroll
    for (int u = 0; u < kColumnsPerWarp; ++u) {
      const float dot = WarpSum(LaneDot(sh.key, t, lane, s0[u]));
      const float v = t < 32 ? v0[u] : v1[u];
      const float rhs = sh.beta[t] * (v - sh.gamma[t] * dot);
      if (lane == t % 32) (t < 32 ? delta0[u] : delta1[u]) = rhs;
    }
  }
  // δₜ = rhsₜ − Σ_{j<t} Aₜⱼ δⱼ, in token order. Lane j's δ for j ≥ t still
  // holds rhs, which A's zeros above the diagonal leave out.
  for (int t = 0; t < tokens; ++t) {
#pragma unroll
    for (int u = 0; u < kColumnsPerWarp; ++u) {
      const float sum =
          WarpSum(RowTimesDelta(sh, t, lane, delta0[u], delta1[u]));
      if (lane == t % 32) (t < 32 ? delta0[u] : delta1[u]) -= sum;
    }
  }
  __syncthreads();
  LowerTriangle<false>(sh, tokens, sh.query, sh.key,
                       [&](int i, int j, float dot) {
                         return expf(sh.g_sum[i] - sh.g_sum[j]) * dot;
                       });
  __syncthreads();

  // oₜ = γₜ × S₀ᵀ qₜ + Σ_{j≤t} Mₜⱼ δⱼ.
  for (int t = 0; t < tokens; ++t) {
#pragma unroll
    for (int u = 0; u < kColumnsPerWarp; ++u) {
      const float part =
          fmaf(sh.gamma[t], LaneDot(sh.query, t, lane, s0[u]),
               RowTimesDelta(sh, t, lane, delta0[u], delta1[u]));
      const float o = WarpSum(part);
      const int column = c0 + warp + u * kWarps;
      if (lane == 0 && column < c1) {
        p.core[int64_t{t} * kValueDim + h * kHeadDim + column] = o;
      }
    }
  }
  // S = γ_{T−1} × S₀ + Σⱼ exp(G_{T−1} − Gⱼ) × kⱼ ⊗ δⱼ.
#pragma unroll
  for (int u = 0; u < kColumnsPerWarp; ++u) {
    const int column = c0 + warp + u * kWarps;
    float s[kDkPerLane];
#pragma unroll
    for (int r = 0; r < kDkPerLane; ++r) {
      s[r] = sh.gamma[tokens - 1] * s0[u][r];
    }
    for (int j = 0; j < tokens; ++j) {
      const float delta =
          __shfl_sync(0xffffffffu, j < 32 ? delta0[u] : delta1[u], j % 32);
      const float weight = sh.to_end[j] * delta;
#pragma unroll
      for (int r = 0; r < kDkPerLane; ++r) {
        s[r] = fmaf(weight, sh.key[j * kStageStride + lane + 32 * r], s[r]);
      }
    }
    if (column < c1) {
      float* out = w.state + int64_t{h * kHeadDim + column} * kHeadDim;
#pragma unroll
      for (int r = 0; r < kDkPerLane; ++r) out[lane + 32 * r] = s[r];
    }
  }
  __syncthreads();
}

// The conv state takes the chunk's last kConvTaps − 1 inputs, after Conv has
// read the state: the last of (state, the chunk's inputs), oldest first.
__device__ void ShiftConvState(const Qwen38PrefillParams& p,
                               const GdnParams& w) {
  constexpr int kPast = kConvTaps - 1;
  for (int c = blockIdx.x * kThreads + threadIdx.x; c < kConvDim;
       c += gridDim.x * kThreads) {
    float* past = w.conv_state + int64_t{c} * kPast;
    float old[kPast];
#pragma unroll
    for (int j = 0; j < kPast; ++j) old[j] = past[j];
#pragma unroll
    for (int j = 0; j < kPast; ++j) {
      const int s = p.tokens - kPast + j;
      past[j] = s >= 0 ? __ldcg(p.proj + int64_t{s} * kGdnInt4Rows + c)
                       : old[kPast + s];
    }
  }
}

// Every CTA below kVHeads × splits takes a (head, run of columns) unit;
// splits = gridDim / kVHeads ≥ 2, so a run is at most half a head.
__device__ void DeltaRule(const Qwen38PrefillParams& p, const GdnParams& w,
                          float* smem) {
  ShiftConvState(p, w);
  const int splits = gridDim.x / kVHeads;
  auto& sh = *reinterpret_cast<DeltaShared*>(smem);
  for (int unit = blockIdx.x; unit < kVHeads * splits; unit += gridDim.x) {
    const int h = unit / splits;
    const int split = unit % splits;
    DeltaRuleUnit(p, w, h, split * kHeadDim / splits,
                  (split + 1) * kHeadDim / splits, sh);
  }
}

// ---------------------------------------------------------------- L5, L6

// A warp per (token, value head): RMSNorm(core) × out_norm × SiLU(z) in bf16,
// as the decode's OutProjection.
__device__ void GatedNorm(const Qwen38PrefillParams& p, const GdnParams& w) {
  const int lane = threadIdx.x % 32;
  const float4 nw = make_float4(__bfloat162float(w.out_norm[4 * lane]),
                                __bfloat162float(w.out_norm[4 * lane + 1]),
                                __bfloat162float(w.out_norm[4 * lane + 2]),
                                __bfloat162float(w.out_norm[4 * lane + 3]));
  for (int item = GlobalWarp(); item < p.tokens * kVHeads;
       item += GridWarps()) {
    const int t = item / kVHeads;
    const int h = item % kVHeads;
    const int at = h * kHeadDim + 4 * lane;
    const float4 o = __ldcg(
        reinterpret_cast<const float4*>(p.core + int64_t{t} * kValueDim + at));
    const float4 z = __ldcg(reinterpret_cast<const float4*>(
        p.proj + int64_t{t} * kGdnInt4Rows + kConvDim + at));
    const float inv = rsqrtf(WarpSum(gdn::Dot4(o, o)) / kHeadDim + w.eps);
    __nv_bfloat16* out = p.attn + int64_t{t} * kValueDim + at;
    out[0] = __float2bfloat16(o.x * inv * nw.x * gdn::Silu(z.x));
    out[1] = __float2bfloat16(o.y * inv * nw.y * gdn::Silu(z.y));
    out[2] = __float2bfloat16(o.z * inv * nw.z * gdn::Silu(z.z));
    out[3] = __float2bfloat16(o.w * inv * nw.w * gdn::Silu(z.w));
  }
}

__device__ void OutProjection(const Qwen38PrefillParams& p, const GdnParams& w,
                              float* partials) {
  if (w.out_proj.format == WeightFormat::kBf16) {
    ProjectBf16<kValueDim>(p, w.out_proj, kDim, p.attn, partials,
                           AddToResidual(p));
  } else {
    Project<kValueDim>(p, w.out_proj, kDim, p.attn, partials,
                       AddToResidual(p));
  }
}

// ---------------------------------------------------------------- F3, F4
//
// Lane l holds dims 8 l .. 8 l + 7 of a 256-dim head. RoPE turns the first
// kQwen38Rotary dims, pairing dim i < 32 (lanes 0 .. 3) with dim i + 32
// (lanes 4 .. 7, the same register).

constexpr int kDimsPerLane = kHead / 32;

__device__ __forceinline__ void LoadLane(const float* src, int lane,
                                         float (&x)[kDimsPerLane]) {
  const float4 a = __ldcg(reinterpret_cast<const float4*>(src) + 2 * lane);
  const float4 b = __ldcg(reinterpret_cast<const float4*>(src) + 2 * lane + 1);
  x[0] = a.x, x[1] = a.y, x[2] = a.z, x[3] = a.w;
  x[4] = b.x, x[5] = b.y, x[6] = b.z, x[7] = b.w;
}

// QK-norm (1 + norm) and partial interleaved M-RoPE of token t's head, as the
// decode's LoadHead. Every lane of the warp calls it.
__device__ void NormRope(const Qwen38PrefillParams& p,
                         const Qwen38LayerParams& f, const __nv_bfloat16* norm,
                         int t, int lane, float (&x)[kDimsPerLane]) {
  float squares = 0.f;
#pragma unroll
  for (int i = 0; i < kDimsPerLane; ++i) squares += x[i] * x[i];
  const float inv = rsqrtf(WarpSum(squares) / kHead + f.eps);
#pragma unroll
  for (int i = 0; i < kDimsPerLane; ++i) {
    x[i] = x[i] * inv *
           NormWeight<Qwen38AttentionDims>(norm[kDimsPerLane * lane + i]);
  }
  float other[kDimsPerLane];
#pragma unroll
  for (int i = 0; i < kDimsPerLane; ++i) {
    other[i] = __shfl_xor_sync(0xffffffffu, x[i], kRotHalf / kDimsPerLane);
  }
  if (lane >= kQwen38Rotary / kDimsPerLane) return;
  const bool first = lane < kRotHalf / kDimsPerLane;
#pragma unroll
  for (int i = 0; i < kDimsPerLane; ++i) {
    const int freq = (kDimsPerLane * lane + i) % kRotHalf;
    const int axis = thinker::MropeAxis<Qwen38AttentionDims>(freq);
    const __nv_bfloat16* cs =
        f.cos_sin + int64_t{p.positions[axis * p.tokens + t]} * kQwen38Rotary;
    const float c = __bfloat162float(cs[freq]);
    const float s = __bfloat162float(cs[kRotHalf + freq]);
    x[i] = first ? x[i] * c - other[i] * s : x[i] * c + other[i] * s;
  }
}

__device__ __forceinline__ uint4 PackBf16x8(const float (&x)[kDimsPerLane]) {
  uint4 out;
  uint32_t* words = &out.x;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __nv_bfloat162 pair = __floats2bfloat162_rn(x[2 * i], x[2 * i + 1]);
    words[i] = *reinterpret_cast<const uint32_t*>(&pair);
  }
  return out;
}

__device__ __forceinline__ void UnpackBf16x8(uint4 v,
                                             float (&x)[kDimsPerLane]) {
  const uint32_t* words = &v.x;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(
        *reinterpret_cast<const __nv_bfloat162*>(&words[i]));
    x[2 * i] = f.x;
    x[2 * i + 1] = f.y;
  }
}

// A warp per (token, KV head): the key after QK-norm and M-RoPE, and the
// value, bf16, to the token's cache slot.
__device__ void ChunkKv(const Qwen38PrefillParams& p,
                        const Qwen38LayerParams& f, int pos0) {
  const int lane = threadIdx.x % 32;
  for (int item = GlobalWarp(); item < p.tokens * kQwen38KvHeads;
       item += GridWarps()) {
    const int t = item / kQwen38KvHeads;
    const int g = item % kQwen38KvHeads;
    const float* row = p.proj + int64_t{t} * kQkvRows;
    float k[kDimsPerLane];
    float v[kDimsPerLane];
    LoadLane(row + kQDim + g * kHead, lane, k);
    LoadLane(row + kQDim + kKvDim + g * kHead, lane, v);
    NormRope(p, f, f.k_norm, t, lane, k);
    const int64_t slot = thinker::SlotOf(f.kv, pos0 + t);
    reinterpret_cast<uint4*>(thinker::SlotAt(f.kv, f.kv.key, g, slot))[lane] =
        PackBf16x8(k);
    reinterpret_cast<uint4*>(thinker::SlotAt(f.kv, f.kv.value, g, slot))[lane] =
        PackBf16x8(v);
  }
}

// Tokens an attention CTA takes: with the group's kGqa heads, 48 (head, token)
// items, three a warp.
constexpr int kAttnTokens = 8;
constexpr int kAttnItems = kAttnTokens * kGqa;
constexpr int kItemsPerWarp = kAttnItems / kWarps;
static_assert(kItemsPerWarp * kWarps == kAttnItems);
// Keys a tile stages, and keys a warp scores together, their warp sums
// interleaved.
constexpr int kTileKeys = 32;
constexpr int kKeysAtOnce = 4;

// The spans of `keys` cache positions: span s covers positions
// [s × kQwen38PrefillAttnSpan, (s + 1) × kQwen38PrefillAttnSpan).
__device__ __forceinline__ int SpansOf(int keys) {
  return (keys + kQwen38PrefillAttnSpan - 1) / kQwen38PrefillAttnSpan;
}

struct AttentionShared {
  uint4 key[kTileKeys][kHead / 8];
  uint4 value[kTileKeys][kHead / 8];
};

// One query's running softmax: max, sum and unnormalized output.
struct Running {
  float m;
  float l;
  float o[kDimsPerLane];
};

// The online-softmax step over keys 0 .. count − 1 of a batch of
// kKeysAtOnce at `first`, as the decode's Attend.
__device__ __forceinline__ void AttendBatch(const AttentionShared& sh,
                                            int first, int count,
                                            const float (&q)[kDimsPerLane],
                                            float scale, int lane,
                                            Running& run) {
  float score[kKeysAtOnce];
  float v[kKeysAtOnce][kDimsPerLane];
#pragma unroll
  for (int u = 0; u < kKeysAtOnce; ++u) {
    float k[kDimsPerLane];
    const int j = min(first + u, kTileKeys - 1);
    UnpackBf16x8(sh.key[j][lane], k);
    UnpackBf16x8(sh.value[j][lane], v[u]);
    float dot = 0.f;
#pragma unroll
    for (int i = 0; i < kDimsPerLane; ++i) dot = fmaf(q[i], k[i], dot);
    score[u] = dot;
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
#pragma unroll
    for (int u = 0; u < kKeysAtOnce; ++u) {
      score[u] += __shfl_xor_sync(0xffffffffu, score[u], offset);
    }
  }
#pragma unroll
  for (int u = 0; u < kKeysAtOnce; ++u) {
    if (u >= count) break;
    const float s = score[u] * scale;
    const float m_new = fmaxf(run.m, s);
    const float correction = expf(run.m - m_new);
    const float weight = expf(s - m_new);
    run.l = run.l * correction + weight;
#pragma unroll
    for (int i = 0; i < kDimsPerLane; ++i) {
      run.o[i] = fmaf(run.o[i], correction, weight * v[u][i]);
    }
    run.m = m_new;
  }
}

// Item (token t, query head h)'s partial over span s, as [t][h][s].
__device__ __forceinline__ int64_t PartialAt(const Qwen38PrefillParams& p,
                                             int t, int h, int s) {
  return (int64_t{t} * kQwen38QHeads + h) * p.max_spans + s;
}

// The first token of attention block `block`, and the spans its keys take:
// every position up to its last token's.
__device__ __forceinline__ int BlockSpans(const Qwen38PrefillParams& p,
                                          int pos0, int block) {
  return SpansOf(pos0 + min(p.tokens, (block + 1) * kAttnTokens));
}

// A CTA per (KV group g, kAttnTokens tokens, span of keys): the group's keys
// in the span up to the block's last token, staged a tile at a time; each
// warp's items attend to the keys at or before their own position and write
// a partial (max, sum, unnormalized output), empty where they see no key,
// which the merge never reads.
// The spans follow the positions alone, so a long cache splits over many
// CTAs at any grid.
__device__ void PrefillAttention(const Qwen38PrefillParams& p,
                                 const Qwen38LayerParams& f, int pos0,
                                 float* smem) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const float scale = rsqrtf(static_cast<float>(kHead));
  auto& sh = *reinterpret_cast<AttentionShared*>(smem);
  const int blocks = (p.tokens + kAttnTokens - 1) / kAttnTokens;
  int per_group = 0;
  for (int b = 0; b < blocks; ++b) per_group += BlockSpans(p, pos0, b);
  for (int unit = blockIdx.x; unit < kQwen38KvHeads * per_group;
       unit += gridDim.x) {
    const int g = unit / per_group;
    int span = unit % per_group;
    int block = 0;
    while (span >= BlockSpans(p, pos0, block)) {
      span -= BlockSpans(p, pos0, block);
      ++block;
    }
    const int t0 = block * kAttnTokens;
    const int begin = span * kQwen38PrefillAttnSpan;
    const int end = min(pos0 + min(p.tokens, t0 + kAttnTokens),
                        begin + kQwen38PrefillAttnSpan);
    float q[kItemsPerWarp][kDimsPerLane];
    Running run[kItemsPerWarp];
#pragma unroll
    for (int k = 0; k < kItemsPerWarp; ++k) {
      const int item = warp + k * kWarps;
      const int t = min(t0 + item / kGqa, p.tokens - 1);
      const int h = g * kGqa + item % kGqa;
      LoadLane(p.proj + int64_t{t} * kQkvRows + h * kHead, lane, q[k]);
      NormRope(p, f, f.q_norm, t, lane, q[k]);
      run[k].m = -INFINITY;
      run[k].l = 0.f;
#pragma unroll
      for (int i = 0; i < kDimsPerLane; ++i) run[k].o[i] = 0.f;
    }
    for (int base = begin; base < end; base += kTileKeys) {
      const int keys = min(kTileKeys, end - base);
      constexpr int kVecs = kHead / 8;
      for (int n = threadIdx.x; n < keys * 2 * kVecs; n += kThreads) {
        const int j = n / (2 * kVecs);
        const bool is_value = n / kVecs % 2 != 0;
        const uint4* src = reinterpret_cast<const uint4*>(thinker::KvAt(
            f.kv, is_value ? f.kv.value : f.kv.key, g, base + j));
        (is_value ? sh.value : sh.key)[j][n % kVecs] = __ldcg(src + n % kVecs);
      }
      __syncthreads();
#pragma unroll
      for (int k = 0; k < kItemsPerWarp; ++k) {
        const int item = warp + k * kWarps;
        const int t = t0 + item / kGqa;
        if (t >= p.tokens) continue;
        const int count = min(keys, pos0 + t + 1 - base);
        for (int j = 0; j < count; j += kKeysAtOnce) {
          AttendBatch(sh, j, count - j, q[k], scale, lane, run[k]);
        }
      }
      __syncthreads();
    }
#pragma unroll
    for (int k = 0; k < kItemsPerWarp; ++k) {
      const int item = warp + k * kWarps;
      const int t = t0 + item / kGqa;
      if (t >= p.tokens) continue;
      const int64_t at = PartialAt(p, t, g * kGqa + item % kGqa, span);
      if (lane == 0) {
        p.partial_ml[2 * at] = run[k].m;
        p.partial_ml[2 * at + 1] = run[k].l;
      }
      float* o = p.partial_o + at * kHead + kDimsPerLane * lane;
#pragma unroll
      for (int i = 0; i < kDimsPerLane; i += 4) {
        *reinterpret_cast<float4*>(o + i) =
            make_float4(run[k].o[i], run[k].o[i + 1], run[k].o[i + 2],
                        run[k].o[i + 3]);
      }
    }
  }
}

// A warp per (token, query head): the partials of the spans its keys reach,
// each holding at least one of them, merged in span order as the decode's
// Merge, then gated into attn.
__device__ void MergeAttention(const Qwen38PrefillParams& p, int pos0) {
  const int lane = threadIdx.x % 32;
  for (int item = GlobalWarp(); item < p.tokens * kQwen38QHeads;
       item += GridWarps()) {
    const int t = item / kQwen38QHeads;
    const int h = item % kQwen38QHeads;
    const int spans = SpansOf(pos0 + t + 1);
    float m = -INFINITY;
    for (int s = 0; s < spans; ++s) {
      const float2 ml = __ldcg(reinterpret_cast<const float2*>(p.partial_ml) +
                               PartialAt(p, t, h, s));
      m = fmaxf(m, ml.x);
    }
    float l = 0.f;
    float o[kDimsPerLane] = {};
    for (int s = 0; s < spans; ++s) {
      const int64_t at = PartialAt(p, t, h, s);
      const float2 ml =
          __ldcg(reinterpret_cast<const float2*>(p.partial_ml) + at);
      const float weight = expf(ml.x - m);
      l = fmaf(ml.y, weight, l);
      float part[kDimsPerLane];
      LoadLane(p.partial_o + at * kHead, lane, part);
#pragma unroll
      for (int i = 0; i < kDimsPerLane; ++i) o[i] = fmaf(part[i], weight, o[i]);
    }
    float gate[kDimsPerLane];
    LoadLane(p.proj + int64_t{t} * kQkvRows + kGateRow + h * kHead, lane, gate);
    const float inv = 1.f / l;
#pragma unroll
    for (int i = 0; i < kDimsPerLane; ++i) {
      o[i] = o[i] * inv * thinker::Sigmoid(gate[i]);
    }
    reinterpret_cast<uint4*>(p.attn + int64_t{t} * kQDim + h * kHead)[lane] =
        PackBf16x8(o);
  }
}

// ---------------------------------------------------------------- M2, M3

__device__ void GateUp(const Qwen38PrefillParams& p, const Qwen38MlpParams& m,
                       float* partials) {
  Project<kDim, prefill::RowPairs>(
      p, m.w13, 2 * kFfn, p.h, partials,
      [&](int row, int t, float gate, float up) {
        p.act[int64_t{t} * kFfn + row / 2] = __float2bfloat16(Silu(gate) * up);
      });
}

// ------------------------------------------------------------------ the kernel

constexpr int kSmemBytes = std::max<int>(
    {static_cast<int>(prefill::kDenseSmemFloats * sizeof(float)),
     prefill::FixedDenseSmemBytes(SlicesFor(kDim)),
     prefill::FixedDenseSmemBytes(SlicesFor(kQDim)),
     static_cast<int>(sizeof(DeltaShared)),
     static_cast<int>(sizeof(AttentionShared)),
     static_cast<int>(kDim * sizeof(__nv_bfloat16))});

// Copies every token's residual into hidden row `row`.
__device__ void DumpHidden(const Qwen38PrefillParams& p, int row) {
  if (p.hidden == nullptr) return;
  const int n = p.tokens * kDim;
  for (int i = blockIdx.x * kThreads + threadIdx.x; i < n;
       i += gridDim.x * kThreads) {
    p.hidden[int64_t{row} * n + i] = __ldcg(p.residual + i);
  }
}

__global__ void __launch_bounds__(kThreads, 1)
    Qwen38PrefillKernel(const __grid_constant__ Qwen38PrefillParams p) {
  extern __shared__ float smem[];
  __shared__ float red[kWarps];
  GridBarrier barrier(p.sync, p.error, p.timeout_ns, 0);
  int64_t* stamps =
      p.profile == nullptr
          ? nullptr
          : p.profile +
                int64_t{blockIdx.x} * 2 * Qwen38PrefillBarriers(p.num_layers);
  // The arrival stamp waits for the whole CTA to finish the phase.
  auto sync = [&] {
    if (stamps != nullptr) {
      __syncthreads();
      if (threadIdx.x == 0) stamps[2 * barrier.index()] = GlobalTimer();
    }
    barrier.Sync();
    if (stamps != nullptr && threadIdx.x == 0) {
      stamps[2 * (barrier.index() - 1) + 1] = GlobalTimer();
    }
  };
  const int pos0 = *p.pos0;
  const int64_t chunk = int64_t{p.tokens} * kDim;

  for (int layer = 0; layer < p.num_layers; ++layer) {
    float* dump = p.hidden != nullptr ? p.hidden + layer * chunk : nullptr;
    const Qwen38MlpParams* mlp;
    if (Qwen38IsFull(layer)) {
      const Qwen38LayerParams& f = p.full[Qwen38FullIndex(layer)];
      TokenNorm(p, f.input_norm, f.eps, dump, red);
      sync();
      Project<kDim>(p, f.wqkv, kQkvRows, p.h, smem,
                    [&](int row, int t, float y) {
                      p.proj[int64_t{t} * kQkvRows + row] = y;
                    });
      sync();
      ChunkKv(p, f, pos0);
      sync();
      PrefillAttention(p, f, pos0, smem);
      sync();
      MergeAttention(p, pos0);
      sync();
      Project<kQDim>(p, f.wo, kDim, p.attn, smem, AddToResidual(p));
      mlp = &f.mlp;
    } else {
      const GdnParams& w = p.linear[Qwen38LinearIndex(layer)];
      TokenNorm(p, w.norm, w.eps, dump, red);
      sync();
      InProjection(p, w, smem);
      sync();
      Conv(p, w);
      sync();
      DeltaRule(p, w, smem);
      sync();
      GatedNorm(p, w);
      sync();
      OutProjection(p, w, smem);
      mlp = &p.linear_mlp[Qwen38LinearIndex(layer)];
    }
    sync();
    TokenNorm(p, mlp->post_norm, mlp->eps, nullptr, red);
    sync();
    GateUp(p, *mlp, smem);
    sync();
    Project<kFfn>(p, mlp->w2, kDim, p.act, smem, AddToResidual(p));
    sync();
  }
  DumpHidden(p, p.num_layers);
  if (p.logits == nullptr) return;

  // Every CTA normalizes the last token itself, then takes its LM head rows,
  // as the decode does.
  auto* xs = reinterpret_cast<uint4*>(smem);
  RmsNorm<qwen38::NormDims>(p.residual + int64_t{p.tokens - 1} * kDim, nullptr,
                            p.final_norm, p.eps,
                            reinterpret_cast<__nv_bfloat16*>(xs), nullptr, red);
  __syncthreads();
  qwen38::LmHeadRows<1>(
      p.lm_head, p.vocab, xs,
      [&](int row, int, float logit) { p.logits[row] = logit; });
}

// One projection alone (LaunchQwen38Projection).
template <int kK, int kSlices>
__global__ void __launch_bounds__(kThreads, 1)
    Qwen38ProjectionKernel(const Weight w, int n, const __nv_bfloat16* x,
                           int tokens, float* y) {
  extern __shared__ float smem[];
  prefill::FixedDense<kSlices>(Int4Of(w), kK, n, x, tokens, smem,
                               [&](int row, int t, float v) {
                                 y[int64_t{t} * n + row] = v;
                               });
}

template <int kK, int kSlices>
cudaError_t LaunchProjection(const Weight& w, int n, const __nv_bfloat16* x,
                             int tokens, float* y, int num_ctas,
                             cudaStream_t stream) {
  auto* kernel = Qwen38ProjectionKernel<kK, kSlices>;
  constexpr int kBytes = prefill::FixedDenseSmemBytes(kSlices);
  const cudaError_t err = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kBytes);
  if (err != cudaSuccess) return err;
  kernel<<<num_ctas, kThreads, kBytes, stream>>>(w, n, x, tokens, y);
  return cudaGetLastError();
}

template <int kK>
cudaError_t LaunchProjection(const Weight& w, int n, const __nv_bfloat16* x,
                             int tokens, bool whole_k, float* y, int num_ctas,
                             cudaStream_t stream) {
  return whole_k ? LaunchProjection<kK, 1>(w, n, x, tokens, y, num_ctas,
                                           stream)
                 : LaunchProjection<kK, SlicesFor(kK)>(w, n, x, tokens, y,
                                                       num_ctas, stream);
}

}  // namespace

cudaError_t LaunchQwen38Projection(const Weight& w, int n, int k,
                                   const __nv_bfloat16* x, int tokens,
                                   bool whole_k, float* y, int num_ctas,
                                   cudaStream_t stream) {
  switch (k) {
    case kQwen38Dim:
      return LaunchProjection<kQwen38Dim>(w, n, x, tokens, whole_k, y,
                                          num_ctas, stream);
    case kQwen38QDim:
      return LaunchProjection<kQwen38QDim>(w, n, x, tokens, whole_k, y,
                                           num_ctas, stream);
    case kQwen38Ffn:
      return LaunchProjection<kQwen38Ffn>(w, n, x, tokens, whole_k, y,
                                          num_ctas, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

cudaError_t LaunchQwen38Prefill(const Qwen38PrefillParams& params,
                                int num_ctas, cudaStream_t stream) {
  cudaError_t err =
      cudaMemsetAsync(params.sync, 0, kSyncWords * sizeof(unsigned), stream);
  if (err != cudaSuccess) return err;
  err = cudaFuncSetAttribute(Qwen38PrefillKernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kSmemBytes);
  if (err != cudaSuccess) return err;
  Qwen38PrefillKernel<<<num_ctas, kThreads, kSmemBytes, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace s2mk
