// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// The phases of a linear-attention layer's decode step (gdn.h), written as a
// block a persistent kernel calls on every CTA, with the grid barrier passed
// in as `sync`.
//
// The delta rule for value head h, with S its dk × dv state, k and q its key
// head's L2-normalized key and query (q also scaled by dk^(-1/2)), v its value,
// β = sigmoid(b) and g = -exp(A_log) × softplus(a + dt_bias):
//
//   S ← exp(g) × S
//   S ← S + k ⊗ (β × (v − Sᵀ k))
//   o = Sᵀ q
//
// Column j of S (its dk values in row j of the stored [dv, dk] block) meets
// only v[j], so the update and the readout of one column depend on no other
// column. Phase 2 hands each CTA a contiguous run of the 48 × 128 columns and
// each warp whole columns: a warp loads its columns once, applies the decay,
// the update and the readout in registers, and stores them once. Every sum
// is a lane's four products followed by a fixed butterfly, and the GEMVs
// give each row to one warp, so no value depends on how the grid is split.

#ifndef S2MK_CSRC_GDN_BLOCK_CUH_
#define S2MK_CSRC_GDN_BLOCK_CUH_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "gdn.h"
#include "gemv_core.cuh"
#include "int4_gemv_core.cuh"

namespace s2mk {
namespace gdn {

constexpr int kDim = kGdnDim;
constexpr int kVHeads = kGdnVHeads;
constexpr int kHeadDim = kGdnHeadDim;
constexpr int kKeyDim = kGdnKeyDim;
constexpr int kValueDim = kGdnValueDim;
constexpr int kConvDim = kGdnConvDim;
constexpr int kConvTaps = kGdnConvWidth;
constexpr int kInt4Rows = kGdnInt4Rows;
constexpr int kGateRows = kGdnGateRows;
constexpr int kHeadsPerKHead = kGdnVHeads / kGdnKHeads;
// State columns, one per (value head, dv index).
constexpr int kColumns = kValueDim;
// The query's scale after its L2 norm, kHeadDim^(-1/2). A lane holds
// kHeadDim / 32 = 4 consecutive values of a head as one float4.
constexpr float kQueryScale = 0.08838834764831845f;
static_assert(kHeadDim == 128);

// Columns a warp holds in registers at once; a CTA owns at most
// kWarps × kColumnsPerWarp of them, so kGdnMinCtas CTAs cover kColumns.
constexpr int kColumnsPerWarp = 4;
constexpr int kMaxColumnsPerCta = kWarps * kColumnsPerWarp;
static_assert(kGdnMinCtas * kMaxColumnsPerCta >= kColumns);
// A run of at most kHeadDim columns meets at most two value heads.
static_assert(kMaxColumnsPerCta <= kHeadDim);
constexpr int kMaxHeadsPerCta = 2;
constexpr int kMaxRowsPerCta = (kInt4Rows + kGdnMinCtas - 1) / kGdnMinCtas;
static_assert(kValueDim >= kDim && kInt4Rows >= kDim);

struct Shared {
  uint4 xs[kValueDim / kVecElems];  // either GEMV's bf16 input
  float ys[kMaxRowsPerCta];
  float red[kWarps];
  // [head][0]: the query, normalized and scaled; [head][1]: the key,
  // normalized.
  float4 qk[kMaxHeadsPerCta][2][kHeadDim / 4];
};

__device__ __forceinline__ float Silu(float x) { return x / (1.f + expf(-x)); }

__device__ __forceinline__ float Sigmoid(float x) {
  return 1.f / (1.f + expf(-x));
}

// torch.nn.functional.softplus with its default threshold of 20.
__device__ __forceinline__ float Softplus(float x) {
  return x > 20.f ? x : log1pf(expf(x));
}

__device__ __forceinline__ float Dot4(float4 a, float4 b) {
  float acc = a.x * b.x;
  acc = fmaf(a.y, b.y, acc);
  acc = fmaf(a.z, b.z, acc);
  return fmaf(a.w, b.w, acc);
}

// Σ v over the CTA, in a fixed order. Called by every thread; every thread
// gets the sum.
__device__ __forceinline__ float BlockSum(float v, float* red) {
  v = WarpSum(v);
  if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = v;
  __syncthreads();
  float sum = 0.f;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) sum += red[w];
  __syncthreads();
  return sum;
}

// The first state column CTA `cta` owns; ColumnBegin(cta + 1) ends its run.
__device__ __forceinline__ int ColumnBegin(int cta) {
  return RowBegin(kColumns, cta);
}

// xs = RMSNorm(residual_in) × (1 + norm) in bf16.
__device__ inline void InputNorm(const GdnParams& p, Shared& sh) {
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(sh.xs);
  float ss = 0.f;
  for (int i = threadIdx.x; i < kDim; i += kThreads) {
    const float x = __ldcg(p.residual_in + i);
    ss = fmaf(x, x, ss);
  }
  const float inv = rsqrtf(BlockSum(ss, sh.red) / kDim + p.eps);
  for (int i = threadIdx.x; i < kDim; i += kThreads) {
    const float scale = 1.f + __bfloat162float(p.norm[i]);
    xs[i] = __float2bfloat16(__ldcg(p.residual_in + i) * inv * scale);
  }
}

// Takes input-projection row `row`'s value x where it goes: a conv channel
// through its conv (shifting x into its state) and SiLU, z as is, b and a
// into the head's β and decay.
__device__ __forceinline__ void Route(const GdnParams& p, int row, float x) {
  if (row < kConvDim) {
    float* past = p.conv_state + int64_t{row} * (kConvTaps - 1);
    const __nv_bfloat16* w = p.conv + int64_t{row} * kConvTaps;
    float taps[kConvTaps];
#pragma unroll
    for (int t = 0; t < kConvTaps - 1; ++t) taps[t] = past[t];
    taps[kConvTaps - 1] = x;
    float acc = 0.f;
#pragma unroll
    for (int t = 0; t < kConvTaps; ++t) {
      acc = fmaf(__bfloat162float(w[t]), taps[t], acc);
    }
#pragma unroll
    for (int t = 0; t < kConvTaps - 1; ++t) past[t] = taps[t + 1];
    p.mixed[row] = Silu(acc);
  } else if (row < kConvDim + kValueDim) {
    p.z[row - kConvDim] = x;
  } else if (row < kConvDim + kValueDim + kVHeads) {
    p.beta[row - kConvDim - kValueDim] = Sigmoid(x);
  } else {
    const int h = row - kConvDim - kValueDim - kVHeads;
    const float g = -expf(p.a_log[h]) * Softplus(x + p.dt_bias[h]);
    p.decay[h] = expf(g);
  }
}

// Phase 1: this CTA's share of the int4 input-projection rows and of the
// gate rows, each through Route.
__device__ inline void InProjection(const GdnParams& p, Shared& sh) {
  InputNorm(p, sh);
  const int begin = RowBegin(kInt4Rows, blockIdx.x);
  const int rows = RowBegin(kInt4Rows, blockIdx.x + 1) - begin;
  Int4ZpGemvRows<1, 32, kDim / (32 * kInt4Group)>(
      static_cast<const uint4*>(p.in_proj.data), p.in_proj.scales,
      p.in_proj.zeros, begin, rows, sh.xs, sh.ys);
  for (int i = threadIdx.x; i < rows; i += kThreads) {
    Route(p, begin + i, sh.ys[i]);
  }
  Bf16WarpRows<kDim>(static_cast<const __nv_bfloat16*>(p.gates.data),
                     RowBegin(kGateRows, blockIdx.x),
                     RowBegin(kGateRows, blockIdx.x + 1), sh.xs,
                     [&](int row, float x) { Route(p, kInt4Rows + row, x); });
}

// Phase 2: the delta rule on this CTA's state columns, writing their readout
// into core.
__device__ inline void DeltaRule(const GdnParams& p, Shared& sh) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int begin = ColumnBegin(blockIdx.x);
  const int columns = ColumnBegin(blockIdx.x + 1) - begin;
  if (columns == 0) return;
  const int first_head = begin / kHeadDim;
  const int heads = (begin + columns - 1) / kHeadDim - first_head + 1;

  // Start the state loads before normalizing q and k.
  float4 s[kColumnsPerWarp];
#pragma unroll
  for (int u = 0; u < kColumnsPerWarp; ++u) {
    const int c = warp + u * kWarps;
    if (c < columns) {
      s[u] = __ldcg(reinterpret_cast<const float4*>(
                        p.state + int64_t{begin + c} * kHeadDim) + lane);
    }
  }

  // Warp 2t + 0 normalizes the query, warp 2t + 1 the key, of head t.
  if (warp < 2 * heads) {
    const int t = warp / 2;
    const int is_key = warp % 2;
    const int k_head = (first_head + t) / kHeadsPerKHead;
    const float* src = p.mixed + is_key * kKeyDim + k_head * kHeadDim;
    const float4 x = __ldcg(reinterpret_cast<const float4*>(src) + lane);
    float inv = rsqrtf(WarpSum(Dot4(x, x)) + 1e-6f);
    if (!is_key) inv *= kQueryScale;
    sh.qk[t][is_key][lane] = make_float4(x.x * inv, x.y * inv, x.z * inv,
                                         x.w * inv);
  }
  __syncthreads();

  const float* v = p.mixed + 2 * kKeyDim;
#pragma unroll
  for (int u = 0; u < kColumnsPerWarp; ++u) {
    const int c = warp + u * kWarps;
    if (c >= columns) break;
    const int column = begin + c;
    const int head = column / kHeadDim;
    const float4 q = sh.qk[head - first_head][0][lane];
    const float4 k = sh.qk[head - first_head][1][lane];
    const float decay = __ldcg(p.decay + head);
    float4 col = s[u];
    col.x *= decay;
    col.y *= decay;
    col.z *= decay;
    col.w *= decay;
    const float delta =
        (__ldcg(v + column) - WarpSum(Dot4(col, k))) * __ldcg(p.beta + head);
    col.x = fmaf(k.x, delta, col.x);
    col.y = fmaf(k.y, delta, col.y);
    col.z = fmaf(k.z, delta, col.z);
    col.w = fmaf(k.w, delta, col.w);
    const float o = WarpSum(Dot4(col, q));
    __stcg(reinterpret_cast<float4*>(p.state + int64_t{column} * kHeadDim) +
               lane,
           col);
    if (lane == 0) p.core[column] = o;
  }
}

// Phase 3: xs = RMSNorm(core) × out_norm × SiLU(z) per value head, in bf16,
// then this CTA's out_proj rows added onto residual_in.
__device__ inline void OutProjection(const GdnParams& p, Shared& sh) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(sh.xs);
  const float4 w = make_float4(__bfloat162float(p.out_norm[4 * lane]),
                               __bfloat162float(p.out_norm[4 * lane + 1]),
                               __bfloat162float(p.out_norm[4 * lane + 2]),
                               __bfloat162float(p.out_norm[4 * lane + 3]));
  for (int h = warp; h < kVHeads; h += kWarps) {
    const int at = h * kHeadDim + 4 * lane;
    const float4 o = __ldcg(reinterpret_cast<const float4*>(p.core + at));
    const float4 z = __ldcg(reinterpret_cast<const float4*>(p.z + at));
    const float inv = rsqrtf(WarpSum(Dot4(o, o)) / kHeadDim + p.eps);
    xs[at] = __float2bfloat16(o.x * inv * w.x * Silu(z.x));
    xs[at + 1] = __float2bfloat16(o.y * inv * w.y * Silu(z.y));
    xs[at + 2] = __float2bfloat16(o.z * inv * w.z * Silu(z.z));
    xs[at + 3] = __float2bfloat16(o.w * inv * w.w * Silu(z.w));
  }
  const int begin = RowBegin(kDim, blockIdx.x);
  const int rows = RowBegin(kDim, blockIdx.x + 1) - begin;
  if (p.out_proj.format == WeightFormat::kBf16) {
    __syncthreads();
    Bf16WarpRows<kValueDim>(
        static_cast<const __nv_bfloat16*>(p.out_proj.data), begin,
        begin + rows, sh.xs, [&](int row, float y) {
          p.residual[row] = __ldcg(p.residual_in + row) + y;
        });
    return;
  }
  Int4ZpGemvRows<1, 32, kValueDim / (32 * kInt4Group)>(
      static_cast<const uint4*>(p.out_proj.data), p.out_proj.scales,
      p.out_proj.zeros, begin, rows, sh.xs, sh.ys);
  for (int i = threadIdx.x; i < rows; i += kThreads) {
    p.residual[begin + i] = __ldcg(p.residual_in + begin + i) + sh.ys[i];
  }
}

// The layer on every CTA, calling sync() for its two barriers.
template <typename Sync>
__device__ void GdnBlock(const GdnParams& p, Sync& sync, Shared& sh) {
  InProjection(p, sh);
  sync();
  DeltaRule(p, sh);
  sync();
  OutProjection(p, sh);
}

}  // namespace gdn
}  // namespace s2mk

#endif  // S2MK_CSRC_GDN_BLOCK_CUH_
