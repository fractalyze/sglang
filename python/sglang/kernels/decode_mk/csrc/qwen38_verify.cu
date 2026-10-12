// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// The verify step of Qwen3.8-27B (qwen38_decode.h): kT tokens through the
// decode step's phases in one persistent launch, with the same grid barriers.
// The full-attention and MLP blocks are qwen38_tokens.cuh's, over the
// layers' kInt4Zp projections; the linear-attention block projects on tensor
// cores the same way. Every token's input is staged side by side in shared
// memory first, except down's, which the gate-up phase left in global memory.
// A token's results are the same bits on every run, at any CTA count with the
// same attention chunks, and whatever tokens follow it in the step. They are
// not in the decode step's order: a token's logits match its decode step's
// within bf16's error, not bitwise. The rest is the decode step's arithmetic:
// the conv and the delta rule run the tokens in order on the same thread or
// warp, each token's states going to its own slot.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "barrier.cuh"
#include "decoder_layer.cuh"
#include "gdn_block.cuh"
#include "gemv_core.cuh"
#include "int4_gemv_core.cuh"
#include "int4_mma_core.cuh"
#include "prefill_dense.cuh"
#include "qwen38_decode.h"
#include "qwen38_layer.cuh"
#include "qwen38_prefetch.cuh"
#include "qwen38_tokens.cuh"
#include "thinker_layer.cuh"

namespace s2mk {
namespace {

using qwen38::PrefetchNext;

constexpr int kDim = kQwen38Dim;
using qwen38::tokens::AddToResidual;
using qwen38::tokens::AttentionBlock;
using qwen38::tokens::AttentionShared;
using qwen38::tokens::Bf16Of;
using qwen38::tokens::Int4Of;
using qwen38::tokens::Int4Projections;
using qwen38::tokens::MlpBlock;
using qwen38::tokens::MlpShared;
using qwen38::tokens::Project;
using qwen38::tokens::StagedProjection;
using qwen38::tokens::Workspace;
using prefill::DenseBegin;

// The most rows a CTA owns of an n-row projection: whole 16-row blocks
// (DenseBegin), at kGdnMinCtas CTAs.
constexpr int MaxDenseRows(int n) {
  return (n / prefill::kBlockRows + kGdnMinCtas - 1) / kGdnMinCtas *
         prefill::kBlockRows;
}

// Each phase's shared memory overlays the others', so that a step of
// kQwen38MaxVerifyTokens tokens fits kSharedBudget (below).
template <int kT>
struct GdnShared {
  union {
    struct {
      StagedProjection<kT, kDim> x;
      // The rows below the gates, for routing.
      float ys[kT * MaxDenseRows(kGdnInt4Rows)];
    } in;
    // [token][head][0]: the query, normalized and scaled; [1]: the key.
    float4 qk[kT][gdn::kMaxHeadsPerCta][2][gdn::kHeadDim / 4];
    StagedProjection<kT, kGdnValueDim> out;
  };
  float red[kWarps];
};

template <int kT>
union Shared {
  GdnShared<kT> gdn;
  AttentionShared<kT> attention;
  MlpShared<kT> mlp;
};
// SM120 splits one array between L1 and shared memory, the shared part in
// the steps the CUDA programming guide lists for compute capability 12.0,
// each less what the system reserves a CTA. Within the step that leaves L1
// half the array, the tiles' per-row scale and zero-point loads keep the L1
// they count on to hold each sector for the next groups; the next step up
// leaves them less than half of that.
constexpr int kCarveoutStep = 64 * 1024;
constexpr int kReservedPerCta = 1024;
constexpr int kSharedBudget = kCarveoutStep - kReservedPerCta;
static_assert(sizeof(Shared<kQwen38MaxVerifyTokens>) <= kSharedBudget);
static_assert(kQwen38MaxVerifyTokens <= kMmaTokens,
              "a step's tokens are one n-tile");

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
  StagedProjection<kT, kDim>& in = sh.in.x;
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(in.xs);
  for (int t = 0; t < kT; ++t) {
    GdnInputNorm(p, v.step.residual + t * kDim, xs + t * kDim, sh.red);
  }
  __syncthreads();
  const SharedX x{in.xs, kDim};
  const int begin = DenseBegin(kInt4Rows, blockIdx.x);
  const int rows = DenseBegin(kInt4Rows, blockIdx.x + 1) - begin;
  Project<kT, kDim>(Int4Of(p.in_proj), kInt4Rows, x, in.partials,
                    [&](int row, int t, float y) {
                      sh.in.ys[t * rows + row - begin] = y;
                    });
  for (int i = threadIdx.x; i < rows; i += kThreads) {
    GdnRoute<kT>(p, v, begin + i, sh.in.ys, rows, i);
  }
  Project<kT, kDim>(Bf16Of(p.gates), kGateRows, x, in.partials,
                    [&](int row, int t, float y) {
                      GdnRouteGate(p, v, t, row, y);
                    });
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
  StagedProjection<kT, kValueDim>& out = sh.out;
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(out.xs);
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
  __syncthreads();
  const SharedX x{out.xs, kValueDim};
  if (p.out_proj.format == WeightFormat::kBf16) {
    Project<kT, kValueDim>(Bf16Of(p.out_proj), kDim, x, out.partials,
                           AddToResidual(v.step.residual));
  } else {
    Project<kT, kValueDim>(Int4Of(p.out_proj), kDim, x, out.partials,
                           AddToResidual(v.step.residual));
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
  const Workspace ws{p.residual, v.qkv, v.partial_ml, v.partial_o, v.act};

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
      AttentionBlock<kT, Int4Projections>(full, ws, pos0, sync, sh.attention);
      mlp = &full.mlp;
    } else {
      GdnBlock<kT>(p.linear[Qwen38LinearIndex(layer)], v, sync, sh.gdn);
      mlp = &p.linear_mlp[Qwen38LinearIndex(layer)];
    }
    sync();
    MlpBlock<kT, Int4Projections>(*mlp, ws, sync, sh.mlp);
    sync();
  }

  StagedProjection<kT, kDim>& in = sh.gdn.in.x;
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(in.xs);
  for (int t = 0; t < kT; ++t) {
    RmsNorm<qwen38::NormDims>(
        p.residual + t * kDim, nullptr, p.final_norm, p.eps, xs + t * kDim,
        blockIdx.x == 0 ? v.final_hidden + t * kDim : nullptr, sh.gdn.red);
  }
  __syncthreads();
  const SharedX x{in.xs, kDim};
  const auto logit = [&](int row, int t, float y) {
    p.logits[int64_t{t} * p.vocab + row] = y;
  };
  switch (p.lm_head.format) {
    case WeightFormat::kBf16:
      Project<kT, kDim>(Bf16Of(p.lm_head), p.vocab, x, in.partials, logit);
      return;
    case WeightFormat::kInt4Zp:
      Project<kT, kDim>(Int4Of(p.lm_head), p.vocab, x, in.partials, logit);
      return;
  }
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
  static_assert(kQwen38MaxVerifyTokens == 5);
  switch (params.num_tokens) {
    case 1:
      return Launch<1>(params, num_ctas, stream);
    case 2:
      return Launch<2>(params, num_ctas, stream);
    case 3:
      return Launch<3>(params, num_ctas, stream);
    case 4:
      return Launch<4>(params, num_ctas, stream);
    case 5:
      return Launch<5>(params, num_ctas, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace s2mk
