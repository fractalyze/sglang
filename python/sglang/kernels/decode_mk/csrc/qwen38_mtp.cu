// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// One step of Qwen3.8-27B's MTP head (qwen38_mtp.h) as one persistent launch
// of one CTA per SM. The layer runs the phases of qwen38_layer.cuh's blocks on
// bf16 weights: rows of width kQwen38Dim or kQwen38QDim through Bf16WarpRows,
// which holds a lane's share of a row in registers, and the wider fc and down
// rows through GemvRowsFixedOrder.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "barrier.cuh"
#include "decoder_layer.cuh"
#include "gemv_core.cuh"
#include "qwen38_layer.cuh"
#include "qwen38_lm_head.cuh"
#include "qwen38_mtp.h"
#include "thinker_layer.cuh"

namespace s2mk {
namespace {

constexpr int kDim = kQwen38Dim;
constexpr int kFfn = kQwen38Ffn;
// The most fc or down rows a CTA owns, at kGdnMinCtas CTAs.
constexpr int kMaxDimRowsPerCta = (kDim + 95) / 96;

struct MlpShared {
  // fc's input, the normalized residual, then SiLU(gate) × up, bf16.
  uint4 x[kFfn / kVecElems];
  float ys[2 * kMaxRowsPerCta];
  float partials[kWarps * kMaxDimRowsPerCta];
  float red[kWarps];
};
static_assert(2 * kDim <= kFfn, "MlpShared::x holds fc's input");

union Shared {
  qwen38::AttentionShared attention;
  MlpShared mlp;
};

// out = RMSNorm(x) × (1 + w) in bf16 for a bf16 x, as RmsNorm<NormDims>
// computes it for an fp32 one.
__device__ void NormBf16(const __nv_bfloat16* x, const __nv_bfloat16* w,
                         float eps, __nv_bfloat16* out, float* red) {
  constexpr int kPerThread = kDim / kThreads;
  float v[kPerThread];
  float squares = 0.f;
#pragma unroll
  for (int j = 0; j < kPerThread; ++j) {
    v[j] = __bfloat162float(x[threadIdx.x + j * kThreads]);
    squares += v[j] * v[j];
  }
  const float inv = rsqrtf(BlockSum(squares, red) / kDim + eps);
#pragma unroll
  for (int j = 0; j < kPerThread; ++j) {
    const int i = threadIdx.x + j * kThreads;
    out[i] = __float2bfloat16(v[j] * inv * (1.f + __bfloat162float(w[i])));
  }
}

// ys[i] = W[begin + i] · xs for this CTA's share [begin, begin + rows) of a
// bf16 W of k columns, in a fixed order.
__device__ void WideRows(const Weight& w, int k, int begin, int rows,
                         const uint4* xs, MlpShared& sh) {
  GemvRowsFixedOrder(static_cast<const __nv_bfloat16*>(w.data), k, begin,
                     rows, xs, sh.ys, sh.partials, rows, 0);
}

template <typename Sync>
__device__ void Attention(const Qwen38MtpParams& m, int pos, Sync& sync,
                          qwen38::AttentionShared& sh) {
  const Qwen38LayerParams& p = m.layer;
  const int cta = blockIdx.x;
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(sh.xs);
  const thinker::BlockStep step{pos, thinker::SlotOf(p.kv, pos),
                                {pos, pos, pos}, m.residual, m.residual};

  // 1. RMSNorm, then this CTA's q, k, v and gate rows.
  RmsNorm<qwen38::NormDims>(m.residual, nullptr, p.input_norm, p.eps, xs,
                            nullptr, sh.red);
  __syncthreads();
  Bf16WarpRows<kDim>(static_cast<const __nv_bfloat16*>(p.wqkv.data),
                     RowBegin(kQwen38QkvRows, cta),
                     RowBegin(kQwen38QkvRows, cta + 1), sh.xs,
                     [&](int row, float y) { p.qkv[row] = y; });
  sync();

  // 2. This CTA's (head, chunk) item.
  thinker::Attend(p, step, sh);
  sync();

  // 3. The merged, gated attention output, then this CTA's O rows.
  thinker::Merge(p, sh, xs);
  __syncthreads();
  Bf16WarpRows<kQwen38QDim>(
      static_cast<const __nv_bfloat16*>(p.wo.data), RowBegin(kDim, cta),
      RowBegin(kDim, cta + 1), sh.xs, [&](int row, float y) {
        m.residual[row] = __ldcg(m.residual + row) + y;
      });
}

template <typename Sync>
__device__ void Mlp(const Qwen38MtpParams& m, Sync& sync, MlpShared& sh) {
  const Qwen38MlpParams& p = m.layer.mlp;
  const int cta = blockIdx.x;

  // 1. RMSNorm, then this CTA's (gate, up) pairs: rows 2j and 2j + 1.
  RmsNorm<qwen38::NormDims>(m.residual, nullptr, p.post_norm, p.eps,
                            reinterpret_cast<__nv_bfloat16*>(sh.x), nullptr,
                            sh.red);
  __syncthreads();
  const int begin = RowBegin(kFfn, cta);
  const int pairs = RowBegin(kFfn, cta + 1) - begin;
  Bf16WarpRows<kDim>(static_cast<const __nv_bfloat16*>(p.w13.data), 2 * begin,
                     2 * (begin + pairs), sh.x,
                     [&](int row, float y) { sh.ys[row - 2 * begin] = y; });
  __syncthreads();
  for (int i = threadIdx.x; i < pairs; i += kThreads) {
    p.act[begin + i] = __float2bfloat16(Silu(sh.ys[2 * i]) * sh.ys[2 * i + 1]);
  }
  sync();

  // 2. Every pair's SiLU(gate) × up, then this CTA's down rows.
  for (int i = threadIdx.x; i < kFfn / kVecElems; i += kThreads) {
    sh.x[i] = __ldcg(reinterpret_cast<const uint4*>(p.act) + i);
  }
  const int row0 = RowBegin(kDim, cta);
  const int rows = RowBegin(kDim, cta + 1) - row0;
  WideRows(p.w2, kFfn, row0, rows, sh.x, sh);
  for (int i = threadIdx.x; i < rows; i += kThreads) {
    m.residual[row0 + i] = __ldcg(m.residual + row0 + i) + sh.ys[i];
  }
}

__global__ void __launch_bounds__(kThreads, 1)
    Qwen38MtpKernel(const __grid_constant__ Qwen38MtpParams m) {
  __shared__ Shared sh;
  GridBarrier barrier(m.sync, m.error, m.timeout_ns, 0);
  auto sync = [&] { barrier.Sync(); };
  const int pos = *m.pos;
  const int cta = blockIdx.x;

  // fc over the two normalized inputs, side by side.
  {
    __nv_bfloat16* x = reinterpret_cast<__nv_bfloat16*>(sh.mlp.x);
    NormBf16(m.embed + int64_t{*m.token} * kDim, m.embed_norm, m.eps, x,
             sh.mlp.red);
    NormBf16(m.hidden_in, m.hidden_norm, m.eps, x + kDim, sh.mlp.red);
    const int begin = RowBegin(kDim, cta);
    const int rows = RowBegin(kDim, cta + 1) - begin;
    WideRows(m.fc, 2 * kDim, begin, rows, sh.mlp.x, sh.mlp);
    for (int i = threadIdx.x; i < rows; i += kThreads) {
      m.residual[begin + i] = sh.mlp.ys[i];
    }
  }
  sync();
  Attention(m, pos, sync, sh.attention);
  sync();
  Mlp(m, sync, sh.mlp);
  sync();

  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(sh.attention.xs);
  RmsNorm<qwen38::NormDims>(m.residual, nullptr, m.final_norm, m.eps, xs,
                            cta == 0 ? m.hidden_out : nullptr,
                            sh.attention.red);
  if (m.logits == nullptr) return;
  __syncthreads();
  qwen38::LmHeadRows<1>(
      m.lm_head, m.vocab, sh.attention.xs,
      [&](int row, int, float logit) { m.logits[row] = logit; });
}

}  // namespace

cudaError_t LaunchQwen38Mtp(const Qwen38MtpParams& params, int num_ctas,
                            cudaStream_t stream) {
  cudaError_t err =
      cudaMemsetAsync(params.sync, 0, kSyncWords * sizeof(unsigned), stream);
  if (err != cudaSuccess) return err;
  Qwen38MtpKernel<<<num_ctas, kThreads, 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace s2mk
