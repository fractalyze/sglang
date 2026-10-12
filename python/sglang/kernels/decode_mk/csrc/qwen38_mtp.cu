// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// One step of Qwen3.8-27B's MTP head (qwen38_mtp.h) as one persistent launch
// of one CTA per SM. The layer runs the phases of qwen38_layer.cuh's blocks on
// bf16 weights: rows of width kQwen38Dim or kQwen38QDim through Bf16WarpRows,
// which holds a lane's share of a row in registers, and the wider fc and down
// rows through GemvRowsFixedOrder.
//
// The extend runs several tokens through the same phases in one launch on
// qwen38_tokens.cuh's blocks, which stream each bf16 weight once for all of
// them on tensor cores. Its fc reads the tokens' normalized inputs from L2:
// side by side in shared memory, 2 × kQwen38Dim bf16 each, five tokens'
// would outgrow it.
//
// The prompt pass (Qwen38MtpPromptParams) is a third persistent launch, over
// a chunk of prompt positions, on the prefill's tensor-core GEMM and
// per-token phases.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "barrier.cuh"
#include "decoder_layer.cuh"
#include "gemv_core.cuh"
#include "prefill_dense.cuh"
#include "qwen38_decode.h"
#include "qwen38_layer.cuh"
#include "qwen38_lm_head.cuh"
#include "qwen38_mtp.h"
#include "qwen38_prefill_per_token.cuh"
#include "qwen38_tokens.cuh"
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

// ---------------------------------------------------------------- extend

namespace tokens = qwen38::tokens;

template <int kT>
union ExtendShared {
  tokens::AttentionShared<kT> attention;
  // Also the fc rows' partials and the last token's final norm.
  tokens::MlpShared<kT> mlp;
};
static_assert(tokens::PartialsFor(2 * kDim) <= tokens::PartialsFor(kDim),
              "fc takes MlpShared's partials");

template <int kT>
__global__ void __launch_bounds__(kThreads, 1)
    Qwen38MtpExtendKernel(const __grid_constant__ Qwen38MtpExtendParams e) {
  extern __shared__ uint4 smem[];
  ExtendShared<kT>& sh = *reinterpret_cast<ExtendShared<kT>*>(smem);
  const Qwen38MtpParams& m = e.head;
  GridBarrier barrier(m.sync, m.error, m.timeout_ns, 0);
  auto sync = [&] { barrier.Sync(); };
  const int pos0 = *m.pos;
  const int cta = blockIdx.x;
  const tokens::Workspace ws{e.residual, e.qkv, e.partial_ml, e.partial_o,
                             e.act};

  // CTA t normalizes token t's two fc inputs into L2, side by side.
  if (cta < kT) {
    __nv_bfloat16* x = e.fc_in + int64_t{cta} * 2 * kDim;
    NormBf16(m.embed + int64_t{e.tokens[cta]} * kDim, m.embed_norm, m.eps, x,
             sh.mlp.red);
    NormBf16(e.hidden_in + int64_t{cta} * kDim, m.hidden_norm, m.eps,
             x + kDim, sh.mlp.red);
  }
  sync();
  tokens::Project<kT, 2 * kDim>(
      tokens::Bf16Of(m.fc), kDim,
      L2X{reinterpret_cast<const uint4*>(e.fc_in), 2 * kDim},
      sh.mlp.x.partials,
      [&](int row, int t, float y) { e.residual[t * kDim + row] = y; });
  sync();
  tokens::AttentionBlock<kT, tokens::Bf16Projections>(m.layer, ws, pos0, sync,
                                                      sh.attention);
  sync();
  tokens::MlpBlock<kT, tokens::Bf16Projections>(m.layer.mlp, ws, sync, sh.mlp);
  sync();

  RmsNorm<qwen38::NormDims>(e.residual + (kT - 1) * kDim, nullptr,
                            m.final_norm, m.eps,
                            reinterpret_cast<__nv_bfloat16*>(sh.mlp.x.xs),
                            cta == 0 ? m.hidden_out : nullptr, sh.mlp.red);
  if (m.logits == nullptr) return;
  __syncthreads();
  qwen38::LmHeadRows<1>(
      m.lm_head, m.vocab, sh.mlp.x.xs,
      [&](int row, int, float logit) { m.logits[row] = logit; });
}

template <int kT>
cudaError_t LaunchExtend(const Qwen38MtpExtendParams& params, int num_ctas,
                         cudaStream_t stream) {
  constexpr int kSmem = sizeof(ExtendShared<kT>);
  cudaError_t err = cudaFuncSetAttribute(
      Qwen38MtpExtendKernel<kT>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSmem);
  if (err != cudaSuccess) return err;
  Qwen38MtpExtendKernel<kT><<<num_ctas, kThreads, kSmem, stream>>>(params);
  return cudaGetLastError();
}

// ------------------------------------------------------------ the prompt pass

constexpr int kKvRows = 2 * kQwen38KvDim;
constexpr int kPromptSmemBytes =
    std::max(prefill::FixedDenseSmemBytes(qwen38::SlicesFor(2 * kDim)),
             prefill::FixedDenseSmemBytes(qwen38::SlicesFor(kDim)));

__global__ void __launch_bounds__(kThreads, 1)
    Qwen38MtpPromptKernel(const __grid_constant__ Qwen38MtpPromptParams m) {
  // The extend's declaration, as the two share this translation unit.
  extern __shared__ uint4 smem[];
  float* partials = reinterpret_cast<float*>(smem);
  __shared__ float red[kWarps];
  const Qwen38PrefillParams& p = m.chunk;
  GridBarrier barrier(p.sync, p.error, p.timeout_ns, 0);

  // 1. fc's two inputs, normalized, side by side.
  for (int t = blockIdx.x; t < p.tokens; t += gridDim.x) {
    __nv_bfloat16* x = m.fc_in + int64_t{t} * 2 * kDim;
    NormBf16(m.embed + int64_t{m.tokens[t]} * kDim, m.embed_norm, m.eps, x,
             red);
    NormBf16(m.hidden + int64_t{t} * kDim, m.hidden_norm, m.eps, x + kDim,
             red);
  }
  barrier.Sync();

  // 2. fc.
  prefill::FixedDense<qwen38::SlicesFor(2 * kDim)>(
      prefill::Bf16Of(m.fc), 2 * kDim, kDim, m.fc_in, p.tokens, partials,
      [&](int row, int t, float y) {
        p.residual[int64_t{t} * kDim + row] = y;
      });
  barrier.Sync();

  // 3. The layer's input norm.
  qwen38::TokenNorm(p, m.layer.input_norm, m.layer.eps, nullptr, red);
  barrier.Sync();

  // 4. The k and v rows, which follow q's in wqkv and in proj.
  const auto* kv_rows = static_cast<const __nv_bfloat16*>(m.layer.wqkv.data) +
                        int64_t{kQwen38QDim} * kDim;
  prefill::FixedDense<qwen38::SlicesFor(kDim)>(
      prefill::Bf16Tile{kv_rows}, kDim, kKvRows, p.h, p.tokens, partials,
      [&](int row, int t, float y) {
        p.proj[int64_t{t} * kQwen38QkvRows + kQwen38QDim + row] = y;
      });
  barrier.Sync();

  // 5. To the cache.
  qwen38::ChunkKv(p, m.layer, *p.pos0);
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

cudaError_t LaunchQwen38MtpExtend(const Qwen38MtpExtendParams& params,
                                  int num_ctas, cudaStream_t stream) {
  cudaError_t err = cudaMemsetAsync(params.head.sync, 0,
                                    kSyncWords * sizeof(unsigned), stream);
  if (err != cudaSuccess) return err;
  static_assert(kQwen38MaxVerifyTokens == 5);
  switch (params.num_tokens) {
    case 1:
      return LaunchExtend<1>(params, num_ctas, stream);
    case 2:
      return LaunchExtend<2>(params, num_ctas, stream);
    case 3:
      return LaunchExtend<3>(params, num_ctas, stream);
    case 4:
      return LaunchExtend<4>(params, num_ctas, stream);
    case 5:
      return LaunchExtend<5>(params, num_ctas, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

cudaError_t LaunchQwen38MtpPrompt(const Qwen38MtpPromptParams& params,
                                  int num_ctas, cudaStream_t stream) {
  cudaError_t err = cudaMemsetAsync(params.chunk.sync, 0,
                                    kSyncWords * sizeof(unsigned), stream);
  if (err != cudaSuccess) return err;
  err = cudaFuncSetAttribute(Qwen38MtpPromptKernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kPromptSmemBytes);
  if (err != cudaSuccess) return err;
  Qwen38MtpPromptKernel<<<num_ctas, kThreads, kPromptSmemBytes, stream>>>(
      params);
  return cudaGetLastError();
}

}  // namespace s2mk
