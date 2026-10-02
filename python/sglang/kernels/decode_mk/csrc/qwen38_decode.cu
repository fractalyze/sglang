// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// One decode step of Qwen3.8-27B (qwen38_decode.h) as a persistent launch of
// one CTA per SM: the embedding, then every layer's attention block (linear
// or full) and MLP block with a grid barrier after each phase, then the final
// norm and the LM head.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "barrier.cuh"
#include "decoder_layer.cuh"
#include "gdn_block.cuh"
#include "gemv_core.cuh"
#include "qwen38_decode.h"
#include "qwen38_layer.cuh"
#include "qwen38_lm_head.cuh"
#include "qwen38_prefetch.cuh"
#include "thinker_layer.cuh"

namespace s2mk {
namespace {

using qwen38::PrefetchNext;

constexpr int kDim = kQwen38Dim;

union Shared {
  gdn::Shared gdn;
  qwen38::Shared layer;
};

// This CTA's rows of the token's embedding, into the residual.
__device__ void Embed(const Qwen38DecodeParams& p) {
  const __nv_bfloat16* row = p.embed + int64_t{*p.token} * kDim;
  const int end = RowBegin(kDim, blockIdx.x + 1);
  for (int i = RowBegin(kDim, blockIdx.x) + threadIdx.x; i < end;
       i += kThreads) {
    p.residual[i] = __bfloat162float(row[i]);
  }
}

// Copies this CTA's rows of the residual into hidden row `row`.
__device__ void DumpHidden(const Qwen38DecodeParams& p, int row) {
  if (p.hidden == nullptr) return;
  const int end = RowBegin(kDim, blockIdx.x + 1);
  for (int i = RowBegin(kDim, blockIdx.x) + threadIdx.x; i < end;
       i += kThreads) {
    p.hidden[int64_t{row} * kDim + i] = __ldcg(p.residual + i);
  }
}

__global__ void __launch_bounds__(kThreads, 1)
    Qwen38DecodeKernel(const __grid_constant__ Qwen38DecodeParams p) {
  __shared__ Shared sh;
  GridBarrier barrier(p.sync, p.error, p.timeout_ns, 0);
  auto sync = [&] {
    PrefetchNext(p, barrier.index());
    barrier.Sync();
  };
  const int pos = *p.pos;

  Embed(p);
  sync();
  for (int layer = 0; layer < p.num_layers; ++layer) {
    DumpHidden(p, layer);
    const Qwen38MlpParams* mlp;
    if (Qwen38IsFull(layer)) {
      const Qwen38LayerParams& full = p.full[Qwen38FullIndex(layer)];
      const thinker::BlockStep step{
          pos,
          thinker::SlotOf(full.kv, pos),
          {p.positions[0], p.positions[1], p.positions[2]},
          p.residual,
          p.residual};
      qwen38::AttentionBlock(full, step, sync, sh.layer.attention);
      mlp = &full.mlp;
    } else {
      GdnParams linear = p.linear[Qwen38LinearIndex(layer)];
      linear.residual_in = p.residual;
      linear.residual = p.residual;
      gdn::GdnBlock(linear, sync, sh.gdn);
      mlp = &p.linear_mlp[Qwen38LinearIndex(layer)];
    }
    sync();
    qwen38::MlpBlock(*mlp, p.residual, p.residual, sync, sh.layer.mlp);
    sync();
  }
  DumpHidden(p, p.num_layers);

  RmsNorm<qwen38::NormDims>(
      p.residual, nullptr, p.final_norm, p.eps,
      reinterpret_cast<__nv_bfloat16*>(sh.layer.attention.xs), nullptr,
      sh.layer.attention.red);
  __syncthreads();
  qwen38::LmHeadRows<1>(
      p.lm_head, p.vocab, sh.layer.attention.xs,
      [&](int row, int, float logit) { p.logits[row] = logit; });
}

}  // namespace

cudaError_t LaunchQwen38Decode(const Qwen38DecodeParams& params, int num_ctas,
                               cudaStream_t stream) {
  cudaError_t err =
      cudaMemsetAsync(params.sync, 0, kSyncWords * sizeof(unsigned), stream);
  if (err != cudaSuccess) return err;
  Qwen38DecodeKernel<<<num_ctas, kThreads, 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace s2mk
