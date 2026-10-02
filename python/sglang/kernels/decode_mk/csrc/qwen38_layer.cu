// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// One token through a Qwen3.8 full-attention layer (qwen38_layer.h) as one
// persistent launch of one CTA per SM; qwen38_layer.cuh has its phases.

#include <cuda_runtime.h>

#include "barrier.cuh"
#include "qwen38_layer.cuh"
#include "qwen38_layer.h"

namespace s2mk {
namespace {

__global__ void __launch_bounds__(kThreads, 1)
    Qwen38LayerKernel(const __grid_constant__ Qwen38LayerParams p) {
  __shared__ qwen38::Shared sh;
  GridBarrier barrier(p.sync, p.error, p.timeout_ns, 0);
  auto sync = [&] { barrier.Sync(); };
  const thinker::BlockStep step{
      p.pos,
      thinker::SlotOf(p.kv, p.pos),
      {p.positions[0], p.positions[1], p.positions[2]},
      p.residual_in,
      p.hidden};
  qwen38::AttentionBlock(p, step, sync, sh.attention);
  sync();
  qwen38::MlpBlock(p.mlp, p.hidden, p.residual, sync, sh.mlp);
}

}  // namespace

cudaError_t LaunchQwen38Layer(const Qwen38LayerParams& params, int num_ctas,
                              cudaStream_t stream) {
  cudaError_t err =
      cudaMemsetAsync(params.sync, 0, kSyncWords * sizeof(unsigned), stream);
  if (err != cudaSuccess) return err;
  Qwen38LayerKernel<<<num_ctas, kThreads, 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace s2mk
