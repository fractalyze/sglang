// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// A linear-attention layer's decode step (gdn.h) as its own persistent launch
// of one CTA per SM; gdn_block.cuh has its phases.

#include <cuda_runtime.h>

#include "barrier.cuh"
#include "gdn.h"
#include "gdn_block.cuh"

namespace s2mk {
namespace {

__global__ void __launch_bounds__(kThreads, 1)
    GdnKernel(const __grid_constant__ GdnParams p) {
  __shared__ gdn::Shared sh;
  GridBarrier barrier(p.sync, p.error, p.timeout_ns, 0);
  auto sync = [&] { barrier.Sync(); };
  gdn::GdnBlock(p, sync, sh);
}

}  // namespace

cudaError_t LaunchGdn(const GdnParams& params, int num_ctas,
                      cudaStream_t stream) {
  cudaError_t err =
      cudaMemsetAsync(params.sync, 0, kSyncWords * sizeof(unsigned), stream);
  if (err != cudaSuccess) return err;
  void* args[] = {const_cast<GdnParams*>(&params)};
  return cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(GdnKernel),
                                     num_ctas, kThreads, args, 0, stream);
}

}  // namespace s2mk
