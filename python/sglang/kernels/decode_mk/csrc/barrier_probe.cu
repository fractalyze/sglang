// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// A kernel that exercises GridBarrier alone, for the tests: in round r every
// thread of every CTA writes r + 1 into its own word of buffer r mod 2,
// passes a barrier, and checks every word of the buffer. Only one thread per
// CTA arrives, so this checks that the arrival publishes the stores of all
// the CTA's threads. Two buffers make one barrier per round enough: a CTA
// writes buffer r mod 2 again only after barrier r + 1, which every CTA
// passes only once it has finished reading round r.

#include <cuda_runtime.h>

#include "barrier.cuh"
#include "barrier.h"
#include "gemv_core.cuh"

namespace s2mk {
namespace {

static_assert(kProbeWordsPerCta == kThreads);

__global__ void __launch_bounds__(kThreads, 1)
    BarrierProbeKernel(const __grid_constant__ BarrierProbeArgs args) {
  GridBarrier barrier(args.sync, args.error, args.timeout_ns, args.step);
  const int words = gridDim.x * kProbeWordsPerCta;
  for (int r = 0; r < args.rounds; ++r) {
    int* slots = args.slots + (r % 2) * words;
    // Every warp but the arriving thread's stores late, by a delay that
    // varies with the round and the warp, so an arrival that did not wait
    // for them would publish the round before they land.
    if (threadIdx.x >= 32) __nanosleep((r * 97 + threadIdx.x / 32 * 31) % 512);
    slots[blockIdx.x * kProbeWordsPerCta + threadIdx.x] = r + 1;
    if (blockIdx.x == args.skip_cta && barrier.index() == args.skip_barrier) {
      return;
    }
    barrier.Sync();
    unsigned stale = 0;
    for (int i = threadIdx.x; i < words; i += kThreads) {
      stale += __ldcg(slots + i) != r + 1;
    }
    if (stale > 0) atomicAdd(args.mismatches, stale);
  }
}

}  // namespace

cudaError_t LaunchBarrierProbe(const BarrierProbeArgs& args, int num_ctas,
                               cudaStream_t stream) {
  cudaError_t err =
      cudaMemsetAsync(args.sync, 0, kSyncWords * sizeof(unsigned), stream);
  if (err != cudaSuccess) return err;
  void* params[] = {const_cast<BarrierProbeArgs*>(&args)};
  return cudaLaunchCooperativeKernel(
      reinterpret_cast<const void*>(BarrierProbeKernel), num_ctas, kThreads,
      params, 0, stream);
}

cudaError_t MappedDevicePointer(void* host, void** device) {
  return cudaHostGetDevicePointer(device, host, 0);
}

}  // namespace s2mk
