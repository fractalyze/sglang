// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// Persistent batch-1 GEMV, y = W x, over bf16 weights packed by
// s2mk/gemv.py. One CTA runs per SM and walks a list of GEMV descriptors
// back to back, so a sequence of GEMVs costs one launch. CTA c owns rows
// [c * n / C, (c + 1) * n / C) of each descriptor; gemv_core.cuh streams them.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "gemv.h"
#include "gemv_core.cuh"

namespace s2mk {
namespace {

__global__ void __launch_bounds__(kThreads, 1)
    GemvKernel(const GemvDesc* descs, int count) {
  extern __shared__ uint4 smem[];

  for (int d = 0; d < count; ++d) {
    const GemvDesc g = descs[d];
    const int row_begin = RowBegin(g.n, blockIdx.x);
    const int rows = RowBegin(g.n, blockIdx.x + 1) - row_begin;
    const int vecs_per_row = g.k / kVecElems;

    uint4* xs = smem;
    float* ys = reinterpret_cast<float*>(smem + vecs_per_row);

    // The previous descriptor's epilogue may still be reading xs and ys.
    __syncthreads();
    const uint4* x = reinterpret_cast<const uint4*>(g.x);
    for (int i = threadIdx.x; i < vecs_per_row; i += kThreads) xs[i] = x[i];

    GemvRows(g.w, g.k, row_begin, rows, xs, ys);

    for (int i = threadIdx.x; i < rows; i += kThreads) {
      g.y[row_begin + i] = __float2bfloat16(ys[i]);
    }
  }
}

}  // namespace

int GemvSmemBytes(int max_k, int max_rows_per_cta) {
  return max_k * static_cast<int>(sizeof(__nv_bfloat16)) +
         max_rows_per_cta * static_cast<int>(sizeof(float));
}

cudaError_t LaunchGemv(const GemvDesc* descs, int count, int num_ctas,
                       int smem_bytes, cudaStream_t stream) {
  cudaError_t err = cudaFuncSetAttribute(
      GemvKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  if (err != cudaSuccess) return err;
  GemvKernel<<<num_ctas, kThreads, smem_bytes, stream>>>(descs, count);
  return cudaGetLastError();
}

}  // namespace s2mk
