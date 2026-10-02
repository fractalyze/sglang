// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// A lone int4 GEMV, one CTA per row slice, and a lone prefill GEMM tile, a
// warp per 16 rows (int4_gemv.h).

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "gemv_core.cuh"
#include "int4_gemv.h"
#include "int4_gemv_core.cuh"
#include "int4_mma_core.cuh"

namespace s2mk {
namespace {

constexpr int kGroupsPerLane = kInt4GemvK / (32 * kInt4Group);
static_assert(kGroupsPerLane * 32 * kInt4Group == kInt4GemvK);

__global__ void __launch_bounds__(kThreads, 1)
    Int4GemvKernel(const uint4* packed, const __nv_bfloat16* scales,
                   const uint32_t* zeros, const uint4* x, float* y, int n) {
  __shared__ uint4 xs[kInt4GemvK / kVecElems];
  __shared__ float ys[kInt4GemvMaxRowsPerCta];
  for (int i = threadIdx.x; i < kInt4GemvK / kVecElems; i += kThreads) {
    xs[i] = x[i];
  }
  const int begin = RowBegin(n, blockIdx.x);
  const int rows = RowBegin(n, blockIdx.x + 1) - begin;
  if (zeros == nullptr) {
    Int4GemvRows<1, 32, kGroupsPerLane>(packed, scales, begin, rows, xs, ys);
  } else {
    Int4ZpGemvRows<1, 32, kGroupsPerLane>(packed, scales, zeros, begin, rows,
                                          xs, ys);
  }
  for (int i = threadIdx.x; i < rows; i += kThreads) y[begin + i] = ys[i];
}

constexpr int kMmaNTiles = kInt4MmaMaxTokens / kMmaTokens;

__global__ void Int4MmaKernel(const uint4* packed, const __nv_bfloat16* scales,
                              const uint32_t* zeros, const __nv_bfloat16* bf16,
                              const __nv_bfloat16* x, float* y, int n,
                              int tokens) {
  constexpr int kGroups = kInt4GemvK / kInt4Group;
  const int row0 = 16 * blockIdx.x;
  float d[kMmaNTiles][4] = {};
  if (packed == nullptr) {
    Bf16MmaRows<kMmaNTiles>(bf16, kInt4GemvK, row0, 16, x, kInt4GemvK, tokens,
                            0, kGroups, d);
  } else if (zeros == nullptr) {
    Int4MmaRows<kMmaNTiles>(packed, scales, kGroups, row0, 16, x, kInt4GemvK,
                            nullptr, tokens, 0, kGroups, d);
  } else {
    Int4ZpMmaRows<kMmaNTiles>(packed, scales, zeros, kGroups, row0, 16, x,
                              kInt4GemvK, tokens, 0, kGroups, d);
  }
  // Lane (g, t) holds rows g and g + 8, tokens 8 n + 2 t and 8 n + 2 t + 1.
  const int g = threadIdx.x / 4;
  const int t = threadIdx.x % 4;
  for (int nt = 0; nt < kMmaNTiles; ++nt) {
    for (int e = 0; e < 4; ++e) {
      const int row = row0 + g + (e >= 2 ? 8 : 0);
      const int token = nt * kMmaTokens + 2 * t + e % 2;
      if (token < tokens) y[int64_t{token} * n + row] = d[nt][e];
    }
  }
}

}  // namespace

cudaError_t LaunchInt4Mma(const int32_t* packed, const __nv_bfloat16* scales,
                          const uint32_t* zeros, const __nv_bfloat16* bf16,
                          const __nv_bfloat16* x, float* y, int n, int tokens,
                          cudaStream_t stream) {
  Int4MmaKernel<<<n / 16, 32, 0, stream>>>(
      reinterpret_cast<const uint4*>(packed), scales, zeros, bf16, x, y, n,
      tokens);
  return cudaGetLastError();
}

cudaError_t LaunchInt4Gemv(const int32_t* packed, const __nv_bfloat16* scales,
                           const uint32_t* zeros, const __nv_bfloat16* x,
                           float* y, int n, int num_ctas, cudaStream_t stream) {
  Int4GemvKernel<<<num_ctas, kThreads, 0, stream>>>(
      reinterpret_cast<const uint4*>(packed), scales, zeros,
      reinterpret_cast<const uint4*>(x), y, n);
  return cudaGetLastError();
}

}  // namespace s2mk
