/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Drives w4fp8_moe_sm90's dequant_fragment through mma.m16n8k32 e4m3, whose
// per-warp A layout equals the wgmma m64k32 register A layout. Runs on any
// sm89+ GPU, so the K permutation is checked without Hopper.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include "gemm/w4fp8_moe_sm90/fragment.cuh"
#include <cstdint>

namespace {

constexpr int kRows = 128;
constexpr int kDepth = 128;
constexpr int kTokens = 8;

// y[row, token] = sum_k (q - z)[row, k] * x[token, k], x in K_PERMUTE16 order.
__global__ void fragment_mma_kernel(const uint4 *__restrict__ words,
                                    const uint8_t *__restrict__ zeros,
                                    const uint8_t *__restrict__ x,
                                    float *__restrict__ y) {
  using namespace sglang::w4fp8_moe_sm90;
  const int wg = threadIdx.x / 128;
  const int tid = threadIdx.x % 128;
  const int warp = tid / 32;
  const int g = tid % 32 / 4;
  const int t = tid % 4;
  const int row_lo = wg * 64 + warp * 16 + g;

  const half2 zero_lo = biased_zero(zeros[row_lo]);
  const half2 zero_hi = biased_zero(zeros[row_lo + 8]);

  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int half = 0; half < 2; ++half) {
    const uint4 q = words[(wg * 2 + half) * 128 + tid];
    const uint32_t slices[4] = {q.x, q.y, q.z, q.w};
    for (int pair = 0; pair < 2; ++pair) {
      uint32_t a[4];
      dequant_fragment(slices[2 * pair], zero_lo, zero_hi, a);
      dequant_fragment(slices[2 * pair + 1], zero_lo, zero_hi, a + 2);
      const int k0 = (half * 2 + pair) * 32 + 4 * t;
      const uint8_t *xr = x + g * kDepth;
      const uint32_t b0 = *reinterpret_cast<const uint32_t *>(xr + k0);
      const uint32_t b1 = *reinterpret_cast<const uint32_t *>(xr + k0 + 16);
      asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                   "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                   "{%0,%1,%2,%3};\n"
                   : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0),
                     "r"(b1));
    }
  }
  y[row_lo * kTokens + 2 * t] = acc[0];
  y[row_lo * kTokens + 2 * t + 1] = acc[1];
  y[(row_lo + 8) * kTokens + 2 * t] = acc[2];
  y[(row_lo + 8) * kTokens + 2 * t + 1] = acc[3];
}

} // namespace

void fragment_mma(const tvm::ffi::TensorView words,
                  const tvm::ffi::TensorView zeros,
                  const tvm::ffi::TensorView x, const tvm::ffi::TensorView y) {
  using namespace sglang::host;
  using sglang::fp32_t;
  using sglang::fp8_e4m3_t;
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({kRows * kDepth / 8})
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(words);
  TensorMatcher({kRows}).with_dtype<uint8_t>().with_device(device).verify(
      zeros);
  TensorMatcher({kTokens, kDepth})
      .with_dtype<fp8_e4m3_t>()
      .with_device(device)
      .verify(x);
  TensorMatcher({kRows, kTokens})
      .with_dtype<fp32_t>()
      .with_device(device)
      .verify(y);
  LaunchKernel(1, 256, device.unwrap())(
      fragment_mma_kernel, static_cast<const uint4 *>(words.data_ptr()),
      static_cast<const uint8_t *>(zeros.data_ptr()),
      static_cast<const uint8_t *>(x.data_ptr()),
      static_cast<float *>(y.data_ptr()));
}
