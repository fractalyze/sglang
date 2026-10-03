// Multiplies one packed TILE_N x TILE_K weight stage by 8 tokens with
// mma.m16n8k16, reading and dequantising A fragments exactly as the sm90
// consumer warpgroups do. Runs on any sm80+ GPU, so the repack and fragment
// math are checked without Hopper.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include "gemm/w4a16_moe_sm90/fragment.cuh"
#include <cstdint>

namespace {

constexpr int kRows = 128;
constexpr int kDepth = 128;
constexpr int kTokens = 8;

// y[row, token] = sum_k W[row, k] * x[token, k]; one thread per consumer-thread
// slot.
__global__ void fragment_mma_kernel(const uint4 *__restrict__ words,
                                    const __nv_bfloat16 *__restrict__ scales,
                                    const uint8_t *__restrict__ zeros,
                                    const __nv_bfloat16 *__restrict__ x,
                                    float *__restrict__ y) {
  using namespace sglang::w4a16_moe_sm90;
  const int wg = threadIdx.x / 128;
  const int tid = threadIdx.x % 128;
  const int warp = tid / 32;
  const int g = tid % 32 / 4;
  const int t = tid % 4;
  const int row_lo = wg * 64 + warp * 16 + g;

  const nv_bfloat162 scale_lo = __bfloat162bfloat162(scales[row_lo]);
  const nv_bfloat162 scale_hi = __bfloat162bfloat162(scales[row_lo + 8]);
  const nv_bfloat162 zero_lo = biased_zero(zeros[row_lo]);
  const nv_bfloat162 zero_hi = biased_zero(zeros[row_lo + 8]);

  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int half = 0; half < 2; ++half) {
    const uint4 q = words[(wg * 2 + half) * 128 + tid];
    const uint32_t slices[4] = {q.x, q.y, q.z, q.w};
    for (int j = 0; j < 4; ++j) {
      uint32_t a[4];
      dequant_fragment(slices[j], zero_lo, zero_hi, scale_lo, scale_hi, a);
      const int k0 = (half * 4 + j) * 16 + 2 * t;
      const __nv_bfloat16 *xr = x + g * kDepth;
      const uint32_t b0 = *reinterpret_cast<const uint32_t *>(xr + k0);
      const uint32_t b1 = *reinterpret_cast<const uint32_t *>(xr + k0 + 8);
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
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
                  const tvm::ffi::TensorView scales,
                  const tvm::ffi::TensorView zeros,
                  const tvm::ffi::TensorView x, const tvm::ffi::TensorView y) {
  using namespace sglang::host;
  using sglang::bf16_t;
  using sglang::fp32_t;
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({kRows * kDepth / 8})
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(words);
  TensorMatcher({kRows}).with_dtype<bf16_t>().with_device(device).verify(
      scales);
  TensorMatcher({kRows}).with_dtype<uint8_t>().with_device(device).verify(
      zeros);
  TensorMatcher({kTokens, kDepth})
      .with_dtype<bf16_t>()
      .with_device(device)
      .verify(x);
  TensorMatcher({kRows, kTokens})
      .with_dtype<fp32_t>()
      .with_device(device)
      .verify(y);
  LaunchKernel(1, 256, device.unwrap())(
      fragment_mma_kernel, static_cast<const uint4 *>(words.data_ptr()),
      static_cast<const __nv_bfloat16 *>(scales.data_ptr()),
      static_cast<const uint8_t *>(zeros.data_ptr()),
      static_cast<const __nv_bfloat16 *>(x.data_ptr()),
      static_cast<float *>(y.data_ptr()));
}
