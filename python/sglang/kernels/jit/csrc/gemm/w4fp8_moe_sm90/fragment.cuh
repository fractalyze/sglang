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

// FP8 register A fragments read from the w4a16_moe_sm90 weight layout.
// Architecture-neutral (sm89+): the per-warp wgmma m64k32 e4m3 register A layout
// equals the mma.m16n8k32 e4m3 A layout, which lets the fragment math be tested on
// any FP8 tensor-core GPU.
//
// A slice word holds thread (g, t)'s bf16 m16k16 columns {2t, 2t + 1, 2t + 8, 2t + 9};
// e4m3 m16k32 wants {4t, ..., 4t + 3} per 16 columns. The weights keep their order and
// the reduction axis is permuted instead: e4m3 column f of each 16-column block is
// weight column K_PERMUTE16[f] (w4fp8_moe_sm90.py), and activations are quantised in
// that same order, which leaves every dot product unchanged.

#pragma once

#include <sgl_kernel/scalar_type.hpp>

#include "../marlin/dequant.h"
#include <cstdint>
#include <cuda_fp8.h>

namespace sglang::w4fp8_moe_sm90 {

// 1024 + z as fp16; subtracting it from dequant<kU4, skip_flop>'s 1024 + q gives q - z exactly.
__device__ __forceinline__ half2 biased_zero(uint8_t z) {
  const uint32_t lane = 0x6400u | z;
  const uint32_t bits = lane | (lane << 16);
  return *reinterpret_cast<const half2*>(&bits);
}

// Two exact small integers in fp16 -> two e4m3 bytes, low half to low byte.
__device__ __forceinline__ uint32_t to_e4m3x2(half2 v) {
  return __nv_cvt_halfraw2_to_fp8x2(static_cast<__half2_raw>(v), __NV_SATFINITE, __NV_E4M3);
}

// Dequantises one w4a16_moe_sm90 slice word to q - z in e4m3. `frag[0]` is row g and
// `frag[1]` row g + 8, each four bytes in e4m3 column order 4t, ..., 4t + 3 of the
// permuted reduction axis. |q - z| <= 15 is exact in e4m3; the group scale is applied
// to the FP32 accumulator instead.
__device__ __forceinline__ void dequant_fragment(uint32_t q, half2 zero_lo, half2 zero_hi, uint32_t* frag) {
  half2 v[4];
  device::marlin::dequant<half2, host::kU4.id(), true>(static_cast<int>(q), v);
  device::marlin::dequant<half2, host::kU4.id(), true>(static_cast<int>(q >> 8), v + 2);
  // v[0], v[2]: row g at bf16 columns (2t, 2t + 1), (2t + 8, 2t + 9); v[1], v[3]: row g + 8.
  frag[0] = to_e4m3x2(__hsub2(v[0], zero_lo)) | (to_e4m3x2(__hsub2(v[2], zero_lo)) << 16);
  frag[1] = to_e4m3x2(__hsub2(v[1], zero_hi)) | (to_e4m3x2(__hsub2(v[3], zero_hi)) << 16);
}

}  // namespace sglang::w4fp8_moe_sm90
