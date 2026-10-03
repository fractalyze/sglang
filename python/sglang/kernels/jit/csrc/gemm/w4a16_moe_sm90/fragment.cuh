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

// Register A fragments of the W4A16 MoE kernel. Architecture-neutral (sm80+): the
// per-warp wgmma m64k16 register A layout equals the mma.m16n8k16 A layout, which
// lets the fragment math be tested on any tensor-core GPU.

#pragma once

#include <sgl_kernel/scalar_type.hpp>

#include "../marlin/dequant.h"
#include <cstdint>

namespace sglang::w4a16_moe_sm90 {

// 128 + z as bf16; subtracting it from dequant<kU4, skip_flop>'s 128 + q gives q - z exactly.
__device__ __forceinline__ nv_bfloat162 biased_zero(uint8_t z) {
  const uint32_t lane = 0x4300u | z;
  const uint32_t bits = lane | (lane << 16);
  return *reinterpret_cast<const nv_bfloat162*>(&bits);
}

// Dequantises one k16 slice of this thread's A fragment: rows (lo, hi) = (g, g + 8).
__device__ __forceinline__ void dequant_fragment(
    uint32_t q,
    nv_bfloat162 zero_lo,
    nv_bfloat162 zero_hi,
    nv_bfloat162 scale_lo,
    nv_bfloat162 scale_hi,
    uint32_t* frag) {
  nv_bfloat162 v[4];
  device::marlin::dequant<nv_bfloat162, host::kU4.id(), true>(static_cast<int>(q), v);
  device::marlin::dequant<nv_bfloat162, host::kU4.id(), true>(static_cast<int>(q >> 8), v + 2);
  const nv_bfloat162 out[4] = {
      __hmul2(__hsub2(v[0], zero_lo), scale_lo),
      __hmul2(__hsub2(v[1], zero_hi), scale_hi),
      __hmul2(__hsub2(v[2], zero_lo), scale_lo),
      __hmul2(__hsub2(v[3], zero_hi), scale_hi),
  };
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    frag[i] = *reinterpret_cast<const uint32_t*>(&out[i]);
  }
}

}  // namespace sglang::w4a16_moe_sm90
