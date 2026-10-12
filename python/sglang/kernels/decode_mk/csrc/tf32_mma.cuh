// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// mma.m16n8k8 on tf32 inputs with fp32 accumulation, for small fp32 matrix
// products that need more than bf16's 8 bits of mantissa. Lane (g, t) =
// (lane / 4, lane % 4) holds, of a 16 × 8 A tile, (g, t), (g + 8, t),
// (g, t + 4) and (g + 8, t + 4); of an 8 × 8 B tile, (t, g) and (t + 4, g);
// of the 16 × 8 accumulator, (g, 2 t), (g, 2 t + 1), (g + 8, 2 t) and
// (g + 8, 2 t + 1).

#ifndef S2MK_CSRC_TF32_MMA_CUH_
#define S2MK_CSRC_TF32_MMA_CUH_

#include <cstdint>

namespace s2mk {

// x rounded to tf32 (to nearest, ties away), as the mma reads it.
__device__ __forceinline__ uint32_t Tf32(float x) {
  uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(x));
  return r;
}

// d += a · b.
__device__ __forceinline__ void MmaTf32(float (&d)[4], const uint32_t (&a)[4],
                                        uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

}  // namespace s2mk

#endif  // S2MK_CSRC_TF32_MMA_CUH_
