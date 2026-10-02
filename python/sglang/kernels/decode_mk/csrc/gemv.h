// Copyright 2026 Fractalyze Inc. All rights reserved.

#ifndef S2MK_CSRC_GEMV_H_
#define S2MK_CSRC_GEMV_H_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace s2mk {

// One y = W x with W an n × k row-major bf16 matrix. The layout matches the
// int64 rows s2mk/gemv.py writes into the descriptor tensor.
struct GemvDesc {
  const __nv_bfloat16* w;
  const __nv_bfloat16* x;
  __nv_bfloat16* y;
  int64_t n;
  int64_t k;
};

// Dynamic shared memory for the largest k and rows-per-CTA in a sequence.
int GemvSmemBytes(int max_k, int max_rows_per_cta);

// Runs `count` device-resident descriptors in order on `num_ctas` CTAs.
cudaError_t LaunchGemv(const GemvDesc* descs, int count, int num_ctas,
                       int smem_bytes, cudaStream_t stream);

}  // namespace s2mk

#endif  // S2MK_CSRC_GEMV_H_
