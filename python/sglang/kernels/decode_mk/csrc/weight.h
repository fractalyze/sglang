// Copyright 2026 Fractalyze Inc. All rights reserved.

#ifndef S2MK_CSRC_WEIGHT_H_
#define S2MK_CSRC_WEIGHT_H_

#include <cuda_bf16.h>

#include <cstdint>

namespace s2mk {

// How a projection's weight is stored.
enum class WeightFormat : int32_t {
  // Dense bf16 rows.
  kBf16 = 1,
  // Asymmetric compressed-tensors W4A16 in groups of 32, with an int4 zero
  // point per group (int4_gemv_core.cuh).
  kInt4Zp = 2,
};

// A [rows, k] projection weight as Qwen3.8's kernels read it. The format tag
// names how it is stored. A kernel reads the format its call site is written
// for: kInt4Zp for every projection the checkpoint quantizes, kBf16 for the
// linear-attention gates (b, a). Two are read by their tag: the
// linear-attention output projection, bf16 in layer 0 and int4 in the rest,
// and the LM head (qwen38_lm_head.cuh), bf16 as the checkpoint keeps it or
// quantized at load.
struct Weight {
  WeightFormat format;
  const void* data;  // kInt4Zp: int32 [rows, k / 8]; kBf16: bf16 [rows, k]
  const __nv_bfloat16* scales;  // kInt4Zp: [rows, k / 32]; kBf16: null
  // kInt4Zp: [rows, k / 256], group j's zero point in bits 4 (j % 8) of word
  // j / 8, so a row's zero points are contiguous like its values; kBf16: null.
  const uint32_t* zeros;
};

}  // namespace s2mk

#endif  // S2MK_CSRC_WEIGHT_H_
