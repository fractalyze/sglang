// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// The per-token phases of a prefill launch's full-attention layer that the
// Qwen3.8 prefill (qwen38_prefill.cu) and the MTP head's prompt pass
// (qwen38_mtp.cu) share: the residual's RMSNorm into h, and the keys' QK-norm
// and M-RoPE with the values to the paged cache. Both run on a
// Qwen38PrefillParams' token fields: tokens, positions, residual, h and proj.

#ifndef S2MK_CSRC_QWEN38_PREFILL_PER_TOKEN_CUH_
#define S2MK_CSRC_QWEN38_PREFILL_PER_TOKEN_CUH_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "decoder_layer.cuh"
#include "gemv_core.cuh"
#include "qwen38_layer.cuh"
#include "qwen38_layer.h"
#include "qwen38_prefill.h"
#include "thinker_layer.cuh"

namespace s2mk {
namespace qwen38 {

constexpr int kRotHalf = kQwen38Rotary / 2;

__device__ __forceinline__ int GlobalWarp() {
  return blockIdx.x * kWarps + threadIdx.x / 32;
}

__device__ __forceinline__ int GridWarps() { return gridDim.x * kWarps; }

// Per token, a CTA each: the residual to dump (when set), then RMSNorm ×
// (1 + norm) into h, as the decode's RmsNorm<NormDims>.
__device__ inline void TokenNorm(const Qwen38PrefillParams& p,
                                 const __nv_bfloat16* norm, float eps,
                                 float* dump, float* red) {
  constexpr int kPerThread = kDim / kThreads;
  static_assert(kPerThread * kThreads == kDim);
  for (int t = blockIdx.x; t < p.tokens; t += gridDim.x) {
    const float* row = p.residual + int64_t{t} * kDim;
    float v[kPerThread];
    float squares = 0.f;
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
      const int i = threadIdx.x + j * kThreads;
      v[j] = __ldcg(row + i);
      squares += v[j] * v[j];
      if (dump != nullptr) dump[int64_t{t} * kDim + i] = v[j];
    }
    const float inv = rsqrtf(BlockSum(squares, red) / kDim + eps);
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
      const int i = threadIdx.x + j * kThreads;
      p.h[int64_t{t} * kDim + i] = __float2bfloat16(
          v[j] * inv * NormWeight<NormDims>(norm[i]));
    }
  }
}

// Lane l holds dims 8 l .. 8 l + 7 of a 256-dim head. RoPE turns the first
// kQwen38Rotary dims, pairing dim i < 32 (lanes 0 .. 3) with dim i + 32
// (lanes 4 .. 7, the same register).

constexpr int kDimsPerLane = kQwen38HeadDim / 32;

__device__ __forceinline__ void LoadLane(const float* src, int lane,
                                         float (&x)[kDimsPerLane]) {
  const float4 a = __ldcg(reinterpret_cast<const float4*>(src) + 2 * lane);
  const float4 b = __ldcg(reinterpret_cast<const float4*>(src) + 2 * lane + 1);
  x[0] = a.x, x[1] = a.y, x[2] = a.z, x[3] = a.w;
  x[4] = b.x, x[5] = b.y, x[6] = b.z, x[7] = b.w;
}

// QK-norm (1 + norm) and partial interleaved M-RoPE of token t's head, as the
// decode's LoadHead. Every lane of the warp calls it.
__device__ inline void NormRope(const Qwen38PrefillParams& p,
                                const Qwen38LayerParams& f,
                                const __nv_bfloat16* norm, int t, int lane,
                                float (&x)[kDimsPerLane]) {
  float squares = 0.f;
#pragma unroll
  for (int i = 0; i < kDimsPerLane; ++i) squares += x[i] * x[i];
  const float inv = rsqrtf(WarpSum(squares) / kQwen38HeadDim + f.eps);
#pragma unroll
  for (int i = 0; i < kDimsPerLane; ++i) {
    x[i] = x[i] * inv *
           NormWeight<Qwen38AttentionDims>(norm[kDimsPerLane * lane + i]);
  }
  float other[kDimsPerLane];
#pragma unroll
  for (int i = 0; i < kDimsPerLane; ++i) {
    other[i] = __shfl_xor_sync(0xffffffffu, x[i], kRotHalf / kDimsPerLane);
  }
  if (lane >= kQwen38Rotary / kDimsPerLane) return;
  const bool first = lane < kRotHalf / kDimsPerLane;
#pragma unroll
  for (int i = 0; i < kDimsPerLane; ++i) {
    const int freq = (kDimsPerLane * lane + i) % kRotHalf;
    const int axis = thinker::MropeAxis<Qwen38AttentionDims>(freq);
    const __nv_bfloat16* cs =
        f.cos_sin + int64_t{p.positions[axis * p.tokens + t]} * kQwen38Rotary;
    const float c = __bfloat162float(cs[freq]);
    const float s = __bfloat162float(cs[kRotHalf + freq]);
    x[i] = first ? x[i] * c - other[i] * s : x[i] * c + other[i] * s;
  }
}

__device__ __forceinline__ uint4 PackBf16x8(const float (&x)[kDimsPerLane]) {
  uint4 out;
  uint32_t* words = &out.x;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __nv_bfloat162 pair = __floats2bfloat162_rn(x[2 * i], x[2 * i + 1]);
    words[i] = *reinterpret_cast<const uint32_t*>(&pair);
  }
  return out;
}

// A warp per (token, KV head): the key after QK-norm and M-RoPE, and the
// value, bf16, to the token's cache slot.
__device__ inline void ChunkKv(const Qwen38PrefillParams& p,
                               const Qwen38LayerParams& f, int pos0) {
  const int lane = threadIdx.x % 32;
  for (int item = GlobalWarp(); item < p.tokens * kQwen38KvHeads;
       item += GridWarps()) {
    const int t = item / kQwen38KvHeads;
    const int g = item % kQwen38KvHeads;
    const float* row = p.proj + int64_t{t} * kQkvRows;
    float k[kDimsPerLane];
    float v[kDimsPerLane];
    LoadLane(row + kQwen38QDim + g * kQwen38HeadDim, lane, k);
    LoadLane(row + kQwen38QDim + kQwen38KvDim + g * kQwen38HeadDim, lane, v);
    NormRope(p, f, f.k_norm, t, lane, k);
    const int64_t slot = thinker::SlotOf(f.kv, pos0 + t);
    reinterpret_cast<uint4*>(thinker::SlotAt(f.kv, f.kv.key, g, slot))[lane] =
        PackBf16x8(k);
    reinterpret_cast<uint4*>(thinker::SlotAt(f.kv, f.kv.value, g, slot))[lane] =
        PackBf16x8(v);
  }
}
}  // namespace qwen38
}  // namespace s2mk

#endif  // S2MK_CSRC_QWEN38_PREFILL_PER_TOKEN_CUH_
