// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// The Qwen3.8 prefill's causal attention on tensor cores, in flash-attention
// form: one query head's chunk of tokens over one span of a KV group's cache,
// q · kᵀ and p · v as mma.m16n8k16 (bf16 in, fp32 out), the softmax online.
//
//   - The CTA holds the head's kFlashRows query rows, bf16, in shared memory,
//     and streams the span's keys and values a tile of kFlashKeys at a time
//     through a ring of kFlashSlots slots: tile i's keys are item 2 i and its
//     values item 2 i + 1, item n in slot n % kFlashSlots. Each slot has a
//     "full" mbarrier, complete once every thread's cp.async of the item has
//     landed, and an "empty" one, complete once every warp has read it. A
//     warp that has read item n refills the slot item n − 1 used, so it runs
//     up to an item ahead of the slowest warp.
//   - Warp (m, j) takes rows 16 m .. 16 m + 15. For the scores it takes keys
//     8 j .. 8 j + 7 of the tile over the whole head; for the output, dims
//     64 j .. 64 j + 63 over every key of the tile, the probabilities in
//     shared memory between the two.
//   - A row's tile max and tile sum come from its four warps' through shared
//     memory, the sum in warp order, so all four keep the same running max
//     and sum, and the result does not depend on which CTA takes the span.
//     The four sync on a named barrier of their own, so the CTA's row blocks
//     do not wait for each other within a tile.
//
// The probabilities enter the p · v mma rounded to bf16; their sum is taken
// from the fp32 values.

#ifndef S2MK_CSRC_QWEN38_PREFILL_ATTENTION_CUH_
#define S2MK_CSRC_QWEN38_PREFILL_ATTENTION_CUH_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "gemv_core.cuh"
#include "int4_mma_core.cuh"
#include "qwen38_layer.h"
#include "qwen38_prefill.h"
#include "qwen38_prefill_gemm.cuh"
#include "thinker_attention.h"
#include "thinker_layer.cuh"

namespace s2mk {
namespace qwen38 {

// Query rows a CTA holds: a head's tokens of one chunk.
constexpr int kFlashRows = kQwen38PrefillChunk;
// Keys a tile stages, and the slots its keys and values stream through.
constexpr int kFlashKeys = 32;
constexpr int kFlashSlots = 3;
// 16-byte chunks of a head's 256 bf16 dims.
constexpr int kFlashChunks = kQwen38HeadDim / 8;
// Warps that share a row block: each takes a quarter of the tile's keys for
// the scores and a quarter of the dims for the output.
constexpr int kFlashSplit = 4;
constexpr int kFlashDims = kQwen38HeadDim / kFlashSplit;
// The probabilities' row stride, in bf16: 16 bytes of padding a row, so the
// eight rows an ldmatrix reads land on distinct bank quads.
constexpr int kProbStride = kFlashKeys + 8;
static_assert(kFlashRows / 16 * kFlashSplit == kWarps);
static_assert(kFlashKeys == 8 * kFlashSplit);
static_assert(kQwen38PrefillAttnSpan % kFlashKeys == 0);

struct FlashShared {
  // Rows of kFlashChunks 16-byte chunks, each at FlashChunk(row, chunk).
  uint4 query[kFlashRows][kFlashChunks];
  // A tile's keys or values, kFlashKeys rows each.
  uint4 slot[kFlashSlots][kFlashKeys][kFlashChunks];
  __nv_bfloat16 prob[kFlashRows][kProbStride];
  float tile_max[kFlashSplit][kFlashRows];
  float tile_sum[kFlashSplit][kFlashRows];
  uint64_t full[kFlashSlots];
  uint64_t empty[kFlashSlots];
};

// Where chunk `chunk` of a staged row sits: XORed with the row's low bits,
// so the eight rows an ldmatrix reads at one chunk hit different banks.
__device__ __forceinline__ int FlashChunk(int row, int chunk) {
  return chunk ^ (row % 8);
}

__device__ __forceinline__ uint32_t SharedAddress(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void LdMatrixX4(uint32_t (&r)[4], const void* p) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(SharedAddress(p)));
}

__device__ __forceinline__ void LdMatrixX4Trans(uint32_t (&r)[4],
                                                const void* p) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, "
      "[%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(SharedAddress(p)));
}

// Copies 16 bytes from src, or zeros when `copy` is false.
__device__ __forceinline__ void CpAsyncOrZero(void* dst, const void* src,
                                              bool copy) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(
                   SharedAddress(dst)),
               "l"(src), "r"(copy ? 16 : 0));
}

// Starts copying the keys or values (`base` is kv.key or kv.value) of KV
// group g at positions [first, first + keys) into `rows`, zeroing the rest of
// the tile, which a masked score still multiplies. Called by the whole CTA;
// each thread arrives on `full` once its copies land.
__device__ __forceinline__ void CopyKvTile(const PagedKv& kv,
                                           __nv_bfloat16* base, int g,
                                           int first, int keys,
                                           uint4 (*rows)[kFlashChunks],
                                           uint64_t* full) {
  for (int i = threadIdx.x; i < kFlashKeys * kFlashChunks; i += kThreads) {
    const int j = i / kFlashChunks;
    const int c = i % kFlashChunks;
    CpAsyncOrZero(&rows[j][FlashChunk(j, c)],
                  reinterpret_cast<const uint4*>(
                      thinker::KvAt(kv, base, g, first + min(j, keys - 1))) +
                      c,
                  j < keys);
  }
  CpAsyncArrive(full);
}

// Syncs the kFlashSplit warps of row block kBlock, on named barrier
// 1 + kBlock. The id is an immediate, so ptxas reserves only the barriers
// used.
template <int kBlock>
__device__ __forceinline__ void RowBlockSyncOf() {
  asm volatile("bar.sync %0, %1;" ::"n"(1 + kBlock), "n"(kFlashSplit * 32)
               : "memory");
}
__device__ __forceinline__ void RowBlockSync(int block) {
  static_assert(kFlashRows / 16 == 4);
  switch (block) {
    case 0: RowBlockSyncOf<0>(); break;
    case 1: RowBlockSyncOf<1>(); break;
    case 2: RowBlockSyncOf<2>(); break;
    default: RowBlockSyncOf<3>(); break;
  }
}

// Row r of the query block: chunk c of the head's dims, as FlashSpan reads it.
__device__ __forceinline__ uint4& FlashQueryChunk(FlashShared& sh, int r,
                                                  int c) {
  return sh.query[r][FlashChunk(r, c)];
}

// Rows 0 .. tokens − 1 of sh.query (FlashQueryChunk) attend, causally, to KV
// group g's keys at positions [begin, end): row t sits at position pos0 + t
// and sees the keys at or before it. Row t's partial (running max m, sum l,
// and kQwen38HeadDim unnormalized outputs) goes to `store`: once as
// store.Row(t, m, l), then store.Pair(t, d, l, o, o') for each pair of dims
// d, d + 1. A row that sees no key gets max −∞ and sum 0. Called by the whole
// CTA once sh.query is written; leaves it unread.
template <typename Store>
__device__ void FlashSpan(FlashShared& sh, const PagedKv& kv, int g,
                          int begin, int end, int pos0, int tokens,
                          const Store& store) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int block = warp / kFlashSplit;  // rows 16 block .. 16 block + 15
  const int split = warp % kFlashSplit;
  const float scale = rsqrtf(static_cast<float>(kQwen38HeadDim));
  // The two rows this lane's accumulators hold, as mma.m16n8 lays them out.
  const int rows[2] = {16 * block + lane / 4, 16 * block + lane / 4 + 8};

  float m[2] = {-INFINITY, -INFINITY};
  float l[2] = {0.f, 0.f};
  float acc[kFlashDims / 8][4] = {};

  const int tiles = (end - begin + kFlashKeys - 1) / kFlashKeys;
  const int items = 2 * tiles;
  // Starts copying item n, tile n / 2's keys or values, into its slot.
  auto copy = [&](int n) {
    const int tile = n / 2;
    CopyKvTile(kv, n % 2 == 0 ? kv.key : kv.value, g,
               begin + tile * kFlashKeys,
               min(kFlashKeys, end - begin - tile * kFlashKeys),
               sh.slot[n % kFlashSlots], &sh.full[n % kFlashSlots]);
  };
  // Waits for item n's copies, then returns its slot.
  auto wait = [&](int n) {
    MbarrierWait(&sh.full[n % kFlashSlots], n / kFlashSlots % 2);
    return sh.slot[n % kFlashSlots];
  };
  // The warp has read item n: once every warp has read item n − 1, its slot
  // takes item n − 1 + kFlashSlots.
  auto release = [&](int n) {
    __syncwarp();
    if (lane == 0) MbarrierArrive(&sh.empty[n % kFlashSlots]);
    const int next = n - 1 + kFlashSlots;
    if (next < items) {
      if (n > 0) {
        MbarrierWait(&sh.empty[(n - 1) % kFlashSlots],
                     (n - 1) / kFlashSlots % 2);
      }
      copy(next);
    }
  };

  if (threadIdx.x < kFlashSlots) {
    MbarrierInit(&sh.full[threadIdx.x], kThreads);
    MbarrierInit(&sh.empty[threadIdx.x], kWarps);
  }
  __syncthreads();  // the barriers, and sh.query
  for (int n = 0; n < kFlashSlots - 1 && n < items; ++n) copy(n);

  for (int tile = 0; tile < tiles; ++tile) {
    const int base = begin + tile * kFlashKeys;
    const uint4 (*keys)[kFlashChunks] = wait(2 * tile);

    // Scores of keys 8 split .. 8 split + 7 over the head's dims.
    float s[4] = {};
#pragma unroll
    for (int kk = 0; kk < kFlashChunks / 2; kk += 2) {
      uint32_t b[4];
      const int key = 8 * split + lane % 8;
      LdMatrixX4(b, &keys[key][FlashChunk(key, 2 * kk + lane / 8)]);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint32_t a[4];
        const int r = 16 * block + lane % 16;
        LdMatrixX4(a, &FlashQueryChunk(sh, r, 2 * (kk + h) + lane / 16));
        MmaBf16(s, a[0], a[1], a[2], a[3], b[2 * h], b[2 * h + 1]);
      }
    }
    release(2 * tile);
    // Masked scores to −∞, then each row's max over the warp's keys.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int t = rows[i / 2];
      const int pos = base + 8 * split + 2 * (lane % 4) + i % 2;
      const bool seen = t < tokens && pos < end && pos <= pos0 + t;
      s[i] = seen ? s[i] * scale : -INFINITY;
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float x = fmaxf(s[2 * r], s[2 * r + 1]);
      x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 1));
      x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 2));
      if (lane % 4 == 0) sh.tile_max[split][rows[r]] = x;
    }
    RowBlockSync(block);

    // The running max, the probabilities and each row's sum over the warp's
    // keys.
    float correction[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float tile_max = sh.tile_max[0][rows[r]];
#pragma unroll
      for (int j = 1; j < kFlashSplit; ++j) {
        tile_max = fmaxf(tile_max, sh.tile_max[j][rows[r]]);
      }
      const float m_new = fmaxf(m[r], tile_max);
      // A row that has seen no key yet keeps max −∞ and output 0.
      correction[r] = m[r] == -INFINITY ? 0.f : expf(m[r] - m_new);
      float p[2];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const float x = s[2 * r + i];
        p[i] = x == -INFINITY ? 0.f : expf(x - m_new);
      }
      m[r] = m_new;
      float sum = p[0] + p[1];
      sum += __shfl_xor_sync(0xffffffffu, sum, 1);
      sum += __shfl_xor_sync(0xffffffffu, sum, 2);
      if (lane % 4 == 0) sh.tile_sum[split][rows[r]] = sum;
      *reinterpret_cast<__nv_bfloat162*>(
          &sh.prob[rows[r]][8 * split + 2 * (lane % 4)]) =
          __floats2bfloat162_rn(p[0], p[1]);
    }
    RowBlockSync(block);

#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float sum = sh.tile_sum[0][rows[r]];
#pragma unroll
      for (int j = 1; j < kFlashSplit; ++j) sum += sh.tile_sum[j][rows[r]];
      l[r] = l[r] * correction[r] + sum;
    }
#pragma unroll
    for (int n = 0; n < kFlashDims / 8; ++n) {
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[n][i] *= correction[i / 2];
    }
    // Outputs of dims 64 split .. 64 split + 63 over the tile's keys.
    const uint4 (*values)[kFlashChunks] = wait(2 * tile + 1);
#pragma unroll
    for (int kk = 0; kk < kFlashKeys / 16; ++kk) {
      uint32_t a[4];
      LdMatrixX4(a, &sh.prob[16 * block + lane % 16][16 * kk + 8 * (lane / 16)]);
#pragma unroll
      for (int n = 0; n < kFlashDims / 8; n += 2) {
        uint32_t b[4];
        const int key = 16 * kk + lane % 16;
        const int chunk = kFlashDims / 8 * split + n + lane / 16;
        LdMatrixX4Trans(b, &values[key][FlashChunk(key, chunk)]);
        MmaBf16(acc[n], a[0], a[1], a[2], a[3], b[0], b[1]);
        MmaBf16(acc[n + 1], a[0], a[1], a[2], a[3], b[2], b[3]);
      }
    }
    release(2 * tile + 1);
  }
  // Every copy has landed, as every item's "full" phase completed. Once
  // every warp has read its last item, the CTA's next unit may take the
  // shared memory.
  __syncthreads();
  if (threadIdx.x < kFlashSlots) {
    MbarrierInval(&sh.full[threadIdx.x]);
    MbarrierInval(&sh.empty[threadIdx.x]);
  }

#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int t = rows[r];
    if (t >= tokens) continue;
    if (split == 0 && lane % 4 == 0) store.Row(t, m[r], l[r]);
    const int dim = kFlashDims * split + 2 * (lane % 4);
#pragma unroll
    for (int n = 0; n < kFlashDims / 8; ++n) {
      store.Pair(t, dim + 8 * n, l[r], acc[n][2 * r], acc[n][2 * r + 1]);
    }
  }
}

}  // namespace qwen38
}  // namespace s2mk

#endif  // S2MK_CSRC_QWEN38_PREFILL_ATTENTION_CUH_
