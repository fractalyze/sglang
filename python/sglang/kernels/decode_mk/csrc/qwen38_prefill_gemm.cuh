// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// The Qwen3.8 prefill's int4 GEMM, after Marlin (the AWQ-Marlin kernels
// SGLang runs): y[t][row] = W[row] · x[t] for every token of a prompt and
// every row of a kInt4Zp weight (weight.h), read in place, the layout the
// decode reads.
//
//   - A CTA works a unit at a time: up to kTileRows rows of one 64-token chunk
//     over a run of k-tiles, kTileGroups groups each. Over one chunk, a weight
//     with many rows takes k whole, each CTA a run of its rows; one with few
//     splits k (Qwen38GemmSplits). Over several chunks, a unit is a (row
//     tile, chunk) over k whole.
//   - A unit's k-tiles stream into shared memory through kStages cp.async
//     stages, their weight words and x. Over one chunk, each k-tile's scales
//     and zero points ride in its stage; over several, a row's for a pair
//     of k-tiles come in one copy each, beside the first (Pipeline). No
//     barrier holds the CTA in step a k-tile: a stage's mbarrier pair says
//     when its copies have landed and when every warp is done reading it, so
//     a warp waits only for the k-tile it computes and for the stage it
//     refills.
//   - Warp (m, j) takes rows 32 m .. 32 m + 31 over group j of each k-tile:
//     two m-tiles by every n-tile, so each dequantized A fragment feeds every
//     n-tile and each B fragment both m-tiles. Warp 4 m + j runs on SM
//     sub-partition j, so a unit short of kTileRows rows idles warps on every
//     sub-partition alike.
//   - A weight enters the mma as bf16 (q − z) × s, its scale applied before
//     the mma, as Marlin does, so the mma accumulates across k in its fp32
//     registers. (q − z) is exact; only the product rounds, as a bf16 copy of
//     the dequantized weight would.
//
// Sums have one order whichever CTA takes a unit: a unit's warps add up in
// group order, then a split tile's splits in split order, through a global
// workspace. The units and splits follow the shape and the tokens alone, so
// the result is the same at every CTA count that holds a tile's splits.

#ifndef S2MK_CSRC_QWEN38_PREFILL_GEMM_CUH_
#define S2MK_CSRC_QWEN38_PREFILL_GEMM_CUH_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "gemv_core.cuh"
#include "int4_gemv_core.cuh"
#include "int4_mma_core.cuh"
#include "prefill_dense.cuh"
#include "qwen38_prefill.h"

namespace s2mk {
namespace qwen38 {

constexpr int kTileRows = kQwen38GemmTileRows;
constexpr int kMWarps = kTileRows / 32;
constexpr int kTileGroups = kWarps / kMWarps;
constexpr int kTileK = kTileGroups * kInt4Group;
constexpr int kStages = 3;
static_assert(kMWarps * kTileGroups == kWarps);
// A pair of k-tiles' zero points sit in one word of a row's (8 groups a
// word), and their scales in 16 bytes of the row's.
constexpr int kPairGroups = 8;
static_assert(2 * kTileGroups == kPairGroups);
static_assert([] {
  for (const auto& gemm : kQwen38PrefillGemms) {
    if (gemm[1] % (kPairGroups * kInt4Group) != 0) return false;
  }
  return true;
}(), "every prefill GEMM's k is whole pairs of k-tiles");

// The weight words [group][row] (16 bytes each) and x [group][token][run]
// (16-byte runs) of a stage's k-tile, over a full chunk whatever the tokens.
struct Stage {
  static constexpr int kX = kTileGroups * kTileRows * 16;
  static constexpr int kBytes =
      kX + kTileGroups * prefill::kMaxTokens * prefill::kGroupRuns * 16;
};

// The scales [row][group] of kGroups groups and each row's zero-point word,
// copied together: a k-tile's (kTileGroups) or a pair of k-tiles'
// (kPairGroups).
template <int kGroups>
struct ScaleBlock {
  static constexpr int kZeros = kTileRows * kGroups * 2;
  static constexpr int kBytes = kZeros + kTileRows * 4;
};

// The fold of a unit's warps: its rows at prefill::kPartStride floats a row.
constexpr int kFoldBytes = kTileRows * prefill::kPartStride * sizeof(float);

// Where a unit's scales go. A k-tile's ride in its stage, after x. Pair p's
// take slot p % 2 of a side ring past the stages, copied with the first of
// its k-tiles the unit computes, in half the copy instructions. The stages'
// mbarriers follow, clear of the fold: kStages "full", then kStages
// "empty".
template <int kScaleGroups>
struct Pipeline {
  static_assert(kScaleGroups == kTileGroups || kScaleGroups == kPairGroups);
  static constexpr bool kInStage = kScaleGroups == kTileGroups;
  using Scales = ScaleBlock<kScaleGroups>;
  static constexpr int kStageBytes =
      Stage::kBytes + (kInStage ? Scales::kBytes : 0);
  static constexpr int kRingOffset = kStages * kStageBytes;
  static constexpr int kBarrierOffset =
      kRingOffset + (kInStage ? 0 : 2 * Scales::kBytes);
  static constexpr int kBytes = kBarrierOffset + 2 * kStages * 8;
  static_assert(kBarrierOffset % 8 == 0);
  static_assert(kFoldBytes <= kRingOffset);

  // Whether a unit from k-tile kt_begin copies k-tile kt's scales with it.
  __device__ static bool CopiesScales(int kt, int kt_begin) {
    return kInStage || kt % 2 == 0 || kt == kt_begin;
  }

  // K-tile kt's scale block, its stage `stage`.
  __device__ static char* ScalesOf(char* smem, int stage, int kt) {
    if constexpr (kInStage) {
      return smem + stage * kStageBytes + Stage::kBytes;
    } else {
      return smem + kRingOffset + kt / 2 % 2 * Scales::kBytes;
    }
  }
};

// Shared memory a GEMM takes, whichever pipeline it runs.
constexpr int kGemmSmemBytes = Pipeline<kTileGroups>::kBytes >
                                       Pipeline<kPairGroups>::kBytes
                                   ? Pipeline<kTileGroups>::kBytes
                                   : Pipeline<kPairGroups>::kBytes;

__device__ __forceinline__ void CpAsync(void* dst, const void* src,
                                        int bytes) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  switch (bytes) {
    case 16:
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s),
                   "l"(src));
      break;
    case 8:
      asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(s),
                   "l"(src));
      break;
    default:
      asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(s),
                   "l"(src));
      break;
  }
}

__device__ __forceinline__ void CpAsyncCommit() {
  asm volatile("cp.async.commit_group;");
}

template <int kPending>
__device__ __forceinline__ void CpAsyncWait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(kPending) : "memory");
}

__device__ __forceinline__ unsigned MbarrierAddress(const uint64_t* bar) {
  return static_cast<unsigned>(__cvta_generic_to_shared(bar));
}

__device__ __forceinline__ void MbarrierInit(uint64_t* bar, int count) {
  const unsigned b = MbarrierAddress(bar);
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(b), "r"(count)
               : "memory");
}

__device__ __forceinline__ void MbarrierInval(uint64_t* bar) {
  const unsigned b = MbarrierAddress(bar);
  asm volatile("mbarrier.inval.shared.b64 [%0];" ::"r"(b) : "memory");
}

__device__ __forceinline__ void MbarrierArrive(uint64_t* bar) {
  const unsigned b = MbarrierAddress(bar);
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];" ::"r"(b) : "memory");
}

// Arrives on `bar` once every cp.async this thread issued before has landed.
// The arrival is one of the barrier's initial count.
__device__ __forceinline__ void CpAsyncArrive(uint64_t* bar) {
  const unsigned b = MbarrierAddress(bar);
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" ::"r"(b)
               : "memory");
}

// Waits for the phase of `bar` with parity `parity` to complete.
__device__ __forceinline__ void MbarrierWait(uint64_t* bar, unsigned parity) {
  const unsigned b = MbarrierAddress(bar);
  asm volatile(
      "{\n"
      ".reg .pred done;\n"
      "wait%=:\n"
      "mbarrier.try_wait.parity.shared.b64 done, [%0], %1;\n"
      "@!done bra wait%=;\n"
      "}" ::"r"(b),
      "r"(parity)
      : "memory");
}

// A kInt4Zp weight of k columns as the GEMM reads it.
struct GemmWeight {
  const uint4* packed;  // [rows, k / 32]: a group's 32 values a vector
  const __nv_bfloat16* scales;  // [rows, k / 32]
  const uint32_t* zeros;  // [rows, k / 256]
};

__device__ __forceinline__ GemmWeight GemmWeightOf(const Weight& w) {
  return {static_cast<const uint4*>(w.data), w.scales, w.zeros};
}

// Starts copying k-tile kt of the tile's `rows` rows from row0 into `stage`,
// and, when `scales` is not null, the scale block of kScaleGroups groups
// that holds kt's into it. Called by the whole CTA; each thread arrives on
// `full` once its copies land.
template <int kScaleGroups>
__device__ __forceinline__ void CopyStage(const GemmWeight& w, int k, int row0,
                                          int rows, const __nv_bfloat16* x,
                                          int tokens, int kt, char* stage,
                                          char* scales, uint64_t* full) {
  const int groups = k / kInt4Group;
  const int group0 = kt * kTileGroups;
  auto* weights = reinterpret_cast<uint4*>(stage);
  for (int i = threadIdx.x; i < rows * kTileGroups; i += kThreads) {
    const int row = i / kTileGroups;
    const int j = i % kTileGroups;
    CpAsync(weights + j * kTileRows + row,
            w.packed + int64_t{row0 + row} * groups + group0 + j, 16);
  }
  auto* xs = reinterpret_cast<uint4*>(stage + Stage::kX);
  constexpr int kRuns = kTileGroups * prefill::kGroupRuns;
  for (int i = threadIdx.x; i < tokens * kRuns; i += kThreads) {
    const int t = i / kRuns;
    const int j = i % kRuns / prefill::kGroupRuns;
    const int run = i % prefill::kGroupRuns;
    CpAsync(xs + (j * prefill::kMaxTokens + t) * prefill::kGroupRuns + run,
            reinterpret_cast<const uint4*>(x + int64_t{t} * k) +
                (group0 + j) * prefill::kGroupRuns + run,
            16);
  }
  if (scales != nullptr) {
    const int scale_group0 = group0 / kScaleGroups * kScaleGroups;
    for (int row = threadIdx.x; row < rows; row += kThreads) {
      CpAsync(scales + row * kScaleGroups * 2,
              w.scales + int64_t{row0 + row} * groups + scale_group0,
              kScaleGroups * 2);
      CpAsync(scales + ScaleBlock<kScaleGroups>::kZeros + row * 4,
              w.zeros + int64_t{row0 + row} * (groups / kPairGroups) +
                  scale_group0 / kPairGroups,
              4);
    }
  }
  CpAsyncArrive(full);
}

// The bf16 pair (q − z) × s of values 2b and 2b + 1 of `word`, for
// zero_pair = Int4ZeroPair(z) and scale pair (s, s).
__device__ __forceinline__ uint32_t Dequant(uint32_t word, int b,
                                            uint32_t zero_pair,
                                            __nv_bfloat162 scale) {
  const uint32_t q = Int4ZpPair(word, b, zero_pair);
  const __nv_bfloat162 v =
      __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&q), scale);
  return *reinterpret_cast<const uint32_t*>(&v);
}

// The warp's mmas over its group of the k-tile in `stage`, its scales and
// zero points in the scale block `block`: d[m][n] += its rows of m-tile m
// times n-tile n's tokens. Lane (g, t) feeds elements 8t .. 8t + 7 of the
// group, as int4_mma_core.cuh pairs them.
template <int kNTiles, int kScaleGroups>
__device__ __forceinline__ void ComputeStage(const char* stage,
                                             const char* block, int kt,
                                             float (&d)[2][kNTiles][4]) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int g = lane / 4;
  const int t = lane % 4;
  const int j = warp % kTileGroups;
  const int rows = warp / kTileGroups * 32;
  const auto* words = reinterpret_cast<const uint32_t*>(stage);
  const auto* scales = reinterpret_cast<const __nv_bfloat16*>(block);
  const auto* zeros = reinterpret_cast<const uint32_t*>(
      block + ScaleBlock<kScaleGroups>::kZeros);
  // The group's place in the block, its scale's, and in the zero-point word.
  const int group = kt * kTileGroups % kScaleGroups + j;
  const int nibble = 4 * (kt * kTileGroups % kPairGroups + j);
  // a[m][step]: the A fragment of m-tile m at mma step `step`.
  uint32_t a[2][2][4];
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    const int ra = rows + 16 * m + g;
    const int rb = ra + 8;
    const uint32_t wa = words[(j * kTileRows + ra) * 4 + t];
    const uint32_t wb = words[(j * kTileRows + rb) * 4 + t];
    const uint32_t za = Int4ZeroPair((zeros[ra] >> nibble) & 0xFu);
    const uint32_t zb = Int4ZeroPair((zeros[rb] >> nibble) & 0xFu);
    const __nv_bfloat162 sa =
        __bfloat162bfloat162(scales[ra * kScaleGroups + group]);
    const __nv_bfloat162 sb =
        __bfloat162bfloat162(scales[rb * kScaleGroups + group]);
#pragma unroll
    for (int step = 0; step < 2; ++step) {
      a[m][step][0] = Dequant(wa, 2 * step, za, sa);
      a[m][step][1] = Dequant(wb, 2 * step, zb, sb);
      a[m][step][2] = Dequant(wa, 2 * step + 1, za, sa);
      a[m][step][3] = Dequant(wb, 2 * step + 1, zb, sb);
    }
  }
  const auto* xs = reinterpret_cast<const uint4*>(stage + Stage::kX) +
                   j * prefill::kMaxTokens * prefill::kGroupRuns;
#pragma unroll
  for (int n = 0; n < kNTiles; ++n) {
    const uint4 xv = xs[(n * kMmaTokens + g) * prefill::kGroupRuns + t];
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      MmaBf16(d[m][n], a[m][0][0], a[m][0][1], a[m][0][2], a[m][0][3], xv.x,
              xv.y);
      MmaBf16(d[m][n], a[m][1][0], a[m][1][1], a[m][1][2], a[m][1][3], xv.z,
              xv.w);
    }
  }
}

// The unit's sum over k-tiles [kt_begin, kt_end) of the tile's `rows` rows
// from row0 (a multiple of 32, at most kTileRows), summed across the warps in
// group order into fold[row][token] (prefill::kPartStride floats a row). The
// warps past `rows` copy and release stages and compute nothing, which leaves
// their sub-partitions' tensor pipes to the rest. `smem` holds
// kGemmSmemBytes.
//
// K-tile kt_begin + u takes stage u % kStages in the stage's phase
// u / kStages. The stage's "full" barrier completes the phase once every
// thread's copies of the k-tile have landed, its "empty" barrier once every
// warp has read them. A warp refills the stage k-tile kt − 1 used after
// computing kt, so it runs up to a k-tile ahead of the slowest warp.
//
// Scales of kScaleGroups groups ride on the "full" barrier of the k-tile
// they are copied with (Pipeline). In the side ring, the copy of k-tile 2q
// into pair q's slot follows the "empty" wait on k-tile 2q − kStages, so it
// overwrites pair q − 2 only once every warp is past its last k-tile,
// 2q − 3, while kStages is at most 3.
static_assert(kStages <= 3);
template <int kNTiles, int kScaleGroups>
__device__ void UnitSum(const GemmWeight& w, int k, int row0, int rows,
                        const __nv_bfloat16* x, int tokens, int kt_begin,
                        int kt_end, char* smem) {
  using Pipe = Pipeline<kScaleGroups>;
  constexpr int kStageBytes = Pipe::kStageBytes;
  auto* full = reinterpret_cast<uint64_t*>(smem + Pipe::kBarrierOffset);
  uint64_t* empty = full + kStages;
  if (threadIdx.x < kStages) {
    MbarrierInit(full + threadIdx.x, kThreads);
    MbarrierInit(empty + threadIdx.x, kWarps);
  }
  __syncthreads();
  const bool live = threadIdx.x / 32 / kTileGroups * 32 < rows;
  // The scale block k-tile kt copies into from its stage, or null.
  auto copied_scales = [&](int kt, int stage) -> char* {
    return Pipe::CopiesScales(kt, kt_begin) ? Pipe::ScalesOf(smem, stage, kt)
                                            : nullptr;
  };
  float d[2][kNTiles][4] = {};
#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    const int kt = kt_begin + s;
    if (kt < kt_end) {
      CopyStage<kScaleGroups>(w, k, row0, rows, x, tokens, kt,
                              smem + s * kStageBytes, copied_scales(kt, s),
                              full + s);
    }
  }
  for (int kt = kt_begin; kt < kt_end; ++kt) {
    const int use = kt - kt_begin;
    const int stage = use % kStages;
    MbarrierWait(full + stage, use / kStages % 2);
    if (live) {
      ComputeStage<kNTiles, kScaleGroups>(smem + stage * kStageBytes,
                                          Pipe::ScalesOf(smem, stage, kt), kt,
                                          d);
    }
    __syncwarp();
    if (threadIdx.x % 32 == 0) MbarrierArrive(empty + stage);
    // The stage k-tile kt − 1 used takes k-tile kt + kStages − 1 once every
    // warp has read it.
    const int next = kt + kStages - 1;
    if (next < kt_end) {
      const int refill = (use + kStages - 1) % kStages;
      if (use > 0) MbarrierWait(empty + refill, (use - 1) / kStages % 2);
      CopyStage<kScaleGroups>(w, k, row0, rows, x, tokens, next,
                              smem + refill * kStageBytes,
                              copied_scales(next, refill), full + refill);
    }
  }
  // Every copy has landed, as every k-tile's "full" phase completed. Once
  // every warp has read its last stage, the fold may overwrite the stages.
  __syncthreads();
  if (threadIdx.x < kStages) {
    MbarrierInval(full + threadIdx.x);
    MbarrierInval(empty + threadIdx.x);
  }
  const int warp = threadIdx.x / 32;
  float* fold = reinterpret_cast<float*>(smem) +
                warp / kTileGroups * 32 * prefill::kPartStride;
  for (int j = 0; j < kTileGroups; ++j) {
    if (warp % kTileGroups == j) {
      if (j == 0) {
        prefill::StoreFragments<false, 2, kNTiles>(d, fold);
      } else {
        prefill::StoreFragments<true, 2, kNTiles>(d, fold);
      }
    }
    __syncthreads();
  }
}

// The CTAs of one tile's splits meet: each arrives on the tile's count
// (counter[0]); the last zeroes it and bumps the generation (counter[1]),
// which the others wait on. Arrivals release, and the wait acquires, every
// thread's stores before the meeting, as GridBarrier's do (barrier.cuh). All
// of a tile's splits run in one round, every CTA resident, so the meeting
// completes.
__device__ __forceinline__ void MeetSplits(unsigned* counter, int splits) {
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned* count = counter;
    unsigned* generation = counter + 1;
    unsigned seen;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                 : "=r"(seen)
                 : "l"(generation)
                 : "memory");
    unsigned arrived;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                 : "=r"(arrived)
                 : "l"(count)
                 : "memory");
    if (arrived == static_cast<unsigned>(splits - 1)) {
      *count = 0;
      asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(generation),
                   "r"(seen + 1)
                   : "memory");
    } else {
      unsigned now;
      do {
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                     : "=r"(now)
                     : "l"(generation)
                     : "memory");
      } while (now == seen);
    }
  }
  __syncthreads();
}

// Hands the first `rows` rows of a unit's fold, row0 on, to store as Emit
// names, its `tokens` tokens as tokens t0 on.
template <typename Emit, typename Store>
__device__ __forceinline__ void EmitFold(const float* fold, int row0, int rows,
                                         int t0, int tokens,
                                         const Store& store) {
  constexpr int kEmit = Emit::kRows;
  const int lines = rows / kEmit;
  for (int i = threadIdx.x; i < lines * tokens; i += kThreads) {
    const int t = i / lines;
    const int row = i % lines * kEmit;
    if constexpr (kEmit == 1) {
      store(row0 + row, t0 + t, fold[row * prefill::kPartStride + t]);
    } else {
      store(row0 + row, t0 + t, fold[row * prefill::kPartStride + t],
            fold[(row + 1) * prefill::kPartStride + t]);
    }
  }
  __syncthreads();
}

// y[t][row] = W[row] · x[t] for every row of n and token, the n × k kInt4Zp
// weight w over bf16 x ([tokens, k]), handed to store as Emit names
// (prefill_dense.cuh). n is a multiple of kTileRows and k of 2 kTileK, and
// gridDim.x at least Qwen38GemmSplits(n, k).
//
// With k whole (Qwen38GemmSplits(n, k) = 1), CTA c takes an even share of
// the 32-row blocks, kTileRows rows a unit. Otherwise every kTileRows-row
// tile splits k: a round takes gridDim.x / splits whole tiles, CTA c split
// c % splits of its tile, so a tile's splits run together. Each split
// publishes its partial to `partials`; once a tile's splits meet, each folds
// a share of the tile's rows over every split, in split order. `counters`
// holds two words a tile, zero before the first launch, as each meeting
// leaves the count. `smem` holds kGemmSmemBytes.
template <int kNTiles, typename Emit, typename Store>
__device__ void GemmRows(const GemmWeight& w, int k, int n,
                         const __nv_bfloat16* x, int tokens, char* smem,
                         float* partials, unsigned* counters,
                         const Store& store) {
  constexpr int kEmit = Emit::kRows;
  constexpr int kLines = kTileRows / kEmit;
  constexpr int kPartFloats = kTileRows * prefill::kMaxTokens;
  const int splits = Qwen38GemmSplits(n, k);
  const int k_tiles = k / kTileK;
  const float* fold = reinterpret_cast<const float*>(smem);
  auto emit = [&](int row0, int rows) {
    EmitFold<Emit>(fold, row0, rows, 0, tokens, store);
  };
  if (splits == 1) {
    // CTA c takes 32-row blocks [c B / G, (c + 1) B / G) over the whole of
    // k, kTileRows rows a unit.
    constexpr int kBlocksPerTile = kTileRows / 32;
    const int blocks = n / 32;
    const int end = (blockIdx.x + 1) * blocks / gridDim.x;
    for (int block = blockIdx.x * blocks / gridDim.x; block < end;
         block += kBlocksPerTile) {
      const int rows = min(kBlocksPerTile, end - block) * 32;
      UnitSum<kNTiles, kTileGroups>(w, k, block * 32, rows, x, tokens, 0,
                                    k_tiles, smem);
      emit(block * 32, rows);
    }
    return;
  }
  const int tiles = n / kTileRows;
  const int per_round = gridDim.x / splits;
  const int split = blockIdx.x % splits;
  if (blockIdx.x >= per_round * splits) return;
  for (int tile = blockIdx.x / splits; tile < tiles; tile += per_round) {
    const int row0 = tile * kTileRows;
    UnitSum<kNTiles, kTileGroups>(w, k, row0, kTileRows, x, tokens,
                                  split * k_tiles / splits,
                                  (split + 1) * k_tiles / splits, smem);
    // The split's partial, [token][row].
    const float* first = partials + int64_t{tile} * splits * kPartFloats;
    float* own = partials + int64_t{tile * splits + split} * kPartFloats;
    for (int i = threadIdx.x; i < kTileRows * tokens; i += kThreads) {
      const int row = i % kTileRows;
      const int t = i / kTileRows;
      own[t * kTileRows + row] = fold[row * prefill::kPartStride + t];
    }
    MeetSplits(counters + 2 * tile, splits);
    // This split's share of the tile's lines, each the sum over the splits
    // in split order.
    const int line0 = split * kLines / splits;
    const int lines = (split + 1) * kLines / splits - line0;
    auto sum = [&](int row, int t) {
      float loaded[kQwen38GemmMaxSplits];
#pragma unroll
      for (int s = 0; s < kQwen38GemmMaxSplits; ++s) {
        if (s < splits) {
          loaded[s] = __ldcg(first + s * kPartFloats + t * kTileRows + row);
        }
      }
      float y = 0.f;
#pragma unroll
      for (int s = 0; s < kQwen38GemmMaxSplits; ++s) {
        if (s < splits) y += loaded[s];
      }
      return y;
    };
    for (int i = threadIdx.x; i < lines * tokens; i += kThreads) {
      const int t = i / lines;
      const int row = (line0 + i % lines) * kEmit;
      if constexpr (kEmit == 1) {
        store(row0 + row, t, sum(row, t));
      } else {
        store(row0 + row, t, sum(row, t), sum(row + 1, t));
      }
    }
  }
}

// UnitSum over a (kTileRows-row tile, chunk) unit's k whole, its scales
// through the side ring. Not inlined: inlined beside a chunk's UnitSum, it
// shares that one's 128 registers a thread, and that one spills more.
template <int kNTiles>
__device__ __noinline__ void RingUnitSum(const GemmWeight& w, int k, int row0,
                                         const __nv_bfloat16* x, int tokens,
                                         char* smem) {
  UnitSum<kNTiles, kPairGroups>(w, k, row0, kTileRows, x, tokens, 0,
                                k / kTileK, smem);
}

// y[t][row] = W[row] · x[t] as GemmRows, over any number of tokens. A chunk
// or fewer is GemmRows' with n-tiles enough for them. More take a unit per
// (kTileRows-row tile, chunk) over k whole, in bands of
// Qwen38GemmBandChunks(k, chunks) chunks: every tile over one band, a tile's
// chunks consecutive, then the next band. The CTAs at work together share a
// few tiles' weights through L2, and a band's x stays in L2 while every tile
// reads it, so x is read from DRAM about once and each tile once a band.
// CTA c takes units c, c + G, ... A row's sum is the unit's, whichever CTA
// takes it.
template <typename Emit = prefill::EachRow, typename Store>
__device__ void Gemm(const GemmWeight& w, int k, int n, const __nv_bfloat16* x,
                     int tokens, void* smem, float* partials,
                     unsigned* counters, const Store& store) {
  char* const shared = static_cast<char*>(smem);
  if (tokens <= kQwen38PrefillChunk) {
    prefill::ForTokens(tokens, [&]<int kNTiles>() {
      GemmRows<kNTiles, Emit>(w, k, n, x, tokens, shared, partials, counters,
                              store);
    });
    return;
  }
  const int chunks = Qwen38PrefillChunks(tokens);
  const int tiles = n / kTileRows;
  const int band_chunks = Qwen38GemmBandChunks(k, chunks);
  const int band_units = tiles * band_chunks;
  for (int unit = blockIdx.x; unit < tiles * chunks; unit += gridDim.x) {
    const int band = unit / band_units;
    const int in_band = unit % band_units;
    const int chunk0 = band * band_chunks;
    const int width = min(band_chunks, chunks - chunk0);
    const int row0 = in_band / width * kTileRows;
    const int t0 = (chunk0 + in_band % width) * kQwen38PrefillChunk;
    const int count = min(kQwen38PrefillChunk, tokens - t0);
    prefill::ForTokens(count, [&]<int kNTiles>() {
      RingUnitSum<kNTiles>(w, k, row0, x + int64_t{t0} * k, count, shared);
    });
    EmitFold<Emit>(reinterpret_cast<const float*>(shared), row0, kTileRows, t0,
                   count, store);
  }
}

}  // namespace qwen38
}  // namespace s2mk

#endif  // S2MK_CSRC_QWEN38_PREFILL_GEMM_CUH_
