// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// The dense GEMM phase the prefill megakernels share (thinker_prefill.cu,
// qwen38_prefill.cu): y[t][row] = W[row] · x[t] for every token of a chunk
// and every row of an n-row weight, on tensor cores (int4_mma_core.cuh).
//
// Each CTA takes whole 16-row blocks, DenseBegin's share. Its warps take
// (block, k slice) units; their C fragments meet in shared memory and each
// row's slices are summed in slice order, so every result has one order.
//
// FixedDense's int4 tiles over a full chunk read x from shared memory: a
// round's warps that share a slice share its x, so the CTA copies each group
// of x once (cp.async, two stages) where each warp would read it from L2.

#ifndef S2MK_CSRC_PREFILL_DENSE_CUH_
#define S2MK_CSRC_PREFILL_DENSE_CUH_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

#include "gemv_core.cuh"
#include "int4_gemv_core.cuh"
#include "int4_mma_core.cuh"

namespace s2mk {
namespace prefill {

// Tokens a prefill chunk takes at most: 8 n-tiles of mma.m16n8k16.
constexpr int kMaxTokens = 8 * kMmaTokens;
constexpr int kBlockRows = 16;
// Row stride of the warps' C fragments in shared memory: off by 8 banks a row.
constexpr int kPartStride = kMaxTokens + 8;
// A dense GEMM's per-warp partials: kWarps slices of 16 rows × 64 tokens.
constexpr int kDenseSmemFloats = kWarps * kBlockRows * kPartStride;

// The warp's C fragments d[m][n] to part[m × kBlockRows + row][token], or
// added to what is there with kAdd.
template <bool kAdd, int kMTiles, int kNTiles>
__device__ __forceinline__ void StoreFragments(
    const float (&d)[kMTiles][kNTiles][4], float* part) {
  const int lane = threadIdx.x % 32;
  const int g = lane / 4;
  const int t = lane % 4;
#pragma unroll
  for (int m = 0; m < kMTiles; ++m) {
#pragma unroll
    for (int n = 0; n < kNTiles; ++n) {
      float* r = part + (m * kBlockRows + g) * kPartStride + n * kMmaTokens +
                 2 * t;
      float2* lo = reinterpret_cast<float2*>(r);
      float2* hi = reinterpret_cast<float2*>(r + 8 * kPartStride);
      if constexpr (kAdd) {
        *lo = make_float2(lo->x + d[m][n][0], lo->y + d[m][n][1]);
        *hi = make_float2(hi->x + d[m][n][2], hi->y + d[m][n][3]);
      } else {
        *lo = make_float2(d[m][n][0], d[m][n][1]);
        *hi = make_float2(d[m][n][2], d[m][n][3]);
      }
    }
  }
}

// The first of CTA cta's rows of a dense phase of n rows: whole 16-row
// tiles, so no mma row goes unused (O's 2048 rows on 170 CTAs were 12 a
// CTA); CTAs past the last tile take none.
__device__ __forceinline__ int DenseBegin(int n, int cta) {
  return static_cast<int>(int64_t{cta} * (n / kBlockRows) / gridDim.x) *
         kBlockRows;
}

// A symmetric int4 weight's tile (Int4MmaRows).
struct Int4Tile {
  const int32_t* packed;
  const __nv_bfloat16* scales;

  template <int kNTiles>
  __device__ __forceinline__ void operator()(int k, int row0, int rows,
                                             const __nv_bfloat16* x,
                                             int tokens, int group_begin,
                                             int group_end,
                                             float (&d)[kNTiles][4]) const {
    Int4MmaRows<kNTiles>(reinterpret_cast<const uint4*>(packed), scales,
                         k / kInt4Group, row0, rows, x, k, nullptr, tokens,
                         group_begin, group_end, d);
  }
};

// An int4 weight's tile with zero points (Int4ZpMmaRows).
struct Int4ZpTile {
  const int32_t* packed;
  const __nv_bfloat16* scales;
  const uint32_t* zeros;

  template <int kNTiles, typename XSource = GlobalX>
  __device__ __forceinline__ void operator()(int k, int row0,
                                             const __nv_bfloat16* x,
                                             int tokens, int group_begin,
                                             int group_end,
                                             float (&d)[kNTiles][4],
                                             const XSource& staged = {}) const {
    Int4ZpMmaRows<kNTiles>(reinterpret_cast<const uint4*>(packed), scales,
                           zeros, k / kInt4Group, row0, kBlockRows, x, k,
                           tokens, group_begin, group_end, d, staged);
  }
};

__device__ __forceinline__ void CpAsync16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(
                   static_cast<unsigned>(__cvta_generic_to_shared(dst))),
               "l"(src));
}

// 16-byte runs of a group of x a token: lane t of an mma reads run t.
constexpr int kGroupRuns = kInt4Group * sizeof(__nv_bfloat16) / 16;

// x of a round's slices in shared memory, a group of each slice a stage, two
// stages: stage j % 2 holds [slice][token][run] of each slice's group j. A
// round's warps all take group j of their slices together, so one barrier a
// group tells every warp that group j has landed and that group j − 1's
// stage is free for group j + 1.
template <int kSlices>
struct StagedX {
  static constexpr bool kStaged = true;
  static constexpr int kStageRuns = kSlices * kMaxTokens * kGroupRuns;

  const __nv_bfloat16* x;
  int k;
  int tokens;
  int slice_groups;  // groups a slice holds
  int slice;  // the warp's
  uint4* stages;  // [2][kStageRuns]

  // Starts copying group j of every slice into stage j % 2. Called by the
  // whole CTA.
  __device__ void Copy(int j) const {
    uint4* stage = stages + j % 2 * kStageRuns;
    const int slice_runs = tokens * kGroupRuns;
    for (int i = threadIdx.x; i < kSlices * slice_runs; i += kThreads) {
      const int s = i / slice_runs;
      const int token = i % slice_runs / kGroupRuns;
      const int run = i % kGroupRuns;
      const int group = s * slice_groups + j;
      CpAsync16(stage + (s * kMaxTokens + token) * kGroupRuns + run,
                reinterpret_cast<const uint4*>(x + int64_t{token} * k) +
                    group * kGroupRuns + run);
    }
    asm volatile("cp.async.commit_group;");
  }

  // Waits until the slice's `group` is staged, then starts copying the next.
  // Called by the whole CTA, once for each group of a slice, in order.
  __device__ void Arrive(int group) const {
    const int j = group - slice * slice_groups;
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
    if (j + 1 < slice_groups) Copy(j + 1);
  }

  // The B fragment of n-tile n at the slice's `group`, as Int4MmaTilesOf
  // reads it from x: lane (g, t) gets run t of token 8n + g.
  __device__ uint4 Fragment(int n, int group) const {
    const int lane = threadIdx.x % 32;
    const int j = group - slice * slice_groups;
    return stages[j % 2 * kStageRuns +
                  (slice * kMaxTokens + n * kMmaTokens + lane / 4) *
                      kGroupRuns +
                  lane % 4];
  }
};

// Shared memory FixedDenseRows takes with `slices` slices: the round's
// partials, then StagedX's two stages.
constexpr int FixedDenseSmemBytes(int slices) {
  return (kWarps / slices) * kBlockRows * kPartStride * sizeof(float) +
         2 * slices * kMaxTokens * kGroupRuns * 16;
}

// A bf16 weight's tile (Bf16MmaRows).
struct Bf16Tile {
  const __nv_bfloat16* w;

  template <int kNTiles>
  __device__ __forceinline__ void operator()(int k, int row0,
                                             const __nv_bfloat16* x,
                                             int tokens, int group_begin,
                                             int group_end,
                                             float (&d)[kNTiles][4]) const {
    Bf16MmaRows<kNTiles>(w, k, row0, kBlockRows, x, k, tokens, group_begin,
                         group_end, d);
  }
};

// Emits each row's y on its own: store(row, t, y).
struct EachRow {
  static constexpr int kRows = 1;
};

// Emits rows 2j and 2j + 1 together, a gate-up pair: store(2j, t, y0, y1).
struct RowPairs {
  static constexpr int kRows = 2;
};

// The warp's (block, slice) unit: block's 16 rows over groups [slice ×
// groups / slices, (slice + 1) × groups / slices), its C fragments into
// partials slot `slot`.
template <int kNTiles, typename Tile>
__device__ __forceinline__ void DenseUnit(const Tile& tile, int k, int row0,
                                          int rows, const __nv_bfloat16* x,
                                          int tokens, int slice, int slices,
                                          float* partials, int slot) {
  const int groups = k / kInt4Group;
  float d[kNTiles][4];
#pragma unroll
  for (int i = 0; i < kNTiles; ++i) d[i][0] = d[i][1] = d[i][2] = d[i][3] = 0.f;
  tile.template operator()<kNTiles>(k, row0, rows, x, tokens,
                                    slice * groups / slices,
                                    (slice + 1) * groups / slices, d);
  StoreFragments<false, 1, kNTiles>(
      reinterpret_cast<const float(&)[1][kNTiles][4]>(d),
      partials + slot * kBlockRows * kPartStride);
}

// y[t][row] for this CTA's rows DenseBegin(n) of W · x[t]^T and every token,
// handed to store(row, t, y). W has n rows of k; tile computes a 16-row
// tile's fragments over a run of its groups. The CTA's blocks split their
// groups between as many slices as its warps leave, so the order of a row's
// sum changes with the grid.
template <int kNTiles, typename Tile, typename Store>
__device__ void DenseRows(const Tile& tile, int k, int n,
                          const __nv_bfloat16* x, int tokens, float* partials,
                          const Store& store) {
  const int warp = threadIdx.x / 32;
  const int begin = DenseBegin(n, blockIdx.x);
  const int rows = DenseBegin(n, blockIdx.x + 1) - begin;
  if (rows == 0) return;
  const int blocks = (rows + kBlockRows - 1) / kBlockRows;
  const int slices = blocks >= kWarps ? 1 : kWarps / blocks;
  for (int unit = warp; unit < blocks * slices; unit += kWarps) {
    const int block = unit % blocks;
    const int slice = unit / blocks;
    const int row0 = block * kBlockRows;
    DenseUnit<kNTiles>(tile, k, begin + row0, min(kBlockRows, rows - row0), x,
                       tokens, slice, slices, partials, slice * blocks + block);
  }
  __syncthreads();
  // Consecutive threads take consecutive rows: whole lines of y.
  for (int i = threadIdx.x; i < rows * tokens; i += kThreads) {
    const int j = i / rows;
    const int row = i % rows;
    const int block = row / kBlockRows;
    float sum = 0.f;
    for (int slice = 0; slice < slices; ++slice) {
      sum += partials[((slice * blocks + block) * kBlockRows +
                       row % kBlockRows) * kPartStride + j];
    }
    store(begin + row, j, sum);
  }
  __syncthreads();
}

// y[t][row] for this CTA's rows DenseBegin(n) of W · x[t]^T and every token,
// handed to store as Emit names; W has n rows of k. Every 16-row block splits
// k into kSlices slices, a warp each, kWarps / kSlices blocks a round, and
// the slices add up in slice order, so a row's sum is the same at every CTA
// count. An int4 tile over 8 n-tiles reads x through StagedX, which needs
// k / 32 a multiple of kSlices, so every slice meets as many barriers; with
// fewer, a group's mmas are too few to cover its barrier, and each warp reads
// x from L2. `smem` holds FixedDenseSmemBytes(kSlices).
template <int kNTiles, int kSlices, typename Emit, typename Tile,
          typename Store>
__device__ void FixedDenseRows(const Tile& tile, int k, int n,
                               const __nv_bfloat16* x, int tokens,
                               float* smem, const Store& store) {
  static_assert(kSlices > 0 && kWarps % kSlices == 0);
  constexpr int kRoundBlocks = kWarps / kSlices;
  constexpr int kEmit = Emit::kRows;
  constexpr bool kStaged = std::is_same_v<Tile, Int4ZpTile> && kNTiles == 8;
  float* partials = smem;
  const int warp = threadIdx.x / 32;
  const int begin = DenseBegin(n, blockIdx.x);
  const int blocks = (DenseBegin(n, blockIdx.x + 1) - begin) / kBlockRows;
  const int groups = k / kInt4Group;
  for (int first = 0; first < blocks; first += kRoundBlocks) {
    const int count = min(kRoundBlocks, blocks - first);
    const bool live = warp < count * kSlices;
    const int block = warp % count;
    const int slice = warp / count;
    const int row0 = begin + (first + block) * kBlockRows;
    const int group_begin = slice * groups / kSlices;
    const int group_end = (slice + 1) * groups / kSlices;
    float d[1][kNTiles][4] = {};
    if constexpr (kStaged) {
      const StagedX<kSlices> staged{
          x, k, tokens, groups / kSlices, slice,
          reinterpret_cast<uint4*>(partials + kRoundBlocks * kBlockRows *
                                                  kPartStride)};
      staged.Copy(0);
      if (live) {
        tile.template operator()<kNTiles>(k, row0, x, tokens, group_begin,
                                          group_end, d[0], staged);
      } else {
        // An idle warp still meets every group's barrier.
        for (int group = group_begin; group < group_end; ++group) {
          staged.Arrive(group);
        }
      }
    } else if (live) {
      tile.template operator()<kNTiles>(k, row0, x, tokens, group_begin,
                                        group_end, d[0]);
    }
    float* part = partials + block * kBlockRows * kPartStride;
    for (int s = 0; s < kSlices; ++s) {
      if (live && slice == s) {
        if (s == 0) {
          StoreFragments<false, 1, kNTiles>(d, part);
        } else {
          StoreFragments<true, 1, kNTiles>(d, part);
        }
      }
      __syncthreads();
    }
    const int lines = count * kBlockRows / kEmit;
    for (int i = threadIdx.x; i < lines * tokens; i += kThreads) {
      const int j = i / lines;
      const int row = i % lines * kEmit;
      const int out = begin + first * kBlockRows + row;
      if constexpr (kEmit == 1) {
        store(out, j, partials[row * kPartStride + j]);
      } else {
        store(out, j, partials[row * kPartStride + j],
              partials[(row + 1) * kPartStride + j]);
      }
    }
    __syncthreads();
  }
}

// Runs rows<kNTiles>() with n-tiles enough for `tokens`.
template <typename Rows>
__device__ __forceinline__ void ForTokens(int tokens, const Rows& rows) {
  if (tokens <= kMmaTokens) {
    rows.template operator()<1>();
  } else if (tokens <= 2 * kMmaTokens) {
    rows.template operator()<2>();
  } else if (tokens <= 4 * kMmaTokens) {
    rows.template operator()<4>();
  } else {
    rows.template operator()<8>();
  }
}

// DenseRows with n-tiles enough for `tokens`.
template <typename Tile, typename Store>
__device__ void Dense(const Tile& tile, int k, int n, const __nv_bfloat16* x,
                      int tokens, float* partials, const Store& store) {
  if (tokens <= kMmaTokens) {
    DenseRows<1>(tile, k, n, x, tokens, partials, store);
  } else if (tokens <= 2 * kMmaTokens) {
    DenseRows<2>(tile, k, n, x, tokens, partials, store);
  } else if (tokens <= 4 * kMmaTokens) {
    DenseRows<4>(tile, k, n, x, tokens, partials, store);
  } else {
    DenseRows<8>(tile, k, n, x, tokens, partials, store);
  }
}

// FixedDenseRows with n-tiles enough for `tokens`.
template <int kSlices, typename Emit = EachRow, typename Tile, typename Store>
__device__ void FixedDense(const Tile& tile, int k, int n,
                           const __nv_bfloat16* x, int tokens, float* smem,
                           const Store& store) {
  ForTokens(tokens, [&]<int kNTiles>() {
    FixedDenseRows<kNTiles, kSlices, Emit>(tile, k, n, x, tokens, smem,
                                           store);
  });
}

}  // namespace prefill
}  // namespace s2mk

#endif  // S2MK_CSRC_PREFILL_DENSE_CUH_
