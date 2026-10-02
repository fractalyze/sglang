// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// The int4 GEMM tile for the prefills (Qwen3-Omni's thinker, Qwen3.8), on
// tensor cores: one warp accumulates D = W[16 rows] · X[tokens]^T over a run
// of W's groups, with W compressed-tensors W4A16 as int4_gemv_core.cuh lays it
// out, symmetric or with zero points, and X bf16 token rows.
//
// A CUDA-core GEMM spends a multiply-add per weight per token; at 32 tokens
// that is compute-bound, ~10x under HBM's pace. Here the tokens are the
// n dimension of mma.m16n8k16 (bf16 in, fp32 out):
//
//   - A dot product may visit k in any order A and B agree on. Within a
//     group, lane (g, t) feeds the mma elements 8t .. 8t + 7 (word t of the
//     packed group, a 16-byte run of X), so each fragment is one load.
//   - A row's four lanes load four consecutive groups, 64 contiguous bytes
//     (whole DRAM bursts), then trade words so each holds word t of all
//     four; no lane holds a copy, so deep prefetch fits in registers. A byte
//     becomes a bf16 pair 128 + q with one or, then exactly q − z with one
//     subtraction of 128 + z: 136 for a symmetric weight, whose z is 8, so
//     the zero-point tile with every z = 8 gives the symmetric tile's bits.
//   - The group's scale multiplies the fp32 mma result, row by row, so the
//     weights enter the mma as exact integers.
//
// Sums run over the groups in order, so the result is deterministic.

#ifndef S2MK_CSRC_INT4_MMA_CORE_CUH_
#define S2MK_CSRC_INT4_MMA_CORE_CUH_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "gemv_core.cuh"
#include "int4_gemv_core.cuh"

namespace s2mk {

// Tokens an n-tile of mma.m16n8k16 holds.
constexpr int kMmaTokens = 8;
// Groups a warp loads before computing any.
constexpr int kMmaPrefetch = 16;

__device__ __forceinline__ void MmaBf16(float (&d)[4], uint32_t a0,
                                        uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Values 2b and 2b + 1 of `word` as a bf16 pair (q − 8, exact).
__device__ __forceinline__ uint32_t Int4Pair(uint32_t word, int b) {
  const uint32_t byte = (word >> (8 * b)) & 0xFFu;
  const uint32_t biased =
      0x43004300u | (byte & 0xFu) | ((byte & 0xF0u) << 12);
  __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(&biased);
  v = __hsub2(v, __floats2bfloat162_rn(136.f, 136.f));
  return *reinterpret_cast<const uint32_t*>(&v);
}

// The bf16 pair (128 + z, 128 + z) for zero point z in [0, 15], exact.
__device__ __forceinline__ uint32_t Int4ZeroPair(uint32_t z) {
  return (0x4300u | z) * 0x10001u;
}

// Values 2b and 2b + 1 of `word` as a bf16 pair (q − z, exact), for
// zero_pair = Int4ZeroPair(z): Int4Pair's subtraction of 136 = 128 + 8, of
// 128 + z instead.
__device__ __forceinline__ uint32_t Int4ZpPair(uint32_t word, int b,
                                               uint32_t zero_pair) {
  const uint32_t byte = (word >> (8 * b)) & 0xFFu;
  const uint32_t biased =
      0x43004300u | (byte & 0xFu) | ((byte & 0xF0u) << 12);
  __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(&biased);
  v = __hsub2(v, *reinterpret_cast<const __nv_bfloat162*>(&zero_pair));
  return *reinterpret_cast<const uint32_t*>(&v);
}

// Zero point `group` of a row whose zero points start at `zeros` (8 a word,
// int4_gemv_core.cuh).
__device__ __forceinline__ uint32_t Int4ZeroPoint(const uint32_t* zeros,
                                                  int group) {
  return (zeros[group / 8] >> (4 * (group % 8))) & 0xFu;
}

// Lane t of each quad holds v = M[t][0..3]; it gets M[0..3][t] back.
__device__ __forceinline__ uint4 TransposeQuad(uint4 v, int t) {
  const bool low = t < 2;
  uint32_t r0 = __shfl_xor_sync(0xffffffffu, low ? v.z : v.x, 2);
  uint32_t r1 = __shfl_xor_sync(0xffffffffu, low ? v.w : v.y, 2);
  // a = M[t][0..1], M[t + 2][0..1] for t < 2; M[t − 2][2..3], M[t][2..3]
  // otherwise.
  const uint32_t a0 = low ? v.x : r0;
  const uint32_t a1 = low ? v.y : r1;
  const uint32_t a2 = low ? r0 : v.z;
  const uint32_t a3 = low ? r1 : v.w;
  const bool odd = (t & 1) != 0;
  r0 = __shfl_xor_sync(0xffffffffu, odd ? a0 : a1, 1);
  r1 = __shfl_xor_sync(0xffffffffu, odd ? a2 : a3, 1);
  return odd ? make_uint4(r0, a1, r1, a3) : make_uint4(a0, r0, a2, r1);
}

// Where a tile reads its B fragments: straight from X's token rows in global
// memory.
struct GlobalX {
  static constexpr bool kStaged = false;
};

// Int4MmaTiles, and with kZeros Int4ZpMmaTiles, whose `zeros` it reads only
// then. With a staged XSource the B fragments come from `staged` instead of
// X's rows (prefill_dense.cuh): staged.Arrive(group) before each group, then
// staged.Fragment(n, group) for each n-tile whose tokens X has.
template <int kMTiles, int kNTiles, int kDepth, bool kZeros,
          typename XSource = GlobalX>
__device__ __forceinline__ void Int4MmaTilesOf(
    const uint4* packed, const __nv_bfloat16* scales, const uint32_t* zeros,
    int groups, int row0, int m_stride, int rows, const __nv_bfloat16* x,
    int x_stride, const int* token_rows, int tokens, int group_begin,
    int group_end, float (&d)[kMTiles][kNTiles][4],
    const XSource& staged = {}) {
  static_assert(kDepth % 4 == 0, "a quad loads groups four at a time");
  constexpr int kQuads = kDepth / 4;
  const int lane = threadIdx.x % 32;
  const int g = lane / 4;
  const int t = lane % 4;
  const int quad = lane & ~3;
  const bool live_a = g < rows;
  const bool live_b = g + 8 < rows;
  int64_t row_a[kMTiles];
  int64_t row_b[kMTiles];
#pragma unroll
  for (int m = 0; m < kMTiles; ++m) {
    row_a[m] = int64_t{row0 + m * m_stride + g} * groups;
    row_b[m] = int64_t{row0 + m * m_stride + g + 8} * groups;
  }
  // Elements 8t .. 8t + 7 of each group of token 8n + g, as uint4.
  const uint4* x_rows[kNTiles];
#pragma unroll
  for (int n = 0; n < kNTiles; ++n) {
    const int j = n * kMmaTokens + g;
    x_rows[n] = nullptr;
    if (j < tokens) {
      const int r = token_rows != nullptr ? token_rows[j] : j;
      x_rows[n] = reinterpret_cast<const uint4*>(x + int64_t{r} * x_stride) + t;
    }
  }
  const uint4 zero = make_uint4(0, 0, 0, 0);
#pragma unroll 1
  for (int base = group_begin; base < group_end; base += kDepth) {
    // Every group of the batch is loaded before any is used: the batch's
    // weights stream from HBM together instead of one round trip a group.
    // Lane t holds group base + 4 i + t.
    uint4 wa_batch[kQuads][kMTiles];
    uint4 wb_batch[kQuads][kMTiles];
    float sa_batch[kQuads][kMTiles];
    float sb_batch[kQuads][kMTiles];
    // kZeros: each group's Int4ZeroPair; dead code otherwise.
    uint32_t za_batch[kQuads][kMTiles];
    uint32_t zb_batch[kQuads][kMTiles];
#pragma unroll
    for (int i = 0; i < kQuads; ++i) {
      const int group = base + 4 * i + t;
      const bool in = group < group_end;
#pragma unroll
      for (int m = 0; m < kMTiles; ++m) {
        wa_batch[i][m] =
            in && live_a ? LoadStream(packed + row_a[m] + group) : zero;
        wb_batch[i][m] =
            in && live_b ? LoadStream(packed + row_b[m] + group) : zero;
        sa_batch[i][m] =
            in && live_a ? __bfloat162float(scales[row_a[m] + group]) : 0.f;
        sb_batch[i][m] =
            in && live_b ? __bfloat162float(scales[row_b[m] + group]) : 0.f;
        if constexpr (kZeros) {
          // A row's zero points: groups / 8 words, as its scales are groups.
          za_batch[i][m] = Int4ZeroPair(
              in && live_a ? Int4ZeroPoint(zeros + row_a[m] / 8, group) : 8);
          zb_batch[i][m] = Int4ZeroPair(
              in && live_b ? Int4ZeroPoint(zeros + row_b[m] / 8, group) : 8);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < kQuads; ++i) {
      // Word u: word t of group base + 4 i + u.
      uint4 words_a[kMTiles];
      uint4 words_b[kMTiles];
#pragma unroll
      for (int m = 0; m < kMTiles; ++m) {
        words_a[m] = TransposeQuad(wa_batch[i][m], t);
        words_b[m] = TransposeQuad(wb_batch[i][m], t);
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int group = base + 4 * i + u;
        if (group >= group_end) break;
        if constexpr (XSource::kStaged) staged.Arrive(group);
        uint4 xv[kNTiles];
#pragma unroll
        for (int n = 0; n < kNTiles; ++n) {
          if constexpr (XSource::kStaged) {
            xv[n] = x_rows[n] != nullptr ? staged.Fragment(n, group) : zero;
          } else {
            xv[n] = x_rows[n] != nullptr ? __ldg(x_rows[n] + 4 * group) : zero;
          }
        }
        float part[kMTiles][kNTiles][4];
#pragma unroll
        for (int m = 0; m < kMTiles; ++m) {
          const uint32_t wa = (&words_a[m].x)[u];
          const uint32_t wb = (&words_b[m].x)[u];
#pragma unroll
          for (int n = 0; n < kNTiles; ++n) {
            part[m][n][0] = part[m][n][1] = part[m][n][2] = part[m][n][3] = 0.f;
          }
          uint32_t za = 0;
          uint32_t zb = 0;
          if constexpr (kZeros) {
            za = __shfl_sync(0xffffffffu, za_batch[i][m], quad | u);
            zb = __shfl_sync(0xffffffffu, zb_batch[i][m], quad | u);
          }
#pragma unroll
          for (int step = 0; step < 2; ++step) {
            uint32_t a0, a1, a2, a3;
            if constexpr (kZeros) {
              a0 = Int4ZpPair(wa, 2 * step, za);
              a1 = Int4ZpPair(wb, 2 * step, zb);
              a2 = Int4ZpPair(wa, 2 * step + 1, za);
              a3 = Int4ZpPair(wb, 2 * step + 1, zb);
            } else {
              a0 = Int4Pair(wa, 2 * step);
              a1 = Int4Pair(wb, 2 * step);
              a2 = Int4Pair(wa, 2 * step + 1);
              a3 = Int4Pair(wb, 2 * step + 1);
            }
#pragma unroll
            for (int n = 0; n < kNTiles; ++n) {
              MmaBf16(part[m][n], a0, a1, a2, a3,
                      step == 0 ? xv[n].x : xv[n].z,
                      step == 0 ? xv[n].y : xv[n].w);
            }
          }
          const float sa = __shfl_sync(0xffffffffu, sa_batch[i][m], quad | u);
          const float sb = __shfl_sync(0xffffffffu, sb_batch[i][m], quad | u);
#pragma unroll
          for (int n = 0; n < kNTiles; ++n) {
            d[m][n][0] = fmaf(sa, part[m][n][0], d[m][n][0]);
            d[m][n][1] = fmaf(sa, part[m][n][1], d[m][n][1]);
            d[m][n][2] = fmaf(sb, part[m][n][2], d[m][n][2]);
            d[m][n][3] = fmaf(sb, part[m][n][3], d[m][n][3]);
          }
        }
      }
    }
  }
}

// d[m][n][·] += (W rows [row0 + m m_stride, .. + rows) · X tokens^T) over
// groups [group_begin, group_end), for rows ≤ 16 and tokens ≤ 8 kNTiles, for
// a symmetric W: the kMTiles row tiles share each B fragment, and kDepth
// groups of each stream from HBM together.
//
// W is `packed` ([.., k / 8] as uint4, one group per vector) and `scales`
// ([.., k / 32]), `groups` = k / 32 a row. Token j of the call is row
// token_rows[j] of X (`x`, bf16, `x_stride` elements a row, 16-byte
// aligned), or row j when token_rows is null. d[m][n] is the warp's C
// fragment of row tile m and n-tile n: lane (g, t) holds rows g and g + 8,
// tokens 8n + 2t and 8n + 2t + 1. Called by one warp.
template <int kMTiles, int kNTiles, int kDepth>
__device__ __forceinline__ void Int4MmaTiles(
    const uint4* packed, const __nv_bfloat16* scales, int groups, int row0,
    int m_stride, int rows, const __nv_bfloat16* x, int x_stride,
    const int* token_rows, int tokens, int group_begin, int group_end,
    float (&d)[kMTiles][kNTiles][4]) {
  Int4MmaTilesOf<kMTiles, kNTiles, kDepth, false>(
      packed, scales, nullptr, groups, row0, m_stride, rows, x, x_stride,
      token_rows, tokens, group_begin, group_end, d);
}

// Int4MmaTiles for one row tile, kMmaPrefetch groups deep.
template <int kNTiles>
__device__ __forceinline__ void Int4MmaRows(
    const uint4* packed, const __nv_bfloat16* scales, int groups, int row0,
    int rows, const __nv_bfloat16* x, int x_stride, const int* token_rows,
    int tokens, int group_begin, int group_end, float (&d)[kNTiles][4]) {
  Int4MmaTiles<1, kNTiles, kMmaPrefetch>(
      packed, scales, groups, row0, 0, rows, x, x_stride, token_rows, tokens,
      group_begin, group_end,
      reinterpret_cast<float(&)[1][kNTiles][4]>(d));
}

// Int4MmaRows for an asymmetric W, whose zero points are `zeros` ([.., k /
// 256], the whole matrix's, int4_gemv_core.cuh), k / 32 a multiple of 8.
// With 8 n-tiles it streams half as deep: there a full batch's weights,
// scales and zero points beside the fragments spill registers.
// With a staged XSource, its B fragments come from `staged` (Int4MmaTilesOf).
template <int kNTiles, typename XSource = GlobalX>
__device__ __forceinline__ void Int4ZpMmaRows(
    const uint4* packed, const __nv_bfloat16* scales, const uint32_t* zeros,
    int groups, int row0, int rows, const __nv_bfloat16* x, int x_stride,
    int tokens, int group_begin, int group_end, float (&d)[kNTiles][4],
    const XSource& staged = {}) {
  constexpr int kDepth = kNTiles < 8 ? kMmaPrefetch : kMmaPrefetch / 2;
  Int4MmaTilesOf<1, kNTiles, kDepth, true>(
      packed, scales, zeros, groups, row0, 0, rows, x, x_stride, nullptr,
      tokens, group_begin, group_end,
      reinterpret_cast<float(&)[1][kNTiles][4]>(d), staged);
}

// Int4ZpMmaRows for a bf16 W ([.., k], 16-byte aligned rows): the same
// fragments, read straight from W's rows, with no scale.
template <int kNTiles>
__device__ __forceinline__ void Bf16MmaRows(const __nv_bfloat16* w, int k,
                                            int row0, int rows,
                                            const __nv_bfloat16* x,
                                            int x_stride, int tokens,
                                            int group_begin, int group_end,
                                            float (&d)[kNTiles][4]) {
  constexpr int kDepth = kMmaPrefetch / 2;
  const int lane = threadIdx.x % 32;
  const int g = lane / 4;
  const int t = lane % 4;
  // Elements 8t .. 8t + 7 of each group of rows g and g + 8, and of token
  // 8n + g, as uint4.
  const uint4* row_a =
      g < rows ? reinterpret_cast<const uint4*>(w + int64_t{row0 + g} * k) + t
               : nullptr;
  const uint4* row_b =
      g + 8 < rows
          ? reinterpret_cast<const uint4*>(w + int64_t{row0 + g + 8} * k) + t
          : nullptr;
  const uint4* x_rows[kNTiles];
#pragma unroll
  for (int n = 0; n < kNTiles; ++n) {
    const int j = n * kMmaTokens + g;
    x_rows[n] = nullptr;
    if (j < tokens) {
      x_rows[n] = reinterpret_cast<const uint4*>(x + int64_t{j} * x_stride) + t;
    }
  }
  const uint4 zero = make_uint4(0, 0, 0, 0);
#pragma unroll 1
  for (int base = group_begin; base < group_end; base += kDepth) {
    uint4 wa[kDepth];
    uint4 wb[kDepth];
#pragma unroll
    for (int i = 0; i < kDepth; ++i) {
      const int group = base + i;
      const bool in = group < group_end;
      wa[i] = in && row_a != nullptr ? LoadStream(row_a + 4 * group) : zero;
      wb[i] = in && row_b != nullptr ? LoadStream(row_b + 4 * group) : zero;
    }
#pragma unroll
    for (int i = 0; i < kDepth; ++i) {
      const int group = base + i;
      if (group >= group_end) break;
#pragma unroll
      for (int n = 0; n < kNTiles; ++n) {
        const uint4 xv =
            x_rows[n] != nullptr ? __ldg(x_rows[n] + 4 * group) : zero;
        MmaBf16(d[n], wa[i].x, wb[i].x, wa[i].y, wb[i].y, xv.x, xv.y);
        MmaBf16(d[n], wa[i].z, wb[i].z, wa[i].w, wb[i].w, xv.z, xv.w);
      }
    }
  }
}

}  // namespace s2mk

#endif  // S2MK_CSRC_INT4_MMA_CORE_CUH_
