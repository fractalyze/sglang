// Copyright 2026 Fractalyze Inc. All rights reserved.
//
// A prompt's prefill of Qwen3.8-27B (qwen38_prefill.h) as a persistent
// launch, layer by layer, a grid barrier after each phase. Every phase runs
// over all the launch's tokens, so each layer's weights are read once for the
// whole prompt. A linear-attention layer:
//
//   L1. Per token (a CTA each): RMSNorm into h.
//   L2. The int4 in_proj rows (q, k, v, z) and the bf16 gate rows (b, a) of
//       every token; the gates become β and the log decay g.
//   L3. A warp per (token, head of q, k or v): the causal conv over the conv
//       state and the prompt, SiLU, and for q and k the L2 norm.
//   L4a. The chunked gated delta rule's intra-chunk matrices, a CTA per
//       (chunk, value head), every chunk at once, into act; the conv state
//       takes the prompt's last inputs.
//   L4b. Its state pass, a CTA per (value head, run of its state columns),
//       stepping through the prompt's chunks in order.
//   L5. A warp per (token, value head): the gated RMSNorm into attn.
//   L6. out_proj rows, added to the residual.
//
// A full-attention layer:
//
//   F1. Per token: RMSNorm into h.
//   F2. q, k, v and gate rows of every token.
//   F3. A warp per (token, KV head): the key's QK-norm and M-RoPE; key and
//       value, bf16, to the paged cache.
//   F4. A CTA per (chunk, query head, span of its keys): the chunk's
//       QK-norm and M-RoPE, then causal flash attention on tensor cores over
//       the head's KV group's keys in the span, into a partial
//       (qwen38_prefill_attention.cuh), or, when one span holds every key,
//       gated straight into attn.
//   F5. A warp per (token, query head): the partials merged, then gated;
//       nothing when F4 has gated them.
//   F6. O rows, added to the residual.
//
// Then the MLP: M1. RMSNorm. M2. gate and up pairs into SiLU(gate) × up.
// M3. down rows, added to the residual.
//
// The int4 projections are qwen38_prefill_gemm.cuh's GEMM and the bf16 ones
// FixedDense (prefill_dense.cuh), each over every token of the launch. Each
// splits k in a fixed way for its shape and the launch's tokens, so every
// output is a sum in one order, whichever CTA takes it.
//
// The per-token phases L1, F1, M1 and F3 are qwen38_prefill_per_token.cuh's,
// which the MTP head's prompt pass shares.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "barrier.cuh"
#include "decoder_layer.cuh"
#include "gdn_block.cuh"
#include "gemv_core.cuh"
#include "prefill_dense.cuh"
#include "qwen38_decode.h"
#include "qwen38_layer.cuh"
#include "qwen38_lm_head.cuh"
#include "qwen38_prefill.h"
#include "qwen38_prefill_attention.cuh"
#include "qwen38_prefill_gemm.cuh"
#include "qwen38_prefill_per_token.cuh"
#include "tf32_mma.cuh"
#include "thinker_layer.cuh"

namespace s2mk {
namespace {

constexpr int kDim = kQwen38Dim;
constexpr int kChunk = kQwen38PrefillChunk;
static_assert(kChunk == prefill::kMaxTokens);
constexpr int kFfn = kQwen38Ffn;
constexpr int kHead = kQwen38HeadDim;
constexpr int kQDim = kQwen38QDim;
constexpr int kKvDim = kQwen38KvDim;
constexpr int kQkvRows = kQwen38QkvRows;
constexpr int kGqa = kQwen38QHeads / kQwen38KvHeads;
// The output gate's rows in the fused q, k, v and gate rows.
constexpr int kGateRow = kQDim + 2 * kKvDim;
static_assert(kQkvRows <= kGdnInt4Rows, "proj holds either layer's rows");
static_assert(kQDim == kGdnValueDim, "attn holds either layer's output");

using gdn::kHeadDim;  // a linear layer's head, 128
using gdn::kHeadsPerKHead;
constexpr int kKHeads = kGdnKHeads;
constexpr int kVHeads = kGdnVHeads;
constexpr int kValueDim = kGdnValueDim;
constexpr int kConvDim = kGdnConvDim;
constexpr int kConvTaps = kGdnConvWidth;

using qwen38::ChunkKv;
using qwen38::CpAsync;
using qwen38::CpAsyncCommit;
using qwen38::CpAsyncWait;
using qwen38::GlobalWarp;
using qwen38::GridWarps;
using qwen38::kDimsPerLane;
using qwen38::LoadLane;
using qwen38::NormRope;
using qwen38::PackBf16x8;
using qwen38::TokenNorm;

// ------------------------------------------------------------ the dense phases

using qwen38::SlicesFor;

// The kInt4Zp weight w of n rows of kK over x, every token's rows to store.
template <int kK, typename Emit = prefill::EachRow, typename Store>
__device__ void Project(const Qwen38PrefillParams& p, const Weight& w, int n,
                        const __nv_bfloat16* x, float* smem,
                        const Store& store) {
  qwen38::Gemm<Emit>(qwen38::GemmWeightOf(w), kK, n, x, p.tokens, smem,
                     p.gemm_partials, p.gemm_counters, store);
}

// The kBf16 weight w of n rows of kK over x, every token's rows to store. A
// chunk or fewer takes each CTA's DenseBegin rows; more take a unit per (a
// round of 16-row blocks, chunk), so a few rows still spread over the grid.
template <int kK, typename Store>
__device__ void ProjectBf16(const Qwen38PrefillParams& p, const Weight& w,
                            int n, const __nv_bfloat16* x, float* partials,
                            const Store& store) {
  constexpr int kSlices = SlicesFor(kK);
  if (p.tokens <= kChunk) {
    prefill::FixedDense<kSlices>(prefill::Bf16Of(w), kK, n, x, p.tokens,
                                 partials, store);
    return;
  }
  constexpr int kRunRows = kWarps / kSlices * prefill::kBlockRows;
  const int chunks = Qwen38PrefillChunks(p.tokens);
  const int units = (n + kRunRows - 1) / kRunRows * chunks;
  for (int unit = blockIdx.x; unit < units; unit += gridDim.x) {
    const int begin = unit / chunks * kRunRows;
    const int t0 = unit % chunks * kChunk;
    const int count = min(kChunk, p.tokens - t0);
    prefill::ForTokens(count, [&]<int kNTiles>() {
      prefill::FixedDenseRange<kNTiles, kSlices, prefill::EachRow>(
          prefill::Bf16Of(w), kK, begin, min(n, begin + kRunRows),
          x + int64_t{t0} * kK, count, partials,
          [&](int row, int t, float y) { store(row, t0 + t, y); });
    });
  }
}

// A store adding row `row` of token t onto the residual.
__device__ __forceinline__ auto AddToResidual(const Qwen38PrefillParams& p) {
  return [&p](int row, int t, float y) {
    float* r = p.residual + int64_t{t} * kDim + row;
    *r = __ldcg(r) + y;
  };
}

// -------------------------------------------------------------- L2, L3

__device__ void InProjection(const Qwen38PrefillParams& p, const GdnParams& w,
                             float* partials) {
  Project<kDim>(p, w.in_proj, kGdnInt4Rows, p.h, partials,
                [&](int row, int t, float y) {
                  p.proj[int64_t{t} * kGdnInt4Rows + row] = y;
                });
  ProjectBf16<kDim>(p, w.gates, kGdnGateRows, p.h, partials,
                    [&](int row, int t, float y) {
                      if (row < kVHeads) {
                        p.beta[t * kVHeads + row] = gdn::Sigmoid(y);
                      } else {
                        const int h = row - kVHeads;
                        p.gate[t * kVHeads + h] =
                            -expf(w.a_log[h]) * gdn::Softplus(y + w.dt_bias[h]);
                      }
                    });
}

__device__ __forceinline__ float RoundTf32(float x) {
  return __uint_as_float(Tf32(x));
}

// Heads of 128 conv channels: q's, then k's, then v's.
constexpr int kConvHeads = kConvDim / kHeadDim;

// A warp per (token, conv head): each lane's four channels through the causal
// conv, whose taps before the prompt come from the conv state, and SiLU; q and
// k heads are then L2-normalized, q also scaled, as the decode's DeltaRule.
__device__ void Conv(const Qwen38PrefillParams& p, const GdnParams& w) {
  const int lane = threadIdx.x % 32;
  for (int item = GlobalWarp(); item < p.tokens * kConvHeads;
       item += GridWarps()) {
    const int t = item / kConvHeads;
    const int head = item % kConvHeads;
    float out[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int c = head * kHeadDim + 4 * lane + r;
      const __nv_bfloat16* tap_w = w.conv + int64_t{c} * kConvTaps;
      float acc = 0.f;
#pragma unroll
      for (int d = 0; d < kConvTaps; ++d) {
        // Input s of the sequence, t − 3 .. t: the prompt's from proj, those
        // before it from the state, oldest first.
        const int s = t - (kConvTaps - 1) + d;
        const float x =
            s >= 0 ? __ldcg(p.proj + int64_t{s} * kGdnInt4Rows + c)
                   : __ldcg(w.conv_state + int64_t{c} * (kConvTaps - 1) +
                            (kConvTaps - 1) + s);
        acc = fmaf(__bfloat162float(tap_w[d]), x, acc);
      }
      out[r] = gdn::Silu(acc);
    }
    float4 y = make_float4(out[0], out[1], out[2], out[3]);
    if (head < 2 * kKHeads) {
      const bool is_key = head >= kKHeads;
      float inv = rsqrtf(WarpSum(gdn::Dot4(y, y)) + 1e-6f);
      if (!is_key) inv *= gdn::kQueryScale;
      // Rounded to tf32 here, as the delta rule's mmas read them, so its
      // phases copy them as they are.
      y = make_float4(RoundTf32(y.x * inv), RoundTf32(y.y * inv),
                      RoundTf32(y.z * inv), RoundTf32(y.w * inv));
      float* dst = (is_key ? p.key : p.query) +
                   (int64_t{t} * kKHeads + head % kKHeads) * kHeadDim;
      reinterpret_cast<float4*>(dst)[lane] = y;
    } else {
      reinterpret_cast<float4*>(p.value + int64_t{t} * kValueDim +
                                (head - 2 * kKHeads) * kHeadDim)[lane] = y;
    }
  }
}

// ---------------------------------------------------------------------- L4
//
// The delta rule over one chunk, for value head h with dk × dv state S₀ at
// the chunk's start. Token t has key kₜ, query qₜ, value vₜ, βₜ and log decay
// gₜ; Gₜ = g₀ + .. + gₜ and γₜ = exp(Gₜ). Token by token, as the decode runs
// it, S ← exp(gₜ) × S then S ← S + kₜ ⊗ δₜ with δₜ = βₜ × (vₜ − Sᵀ kₜ), and
// oₜ = Sᵀ qₜ. Over the chunk, in matrices with a row per token (FLA's
// chunk_gated_delta_rule, its WY form):
//
//   (I + A) Δ = R,  R = β ⊙ (V − γ ⊙ K S₀),
//                   Aₜⱼ = βₜ × exp(Gₜ − Gⱼ) × (kₜ · kⱼ) for j < t, else 0,
//   O = γ ⊙ Q S₀ + M Δ,  Mₜⱼ = exp(Gₜ − Gⱼ) × (qₜ · kⱼ) for j ≤ t, else 0,
//   S = γ_{T−1} × S₀ + Kᵀ (e ⊙ Δ),  eⱼ = exp(G_{T−1} − Gⱼ).
//
// Δ = T R with T = (I + A)⁻¹, unit lower triangular, inverted in fp32 in
// 16-token blocks: each diagonal block by forward substitution, then each
// block below it from those nearer the diagonal,
//
//   Tᵢⱼ = −Tᵢᵢ Σ_{j ≤ k < i} Aᵢₖ Tₖⱼ.
//
// A, M and T do not depend on S₀, so L4a computes them for every (chunk,
// head) at once and leaves T and M in act. L4b then steps through a head's
// chunks in order with only the products that read S₀.
//
// Every product is a tf32 mma with fp32 accumulation. A state column (dv
// index) meets only its own columns of V, R, Δ, O and S, so an L4b CTA takes
// a run of one head's columns as the mmas' n dimension, and an element of a
// product sums over k in one order whichever tile holds its column.

// Blocks of the inverse, and the token rows of an mma tile.
constexpr int kBlock = 16;
constexpr int kBlocks = kChunk / kBlock;

struct DeltaShared {
  float key[kChunk * kHeadDim];    // swizzled
  float query[kChunk * kHeadDim];  // swizzled
  // Swizzled: A below the diagonal, T's blocks above it transposed; then M.
  float am[kChunk * kChunk];
  float g_sum[kChunk];  // Gₜ
  float beta[kChunk];

  // Once the queries' products are taken, their space holds Σₖ Aᵢₖ Tₖⱼ of an
  // inverse round's blocks.
  __device__ float* Sums() { return query; }
};
static_assert((kBlocks - 1) * kBlock * kBlock <= kChunk * kHeadDim);

// Element (row, col) of a row-major matrix of `stride` floats a row, the
// column XORed by the row's low 3 bits × 4: an A fragment's (g, t) and
// (g + 8, t) hit 32 banks.
__device__ __forceinline__ int Swizzled(int row, int col, int stride) {
  return row * stride + (col ^ ((row & 7) << 2));
}

// T at (i, j), from `am` as sh.am holds it after InvertUnitLower.
__device__ __forceinline__ float InverseAt(const float* am, int i, int j) {
  if (i == j) return 1.f;
  return i > j ? am[Swizzled(j, i, kChunk)] : 0.f;
}

// The A fragment of rows m0 .. m0 + 15 and columns k0 .. k0 + 7 of a
// swizzled, tf32-rounded matrix.
__device__ __forceinline__ void LoadA(const float* m, int stride, int m0,
                                      int k0, int lane, uint32_t (&a)[4]) {
  const int g = lane / 4;
  const int t = lane % 4;
  a[0] = __float_as_uint(m[Swizzled(m0 + g, k0 + t, stride)]);
  a[1] = __float_as_uint(m[Swizzled(m0 + g + 8, k0 + t, stride)]);
  a[2] = __float_as_uint(m[Swizzled(m0 + g, k0 + t + 4, stride)]);
  a[3] = __float_as_uint(m[Swizzled(m0 + g + 8, k0 + t + 4, stride)]);
}

// Row and column of accumulator element e of a tile at (m0, n0).
__device__ __forceinline__ int TileRow(int m0, int lane, int e) {
  return m0 + lane / 4 + (e >= 2 ? 8 : 0);
}
__device__ __forceinline__ int TileColumn(int n0, int lane, int e) {
  return n0 + 2 * (lane % 4) + (e & 1);
}

// Inverts I + A, held below sh.am's diagonal, into T above it.
__device__ void InvertUnitLower(DeltaShared& sh) {
  // A diagonal block's column c by forward substitution, a thread each.
  if (threadIdx.x < kChunk) {
    const int base = threadIdx.x / kBlock * kBlock;
    const int c = threadIdx.x % kBlock;
    float x[kBlock];
#pragma unroll
    for (int r = 0; r < kBlock; ++r) {
      float sum = 0.f;
#pragma unroll
      for (int s = 0; s < r; ++s) {
        sum = fmaf(sh.am[Swizzled(base + r, base + s, kChunk)], x[s], sum);
      }
      if (r < c) {
        x[r] = 0.f;
      } else if (r == c) {
        x[r] = 1.f;
      } else {
        x[r] = -sum;
      }
    }
#pragma unroll
    for (int r = 0; r < kBlock; ++r) {
      if (r > c) sh.am[Swizzled(base + c, base + r, kChunk)] = x[r];
    }
  }
  __syncthreads();
  // The blocks d below the diagonal, (i, j) = (b + d, b): their sums, then
  // Tᵢᵢ times them.
  constexpr int kPerBlock = kBlock * kBlock;
  for (int d = 1; d < kBlocks; ++d) {
    const int elements = (kBlocks - d) * kPerBlock;
    for (int e = threadIdx.x; e < elements; e += kThreads) {
      const int j = e / kPerBlock * kBlock;
      const int i = j + d * kBlock;
      const int r = e / kBlock % kBlock;
      const int c = e % kBlock;
      float sum = 0.f;
      for (int k = j; k < i; ++k) {
        sum = fmaf(sh.am[Swizzled(i + r, k, kChunk)],
                   InverseAt(sh.am, k, j + c), sum);
      }
      sh.Sums()[e] = sum;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < elements; e += kThreads) {
      const int j = e / kPerBlock * kBlock;
      const int i = j + d * kBlock;
      const int r = e / kBlock % kBlock;
      const int c = e % kBlock;
      const float* sums = sh.Sums() + e / kPerBlock * kPerBlock;
      float t = sums[r * kBlock + c];
      for (int s = 0; s < r; ++s) {
        t = fmaf(InverseAt(sh.am, i + r, i + s), sums[s * kBlock + c], t);
      }
      sh.am[Swizzled(j + c, i + r, kChunk)] = -t;
    }
    __syncthreads();
  }
}

// T = (I + A)⁻¹ of (chunk, value head h) in act, as sh.am holds it after
// InvertUnitLower; M follows it, swizzled and tf32-rounded as sh.am holds it
// for O.
__device__ __forceinline__ float* ChunkInverse(const Qwen38PrefillParams& p,
                                               int chunk, int h) {
  return reinterpret_cast<float*>(p.act) +
         (int64_t{chunk} * kVHeads + h) * 2 * kChunk * kChunk;
}
__device__ __forceinline__ float* ChunkScores(const Qwen38PrefillParams& p,
                                              int chunk, int h) {
  return ChunkInverse(p, chunk, h) + kChunk * kChunk;
}

// Value head h's chunk at t0 of `tokens` tokens into shared memory: its keys
// and queries (tf32 from Conv), Gₜ and βₜ.
__device__ void LoadChunk(const Qwen38PrefillParams& p, int h, int t0,
                          int tokens, DeltaShared& sh) {
  const int k_head = h / kHeadsPerKHead;
  for (int idx = threadIdx.x; idx < kChunk * kHeadDim; idx += kThreads) {
    const int t = idx / kHeadDim;
    const int d = idx % kHeadDim;
    const int64_t at = (int64_t{t0 + t} * kKHeads + k_head) * kHeadDim + d;
    const bool live = t < tokens;
    const int to = Swizzled(t, d, kHeadDim);
    sh.key[to] = live ? __ldcg(p.key + at) : 0.f;
    sh.query[to] = live ? __ldcg(p.query + at) : 0.f;
  }
  // The gates load at once into g_sum, then one thread sums them in order.
  if (threadIdx.x < kChunk) {
    const int t = threadIdx.x;
    const bool live = t < tokens;
    sh.g_sum[t] = live ? __ldcg(p.gate + (t0 + t) * kVHeads + h) : 0.f;
    sh.beta[t] = live ? __ldcg(p.beta + (t0 + t) * kVHeads + h) : 0.f;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float g_sum = 0.f;
    for (int t = 0; t < tokens; ++t) {
      g_sum += sh.g_sum[t];
      sh.g_sum[t] = g_sum;
    }
    for (int t = tokens; t < kChunk; ++t) sh.g_sum[t] = g_sum;
  }
  __syncthreads();
}

// L4a for (chunk, value head h): A and M from K Kᵀ and Q Kᵀ, T = (I + A)⁻¹,
// then T and M to act, and Gₜ over the chunk's gates. Warp w takes the token
// rows 16 (w / 4) .. + 15 and the token columns 16 (w % 4) .. + 15.
__device__ void ChunkMatricesUnit(const Qwen38PrefillParams& p, int chunk,
                                  int h, DeltaShared& sh) {
  static_assert(kWarps == 16 && kChunk == 64,
                "a warp per 16 × 16 of a 64 × 64 product");
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int g = lane / 4;
  const int quad = lane % 4;
  const int t0 = chunk * kChunk;
  const int tokens = min(kChunk, p.tokens - t0);
  LoadChunk(p, h, t0, tokens, sh);
  // No other unit reads these gates, so Gₜ replaces them for L4b.
  if (threadIdx.x < tokens) {
    p.gate[(t0 + threadIdx.x) * kVHeads + h] = sh.g_sum[threadIdx.x];
  }

  const int m0 = warp / 4 * kBlock;
  const int n0 = warp % 4 * kBlock;
  float kk[2][4] = {};
  float qk[2][4] = {};
  for (int k0 = 0; k0 < kHeadDim; k0 += 8) {
    uint32_t ak[4];
    uint32_t aq[4];
    LoadA(sh.key, kHeadDim, m0, k0, lane, ak);
    LoadA(sh.query, kHeadDim, m0, k0, lane, aq);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int n = n0 + 8 * j + g;
      const uint32_t bk0 =
          __float_as_uint(sh.key[Swizzled(n, k0 + quad, kHeadDim)]);
      const uint32_t bk1 =
          __float_as_uint(sh.key[Swizzled(n, k0 + quad + 4, kHeadDim)]);
      MmaTf32(kk[j], ak, bk0, bk1);
      MmaTf32(qk[j], aq, bk0, bk1);
    }
  }
  float* const scores = ChunkScores(p, chunk, h);
#pragma unroll
  for (int j = 0; j < 2; ++j) {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int t = TileRow(m0, lane, e);
      const int col = TileColumn(n0 + 8 * j, lane, e);
      // A token pair, col as the second token.
      const float decay = expf(sh.g_sum[t] - sh.g_sum[col]);
      sh.am[Swizzled(t, col, kChunk)] =
          col < t ? sh.beta[t] * decay * kk[j][e] : 0.f;
      scores[Swizzled(t, col, kChunk)] =
          __uint_as_float(Tf32(col <= t ? decay * qk[j][e] : 0.f));
    }
  }
  __syncthreads();
  InvertUnitLower(sh);
  float* const inverse = ChunkInverse(p, chunk, h);
  for (int i = threadIdx.x; i < kChunk * kChunk; i += kThreads) {
    inverse[i] = sh.am[i];
  }
  __syncthreads();
}

// The conv state takes the prompt's last kConvTaps − 1 inputs, after Conv has
// read the state: the last of (state, the prompt's inputs), oldest first.
__device__ void ShiftConvState(const Qwen38PrefillParams& p,
                               const GdnParams& w) {
  constexpr int kPast = kConvTaps - 1;
  for (int c = blockIdx.x * kThreads + threadIdx.x; c < kConvDim;
       c += gridDim.x * kThreads) {
    float* past = w.conv_state + int64_t{c} * kPast;
    float old[kPast];
#pragma unroll
    for (int j = 0; j < kPast; ++j) old[j] = past[j];
#pragma unroll
    for (int j = 0; j < kPast; ++j) {
      const int s = p.tokens - kPast + j;
      past[j] = s >= 0 ? __ldcg(p.proj + int64_t{s} * kGdnInt4Rows + c)
                       : old[kPast + s];
    }
  }
}

// L4a: a unit per (chunk, value head) over the grid.
__device__ void ChunkMatrices(const Qwen38PrefillParams& p, const GdnParams& w,
                              float* smem) {
  ShiftConvState(p, w);
  auto& sh = *reinterpret_cast<DeltaShared*>(smem);
  const int units = Qwen38PrefillChunks(p.tokens) * kVHeads;
  for (int unit = blockIdx.x; unit < units; unit += gridDim.x) {
    ChunkMatricesUnit(p, unit / kVHeads, unit % kVHeads, sh);
  }
}

// L4b's runs: each value head's state columns in kStateSplits runs of up to
// kStateColumns, a multiple of 4 so a run's values copy 16 bytes at a time.
constexpr int kStateSplits = 3;
constexpr int kStateColumns = 44;
static_assert(kStateSplits * kStateColumns >= kHeadDim &&
              kStateColumns % 4 == 0);
// A chunk's products take the run's columns in kRunTiles tiles of 16, a warp
// per (16 token rows, tile); the state's, in two halves of kHalfTiles tiles
// of 8, a warp per (16 dk rows, half).
constexpr int kRunTiles = 3;
constexpr int kTileWarps = kChunk / kBlock * kRunTiles;
constexpr int kHalfTiles = kRunTiles;
static_assert(kRunTiles * 16 >= kStateColumns && kTileWarps <= kWarps &&
              kHeadDim / kBlock * 2 == kWarps);
// Row strides, in floats, each so a B fragment's 32 elements hit 32 banks:
// a state column's dk row 4 banks off the next, and R's and Δ's token row 24
// banks off the next.
constexpr int kStateStride = kHeadDim + 4;
constexpr int kRunStride = 56;
static_assert(kRunStride >= kRunTiles * 16 && kRunStride % 32 == 24);

// L4b's shared memory. The run's state stays in it from the first chunk to
// the last; each operand of a chunk is copied in as soon as the one before
// it in the same space has been read.
struct DeltaStateShared {
  float key[kChunk * kHeadDim];  // swizzled
  // The queries (swizzled); once their products are taken, R and then Δ,
  // [t][kRunStride], and after them T, then M, as sh.am holds them in L4a.
  float query[kChunk * kHeadDim];
  float state[kStateColumns * kStateStride];  // [column][dk]
  float value[kChunk * kStateColumns];        // [t][column]
  float g_sum[kChunk];  // Gₜ
  float beta[kChunk];
  float gamma[kChunk];   // γₜ = exp(Gₜ)
  float to_end[kChunk];  // eₜ = exp(G_{T−1} − Gₜ)

  __device__ float* Delta() { return query; }
  __device__ float* Matrix() { return query + kChunk * kRunStride; }
};
static_assert(kChunk * kRunStride + kChunk * kChunk <= kChunk * kHeadDim);

// Starts copying the chunk at t0's keys or queries (`src` is p.key or
// p.query) of key head k_head into `rows`, swizzled; rows past `tokens` are
// zeroed.
__device__ void CopyChunkRows(const float* src, int k_head, int t0,
                              int tokens, float* rows) {
  constexpr int kQuads = kHeadDim / 4;
  for (int i = threadIdx.x; i < kChunk * kQuads; i += kThreads) {
    const int t = i / kQuads;
    const int d = i % kQuads * 4;
    float* dst = rows + Swizzled(t, d, kHeadDim);
    if (t < tokens) {
      CpAsync(dst, src + (int64_t{t0 + t} * kKHeads + k_head) * kHeadDim + d,
              16);
    } else {
      *reinterpret_cast<float4*>(dst) = make_float4(0.f, 0.f, 0.f, 0.f);
    }
  }
}

// Starts copying a kChunk × kChunk matrix of act (T or M) into `dst`.
__device__ void CopyChunkMatrix(const float* src, float* dst) {
  for (int i = 4 * threadIdx.x; i < kChunk * kChunk; i += 4 * kThreads) {
    CpAsync(dst + i, src + i, 16);
  }
}

// Starts copying value head h's values of columns c0 .. c0 + columns − 1 and
// the gates (Gₜ, βₜ) of the chunk at t0 into sh. Rows past `tokens` are
// left: every product masks them.
__device__ void CopyChunkValues(const Qwen38PrefillParams& p, int h, int c0,
                                int columns, int t0, int tokens,
                                DeltaStateShared& sh) {
  const int quads = columns / 4;
  for (int i = threadIdx.x; i < tokens * quads; i += kThreads) {
    const int t = i / quads;
    const int c = i % quads * 4;
    CpAsync(&sh.value[t * kStateColumns + c],
            p.value + int64_t{t0 + t} * kValueDim + h * kHeadDim + c0 + c, 16);
  }
}
__device__ void CopyChunkGates(const Qwen38PrefillParams& p, int h, int t0,
                               int tokens, DeltaStateShared& sh) {
  if (threadIdx.x < tokens) {
    const int at = (t0 + threadIdx.x) * kVHeads + h;
    CpAsync(&sh.g_sum[threadIdx.x], p.gate + at, 4);
    CpAsync(&sh.beta[threadIdx.x], p.beta + at, 4);
  }
}

// L4b for columns c0 .. c1 − 1 of value head h, c1 − c0 ≤ kStateColumns,
// through every chunk of the prompt in order, from T and M in act. Warp
// w < kTileWarps takes the token rows 16 (w / kRunTiles) .. + 15 and the
// columns 16 (w % kRunTiles) .. + 15 of a chunk's products; every warp takes
// the dk rows 16 (w / 2) .. + 15 and the columns 8 kHalfTiles (w % 2) .. of
// the state's. Columns past the run are computed and never stored.
//
// A chunk's copies: its keys, queries, gates and values come in while the
// chunk before it runs, T once the queries' products are taken, M once T's
// is. The state update runs before O, so the next keys load during O.
__device__ void DeltaStateRun(const Qwen38PrefillParams& p, const GdnParams& w,
                              int h, int c0, int c1, DeltaStateShared& sh) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int g = lane / 4;
  const int quad = lane % 4;
  const int columns = c1 - c0;
  const int k_head = h / kHeadsPerKHead;
  const int chunks = Qwen38PrefillChunks(p.tokens);
  auto tokens_of = [&](int chunk) {
    return min(kChunk, p.tokens - chunk * kChunk);
  };
  float* const state = w.state + int64_t{h * kHeadDim + c0} * kHeadDim;
  float* const delta_rows = sh.Delta();
  float* const am = sh.Matrix();

  for (int i = threadIdx.x; i < columns * kHeadDim / 4; i += kThreads) {
    const int col = i / (kHeadDim / 4);
    const int d = i % (kHeadDim / 4) * 4;
    CpAsync(&sh.state[col * kStateStride + d], state + col * kHeadDim + d, 16);
  }
  CopyChunkRows(p.key, k_head, 0, tokens_of(0), sh.key);
  CopyChunkRows(p.query, k_head, 0, tokens_of(0), sh.query);
  CopyChunkGates(p, h, 0, tokens_of(0), sh);
  CopyChunkValues(p, h, c0, columns, 0, tokens_of(0), sh);
  CpAsyncCommit();

  const bool tiles = warp < kTileWarps;
  const int m0 = warp / kRunTiles * kBlock;
  const int n0 = warp % kRunTiles * kBlock;
  const int d0 = warp / 2 * kBlock;
  const int s_n0 = warp % 2 * 8 * kHalfTiles;
  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int t0 = chunk * kChunk;
    const int tokens = tokens_of(chunk);
    const bool more = chunk + 1 < chunks;
    CpAsyncWait<0>();
    __syncthreads();
    if (threadIdx.x < kChunk) {
      const int t = threadIdx.x;
      const bool live = t < tokens;
      if (!live) sh.beta[t] = 0.f;
      sh.gamma[t] = live ? expf(sh.g_sum[t]) : 0.f;
      sh.to_end[t] = live ? expf(sh.g_sum[tokens - 1] - sh.g_sum[t]) : 0.f;
    }
    __syncthreads();

    // K S₀ and Q S₀ over the warp's tiles.
    float ks[2][4] = {};
    float qs[2][4] = {};
    for (int k0 = 0; tiles && k0 < kHeadDim; k0 += 8) {
      uint32_t ak[4];
      uint32_t aq[4];
      LoadA(sh.key, kHeadDim, m0, k0, lane, ak);
      LoadA(sh.query, kHeadDim, m0, k0, lane, aq);
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int n = n0 + 8 * j + g;
        const bool live = n < columns;
        const float* s0 = &sh.state[n * kStateStride + k0 + quad];
        const uint32_t bs0 = Tf32(live ? s0[0] : 0.f);
        const uint32_t bs1 = Tf32(live ? s0[4] : 0.f);
        MmaTf32(ks[j], ak, bs0, bs1);
        MmaTf32(qs[j], aq, bs0, bs1);
      }
    }
    __syncthreads();  // the queries' space becomes R's and T's
    CopyChunkMatrix(ChunkInverse(p, chunk, h), am);
    CpAsyncCommit();

    // R into delta_rows; γ ⊙ Q S₀ stays with the warp.
#pragma unroll
    for (int j = 0; tiles && j < 2; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int t = TileRow(m0, lane, e);
        const int col = TileColumn(n0 + 8 * j, lane, e);
        float r = 0.f;
        if (t < tokens && col < columns) {
          const float v = sh.value[t * kStateColumns + col];
          r = sh.beta[t] * (v - sh.gamma[t] * ks[j][e]);
        }
        delta_rows[t * kRunStride + col] = r;
        qs[j][e] *= sh.gamma[t];
      }
    }
    __syncthreads();  // the values are read
    if (more) {
      CopyChunkValues(p, h, c0, columns, t0 + kChunk, tokens_of(chunk + 1),
                      sh);
    }
    CpAsyncCommit();
    CpAsyncWait<1>();  // T
    __syncthreads();

    // Δ = T R. In the tile's rows, T is zero past the tile's last token.
    float delta[2][4] = {};
    for (int k0 = 0; tiles && k0 < m0 + kBlock; k0 += 8) {
      uint32_t a[4];
      a[0] = Tf32(InverseAt(am, m0 + g, k0 + quad));
      a[1] = Tf32(InverseAt(am, m0 + g + 8, k0 + quad));
      a[2] = Tf32(InverseAt(am, m0 + g, k0 + quad + 4));
      a[3] = Tf32(InverseAt(am, m0 + g + 8, k0 + quad + 4));
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int n = n0 + 8 * j + g;
        MmaTf32(delta[j], a, Tf32(delta_rows[(k0 + quad) * kRunStride + n]),
                Tf32(delta_rows[(k0 + quad + 4) * kRunStride + n]));
      }
    }
    __syncthreads();  // T and R are read
    CopyChunkMatrix(ChunkScores(p, chunk, h), am);
    CpAsyncCommit();
#pragma unroll
    for (int j = 0; tiles && j < 2; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int t = TileRow(m0, lane, e);
        const int col = TileColumn(n0 + 8 * j, lane, e);
        delta_rows[t * kRunStride + col] = delta[j][e];
      }
    }
    __syncthreads();

    // S = γ_{T−1} × S₀ + Kᵀ (e ⊙ Δ), stored over S₀ by the warp that read it.
    const float gamma_end = sh.gamma[tokens - 1];
    float s[kHalfTiles][4];
#pragma unroll
    for (int j = 0; j < kHalfTiles; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int col = TileColumn(s_n0 + 8 * j, lane, e);
        s[j][e] = col < columns
                      ? gamma_end *
                            sh.state[col * kStateStride + TileRow(d0, lane, e)]
                      : 0.f;
      }
    }
    for (int k0 = 0; k0 < tokens; k0 += 8) {
      // Kᵀ at (d, t) is K at (t, d).
      uint32_t a[4];
      a[0] = __float_as_uint(sh.key[Swizzled(k0 + quad, d0 + g, kHeadDim)]);
      a[1] =
          __float_as_uint(sh.key[Swizzled(k0 + quad, d0 + g + 8, kHeadDim)]);
      a[2] =
          __float_as_uint(sh.key[Swizzled(k0 + quad + 4, d0 + g, kHeadDim)]);
      a[3] = __float_as_uint(
          sh.key[Swizzled(k0 + quad + 4, d0 + g + 8, kHeadDim)]);
      const float e0 = sh.to_end[k0 + quad];
      const float e1 = sh.to_end[k0 + quad + 4];
#pragma unroll
      for (int j = 0; j < kHalfTiles; ++j) {
        const int n = s_n0 + 8 * j + g;
        MmaTf32(s[j], a, Tf32(e0 * delta_rows[(k0 + quad) * kRunStride + n]),
                Tf32(e1 * delta_rows[(k0 + quad + 4) * kRunStride + n]));
      }
    }
#pragma unroll
    for (int j = 0; j < kHalfTiles; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int col = TileColumn(s_n0 + 8 * j, lane, e);
        if (col < columns) {
          sh.state[col * kStateStride + TileRow(d0, lane, e)] = s[j][e];
        }
      }
    }
    __syncthreads();  // the keys and gates are read
    if (more) {
      CopyChunkRows(p.key, k_head, t0 + kChunk, tokens_of(chunk + 1), sh.key);
      CopyChunkGates(p, h, t0 + kChunk, tokens_of(chunk + 1), sh);
    }
    CpAsyncCommit();
    CpAsyncWait<1>();  // M
    __syncthreads();

    // O = γ ⊙ Q S₀ + M Δ.
    for (int k0 = 0; tiles && k0 < m0 + kBlock; k0 += 8) {
      uint32_t a[4];
      LoadA(am, kChunk, m0, k0, lane, a);
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int n = n0 + 8 * j + g;
        MmaTf32(qs[j], a, Tf32(delta_rows[(k0 + quad) * kRunStride + n]),
                Tf32(delta_rows[(k0 + quad + 4) * kRunStride + n]));
      }
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int t = TileRow(m0, lane, e);
        const int col = TileColumn(n0 + 8 * j, lane, e);
        if (tiles && t < tokens && col < columns) {
          p.core[int64_t{t0 + t} * kValueDim + h * kHeadDim + c0 + col] =
              qs[j][e];
        }
      }
    }
    __syncthreads();  // Δ and M are read
    if (more) {
      CopyChunkRows(p.query, k_head, t0 + kChunk, tokens_of(chunk + 1),
                    sh.query);
    }
    CpAsyncCommit();
  }

  for (int i = threadIdx.x; i < columns * kHeadDim; i += kThreads) {
    const int col = i / kHeadDim;
    const int d = i % kHeadDim;
    state[col * kHeadDim + d] = sh.state[col * kStateStride + d];
  }
  __syncthreads();
}

// L4b: a (head, run of columns) unit per CTA, kVHeads × kStateSplits in all,
// each through the prompt's chunks in order.
__device__ void DeltaState(const Qwen38PrefillParams& p, const GdnParams& w,
                           float* smem) {
  auto& sh = *reinterpret_cast<DeltaStateShared*>(smem);
  for (int unit = blockIdx.x; unit < kVHeads * kStateSplits;
       unit += gridDim.x) {
    const int h = unit / kStateSplits;
    const int c0 = unit % kStateSplits * kStateColumns;
    DeltaStateRun(p, w, h, c0, min(kHeadDim, c0 + kStateColumns), sh);
  }
}

// ---------------------------------------------------------------- L5, L6

// A warp per (token, value head): RMSNorm(core) × out_norm × SiLU(z) in bf16,
// as the decode's OutProjection.
__device__ void GatedNorm(const Qwen38PrefillParams& p, const GdnParams& w) {
  const int lane = threadIdx.x % 32;
  const float4 nw = make_float4(__bfloat162float(w.out_norm[4 * lane]),
                                __bfloat162float(w.out_norm[4 * lane + 1]),
                                __bfloat162float(w.out_norm[4 * lane + 2]),
                                __bfloat162float(w.out_norm[4 * lane + 3]));
  for (int item = GlobalWarp(); item < p.tokens * kVHeads;
       item += GridWarps()) {
    const int t = item / kVHeads;
    const int h = item % kVHeads;
    const int at = h * kHeadDim + 4 * lane;
    const float4 o = __ldcg(
        reinterpret_cast<const float4*>(p.core + int64_t{t} * kValueDim + at));
    const float4 z = __ldcg(reinterpret_cast<const float4*>(
        p.proj + int64_t{t} * kGdnInt4Rows + kConvDim + at));
    const float inv = rsqrtf(WarpSum(gdn::Dot4(o, o)) / kHeadDim + w.eps);
    __nv_bfloat16* out = p.attn + int64_t{t} * kValueDim + at;
    out[0] = __float2bfloat16(o.x * inv * nw.x * gdn::Silu(z.x));
    out[1] = __float2bfloat16(o.y * inv * nw.y * gdn::Silu(z.y));
    out[2] = __float2bfloat16(o.z * inv * nw.z * gdn::Silu(z.z));
    out[3] = __float2bfloat16(o.w * inv * nw.w * gdn::Silu(z.w));
  }
}

__device__ void OutProjection(const Qwen38PrefillParams& p, const GdnParams& w,
                              float* partials) {
  if (w.out_proj.format == WeightFormat::kBf16) {
    ProjectBf16<kValueDim>(p, w.out_proj, kDim, p.attn, partials,
                           AddToResidual(p));
  } else {
    Project<kValueDim>(p, w.out_proj, kDim, p.attn, partials,
                       AddToResidual(p));
  }
}

// -------------------------------------------------------------------- F4

// How a launch splits each token's keys (qwen38_prefill.h): one chunk into
// spans of kQwen38PrefillAttnSpan positions, so a long cache spreads over the
// grid; several chunks into one span, each (chunk, query head) a CTA already.
// Span s covers positions [s × length, (s + 1) × length); `count` cover every
// position the launch reaches, and a partial sits at [t][h][count].
struct AttnSpans {
  int length;
  int count;

  __device__ AttnSpans(const Qwen38PrefillParams& p, int pos0)
      : length(p.tokens <= kChunk ? kQwen38PrefillAttnSpan : pos0 + p.tokens),
        count(Of(pos0 + p.tokens)) {}

  // The spans of `keys` positions.
  __device__ int Of(int keys) const { return (keys + length - 1) / length; }

  // Item (token t, query head h)'s partial over span s.
  __device__ int64_t PartialAt(int t, int h, int s) const {
    return (int64_t{t} * kQwen38QHeads + h) * count + s;
  }
};

// FlashSpan's store of chunk t0's rows of query head h over one span, into
// that span's partials.
struct Partial {
  const Qwen38PrefillParams& p;
  const AttnSpans& spans;
  int t0;
  int h;
  int span;

  __device__ void Row(int t, float m, float l) const {
    reinterpret_cast<float2*>(p.partial_ml)[spans.PartialAt(t0 + t, h, span)] =
        make_float2(m, l);
  }
  __device__ void Pair(int t, int dim, float, float o0, float o1) const {
    float* out = p.partial_o + spans.PartialAt(t0 + t, h, span) * kHead + dim;
    *reinterpret_cast<float2*>(out) = make_float2(o0, o1);
  }
};

// FlashSpan's store when a token's keys all fall in one span: its output
// normalized and gated into attn, as MergeAttention does with one partial.
struct GatedOutput {
  const Qwen38PrefillParams& p;
  int t0;
  int h;

  __device__ void Row(int, float, float) const {}
  __device__ void Pair(int t, int dim, float l, float o0, float o1) const {
    const int64_t token = t0 + t;
    const float2 gate = *reinterpret_cast<const float2*>(
        p.proj + token * kQkvRows + kGateRow + h * kHead + dim);
    const float inv = 1.f / l;
    *reinterpret_cast<__nv_bfloat162*>(p.attn + token * kQDim + h * kHead +
                                       dim) =
        __floats2bfloat162_rn(o0 * inv * thinker::Sigmoid(gate.x),
                              o1 * inv * thinker::Sigmoid(gate.y));
  }
};

// A CTA per (chunk, query head h, span of keys): the chunk's tokens of the
// head, after QK-norm and M-RoPE, attend to its KV group's keys in the span
// up to the chunk's last position (qwen38_prefill_attention.cuh), each into a
// partial, empty where the token sees no key in the span, which the merge
// never reads. When the launch's keys fit one span, each token's output is
// whole, and goes normalized and gated straight to attn. The last chunks,
// which see the most keys, go first. The spans follow the positions and the
// launch's tokens alone, so a partial is the same whichever CTA takes it.
__device__ void PrefillAttention(const Qwen38PrefillParams& p,
                                 const Qwen38LayerParams& f, int pos0,
                                 float* smem) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  auto& sh = *reinterpret_cast<qwen38::FlashShared*>(smem);
  const AttnSpans spans(p, pos0);
  const int chunks = Qwen38PrefillChunks(p.tokens);
  const int per_chunk = spans.count * kQwen38QHeads;
  for (int unit = blockIdx.x; unit < chunks * per_chunk; unit += gridDim.x) {
    const int t0 = (chunks - 1 - unit / per_chunk) * kChunk;
    const int span = unit % per_chunk / kQwen38QHeads;
    const int h = unit % kQwen38QHeads;
    const int tokens = min(kChunk, p.tokens - t0);
    const int begin = span * spans.length;
    const int end = min(pos0 + t0 + tokens, begin + spans.length);
    if (begin >= end) continue;  // past the chunk's last position
    // Lane l holds dims 8 l .. 8 l + 7, the row's chunk l.
    for (int t = warp; t < qwen38::kFlashRows; t += kWarps) {
      float q[kDimsPerLane] = {};
      if (t < tokens) {
        LoadLane(p.proj + int64_t{t0 + t} * kQkvRows + h * kHead, lane, q);
        NormRope(p, f, f.q_norm, t0 + t, lane, q);
      }
      qwen38::FlashQueryChunk(sh, t, lane) = PackBf16x8(q);
    }
    if (spans.count == 1) {
      qwen38::FlashSpan(sh, f.kv, h / kGqa, begin, end, pos0 + t0, tokens,
                        GatedOutput{p, t0, h});
    } else {
      qwen38::FlashSpan(sh, f.kv, h / kGqa, begin, end, pos0 + t0, tokens,
                        Partial{p, spans, t0, h, span});
    }
  }
}

// A warp per (token, query head): the partials of the spans its keys reach,
// each holding at least one of them, merged in span order as the decode's
// Merge, then gated into attn. With one span, PrefillAttention has done it.
__device__ void MergeAttention(const Qwen38PrefillParams& p, int pos0) {
  const int lane = threadIdx.x % 32;
  const AttnSpans spans(p, pos0);
  if (spans.count == 1) return;
  for (int item = GlobalWarp(); item < p.tokens * kQwen38QHeads;
       item += GridWarps()) {
    const int t = item / kQwen38QHeads;
    const int h = item % kQwen38QHeads;
    const int reached = spans.Of(pos0 + t + 1);
    float m = -INFINITY;
    for (int s = 0; s < reached; ++s) {
      const float2 ml = __ldcg(reinterpret_cast<const float2*>(p.partial_ml) +
                               spans.PartialAt(t, h, s));
      m = fmaxf(m, ml.x);
    }
    float l = 0.f;
    float o[kDimsPerLane] = {};
    for (int s = 0; s < reached; ++s) {
      const int64_t at = spans.PartialAt(t, h, s);
      const float2 ml =
          __ldcg(reinterpret_cast<const float2*>(p.partial_ml) + at);
      const float weight = expf(ml.x - m);
      l = fmaf(ml.y, weight, l);
      float part[kDimsPerLane];
      LoadLane(p.partial_o + at * kHead, lane, part);
#pragma unroll
      for (int i = 0; i < kDimsPerLane; ++i) o[i] = fmaf(part[i], weight, o[i]);
    }
    float gate[kDimsPerLane];
    LoadLane(p.proj + int64_t{t} * kQkvRows + kGateRow + h * kHead, lane, gate);
    const float inv = 1.f / l;
#pragma unroll
    for (int i = 0; i < kDimsPerLane; ++i) {
      o[i] = o[i] * inv * thinker::Sigmoid(gate[i]);
    }
    reinterpret_cast<uint4*>(p.attn + int64_t{t} * kQDim + h * kHead)[lane] =
        PackBf16x8(o);
  }
}

// ---------------------------------------------------------------- M2, M3

__device__ void GateUp(const Qwen38PrefillParams& p, const Qwen38MlpParams& m,
                       float* partials) {
  Project<kDim, prefill::RowPairs>(
      p, m.w13, 2 * kFfn, p.h, partials,
      [&](int row, int t, float gate, float up) {
        p.act[int64_t{t} * kFfn + row / 2] = __float2bfloat16(Silu(gate) * up);
      });
}

// ------------------------------------------------------------------ the kernel

constexpr int kSmemBytes = std::max<int>(
    {qwen38::kGemmSmemBytes,
     prefill::FixedDenseSmemBytes(SlicesFor(kDim)),
     prefill::FixedDenseSmemBytes(SlicesFor(kQDim)),
     static_cast<int>(sizeof(DeltaShared)),
     static_cast<int>(sizeof(DeltaStateShared)),
     static_cast<int>(sizeof(qwen38::FlashShared)),
     static_cast<int>(kDim * sizeof(__nv_bfloat16))});

// Copies every token's residual to dst.
__device__ void DumpResidual(const Qwen38PrefillParams& p, float* dst) {
  const int64_t n = int64_t{p.tokens} * kDim;
  for (int64_t i = blockIdx.x * kThreads + threadIdx.x; i < n;
       i += gridDim.x * kThreads) {
    dst[i] = __ldcg(p.residual + i);
  }
}

__global__ void __launch_bounds__(kThreads, 1)
    Qwen38PrefillKernel(const __grid_constant__ Qwen38PrefillParams p) {
  extern __shared__ float smem[];
  __shared__ float red[kWarps];
  GridBarrier barrier(p.sync, p.error, p.timeout_ns, 0);
  int64_t* stamps =
      p.profile == nullptr
          ? nullptr
          : p.profile +
                int64_t{blockIdx.x} * 2 * Qwen38PrefillBarriers(p.num_layers);
  // The arrival stamp waits for the whole CTA to finish the phase.
  auto sync = [&] {
    if (stamps != nullptr) {
      __syncthreads();
      if (threadIdx.x == 0) stamps[2 * barrier.index()] = GlobalTimer();
    }
    barrier.Sync();
    if (stamps != nullptr && threadIdx.x == 0) {
      stamps[2 * (barrier.index() - 1) + 1] = GlobalTimer();
    }
  };
  const int pos0 = *p.pos0;
  const int64_t residual_floats = int64_t{p.tokens} * kDim;

  for (int layer = 0; layer < p.num_layers; ++layer) {
    float* dump = p.hidden != nullptr ? p.hidden + layer * residual_floats : nullptr;
    const Qwen38MlpParams* mlp;
    if (Qwen38IsFull(layer)) {
      const Qwen38LayerParams& f = p.full[Qwen38FullIndex(layer)];
      TokenNorm(p, f.input_norm, f.eps, dump, red);
      sync();
      Project<kDim>(p, f.wqkv, kQkvRows, p.h, smem,
                    [&](int row, int t, float y) {
                      p.proj[int64_t{t} * kQkvRows + row] = y;
                    });
      sync();
      ChunkKv(p, f, pos0);
      sync();
      PrefillAttention(p, f, pos0, smem);
      sync();
      MergeAttention(p, pos0);
      sync();
      Project<kQDim>(p, f.wo, kDim, p.attn, smem, AddToResidual(p));
      mlp = &f.mlp;
    } else {
      const GdnParams& w = p.linear[Qwen38LinearIndex(layer)];
      TokenNorm(p, w.norm, w.eps, dump, red);
      sync();
      InProjection(p, w, smem);
      sync();
      Conv(p, w);
      sync();
      ChunkMatrices(p, w, smem);
      sync();
      DeltaState(p, w, smem);
      sync();
      GatedNorm(p, w);
      sync();
      OutProjection(p, w, smem);
      mlp = &p.linear_mlp[Qwen38LinearIndex(layer)];
    }
    sync();
    TokenNorm(p, mlp->post_norm, mlp->eps, nullptr, red);
    sync();
    GateUp(p, *mlp, smem);
    sync();
    Project<kFfn>(p, mlp->w2, kDim, p.act, smem, AddToResidual(p));
    sync();
  }
  if (p.hidden != nullptr) DumpResidual(p, p.hidden + p.num_layers * residual_floats);
  if (p.last_hidden != nullptr) DumpResidual(p, p.last_hidden);
  if (p.logits == nullptr) return;

  // Every CTA normalizes the last token itself, then takes its LM head rows,
  // as the decode does.
  auto* xs = reinterpret_cast<uint4*>(smem);
  RmsNorm<qwen38::NormDims>(p.residual + int64_t{p.tokens - 1} * kDim, nullptr,
                            p.final_norm, p.eps,
                            reinterpret_cast<__nv_bfloat16*>(xs), nullptr, red);
  __syncthreads();
  qwen38::LmHeadRows<1>(
      p.lm_head, p.vocab, xs,
      [&](int row, int, float logit) { p.logits[row] = logit; });
}

// One projection alone (LaunchQwen38Projection).
template <int kK>
__global__ void __launch_bounds__(kThreads, 1)
    Qwen38ProjectionKernel(const Weight w, int n, const __nv_bfloat16* x,
                           int tokens, float* y, float* partials,
                           unsigned* counters) {
  extern __shared__ float smem[];
  qwen38::Gemm(qwen38::GemmWeightOf(w), kK, n, x, tokens, smem, partials,
               counters, [&](int row, int t, float v) {
                 y[int64_t{t} * n + row] = v;
               });
}

template <int kK>
cudaError_t LaunchProjection(const Weight& w, int n, const __nv_bfloat16* x,
                             int tokens, float* y, float* partials,
                             unsigned* counters, int num_ctas,
                             cudaStream_t stream) {
  auto* kernel = Qwen38ProjectionKernel<kK>;
  constexpr int kBytes = qwen38::kGemmSmemBytes;
  const cudaError_t err = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kBytes);
  if (err != cudaSuccess) return err;
  kernel<<<num_ctas, kThreads, kBytes, stream>>>(w, n, x, tokens, y, partials,
                                                 counters);
  return cudaGetLastError();
}

}  // namespace

cudaError_t LaunchQwen38Projection(const Weight& w, int n, int k,
                                   const __nv_bfloat16* x, int tokens,
                                   float* y, float* partials,
                                   unsigned* counters, int num_ctas,
                                   cudaStream_t stream) {
  switch (k) {
    case kQwen38Dim:
      return LaunchProjection<kQwen38Dim>(w, n, x, tokens, y, partials,
                                          counters, num_ctas, stream);
    case kQwen38QDim:
      return LaunchProjection<kQwen38QDim>(w, n, x, tokens, y, partials,
                                           counters, num_ctas, stream);
    case kQwen38Ffn:
      return LaunchProjection<kQwen38Ffn>(w, n, x, tokens, y, partials,
                                          counters, num_ctas, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

cudaError_t LaunchQwen38Prefill(const Qwen38PrefillParams& params,
                                int num_ctas, cudaStream_t stream) {
  cudaError_t err =
      cudaMemsetAsync(params.sync, 0, kSyncWords * sizeof(unsigned), stream);
  if (err != cudaSuccess) return err;
  err = cudaFuncSetAttribute(Qwen38PrefillKernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kSmemBytes);
  if (err != cudaSuccess) return err;
  Qwen38PrefillKernel<<<num_ctas, kThreads, kSmemBytes, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace s2mk
