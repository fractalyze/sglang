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

// Hopper grouped W4A16 MoE GEMM, swap-AB: out[r, n] = sum_k W[e(r)][n, k] * a[r / div, k].
//
// The dequantised weight tile is the wgmma A operand (register-sourced, 64-row
// atoms; each consumer warpgroup owns one 128-row weight tile, so a CTA covers 256
// output rows); the routed tokens of one moe_align block are the wgmma N operand,
// gathered row by row into 128B-swizzled shared memory once for both warpgroups. A
// producer warpgroup streams weights, scales and zeros with bulk async copies and
// gathers tokens with cp.async; consumers dequantise in registers. Work is
// persistent over (token block, 256-row tile) pairs, rows fastest so concurrent
// CTAs share the block's tokens in L2.
//
// With kPingPong, the consumer warpgroups instead take whole pairs in turn, each
// running all four atoms, so one warpgroup's epilogue runs under the other's
// mainloop. That pays off when a pair has few k-tiles, as in the down projection.
// Every output element still comes from one wgmma chain over K, so the result is
// bitwise the same as the cooperative split.
//
// The weight layout is produced by sglang/kernels/ops/moe/w4a16_moe_sm90.py.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/mbarrier.cuh>
#include <sgl_kernel/utils.cuh>

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <tvm/ffi/container/tensor.h>

#include "fragment.cuh"
#include <cstdint>

namespace sglang {

namespace w4a16_moe_sm90 {

using bf16 = cute::bfloat16_t;

// Shared with the Python repack; see W4A16MoeWeights there.
inline constexpr int kTileN = 128;
inline constexpr int kTileK = 128;
inline constexpr int kAtomRows = 64;  // wgmma M
// A CTA covers two adjacent weight tiles, one per consumer warpgroup, so the
// gathered tokens and the per-tile overhead are shared by 256 output rows.
inline constexpr int kTilesPerCta = 2;
inline constexpr int kWordsPerStage = kTileN * kTileK / 8;

inline constexpr int kWeightBytes = kWordsPerStage * 4;
inline constexpr int kScaleBytes = kTileN * 2;
inline constexpr int kZeroBytes = kTileN;
inline constexpr int kChunksPerTokenRow = kTileK * 2 / 16;

inline constexpr int kProducerThreads = 128;
inline constexpr int kConsumerThreads = 256;
inline constexpr int kThreads = kProducerThreads + kConsumerThreads;
// Hardware limit for one CTA's dynamic shared memory on sm90.
inline constexpr int kSmemLimit = 227 * 1024;
// Arbitrary cap; enough bytes in flight per SM to cover HBM latency at 16 tokens.
inline constexpr int kMaxStages = 12;

constexpr int round_up(int x, int m) {
  return (x + m - 1) / m * m;
}

template <int kTokenBlock, bool kPingPong>
struct Config {
  static_assert(kTokenBlock % 8 == 0 && kTokenBlock <= 256, "wgmma N must be a multiple of 8 up to 256");

  static constexpr int kTokenBytes = kTokenBlock * kTileK * 2;
  // The token tile is the swizzled wgmma B operand and needs 1024B alignment.
  // Stage: [weights x2][tokens][scales x2][zeros x2], warpgroup w's at index w.
  static constexpr int kTokenOffset = kTilesPerCta * kWeightBytes;
  static constexpr int kScaleOffset = kTokenOffset + kTokenBytes;
  static constexpr int kZeroOffset = kScaleOffset + kTilesPerCta * kScaleBytes;
  static constexpr int kStageBytes = round_up(kZeroOffset + kTilesPerCta * kZeroBytes, 1024);

  static_assert(!kPingPong || kTokenBlock <= 32, "ping-pong accumulators spill past 32 tokens");
  // Output rows one consumer warpgroup computes per tile: its own weight tile, or
  // with kPingPong the whole pair.
  static constexpr int kWarpgroupRows = kPingPong ? kTilesPerCta * kTileN : kTileN;
  // Padded so a warp's column-wise accumulator stores spread over banks.
  static constexpr int kEpilogueStride = kWarpgroupRows + 8;
  static constexpr int kEpilogueBytes = 2 * kTokenBlock * kEpilogueStride * 2;
  // Per consumer warpgroup: the tile's routed row ids, then their top-k weights.
  static constexpr int kMetaBytes = 2 * kTokenBlock * 8;
  static constexpr int kBarrierBytes = 2 * kMaxStages * 8;
  // Dynamic smem is only 16B-aligned; the stages are realigned to 1024B in-kernel.
  static constexpr int kAlignSlack = 1024;

  static constexpr int kStages =
      std::min(kMaxStages, (kSmemLimit - kEpilogueBytes - kMetaBytes - kBarrierBytes - kAlignSlack) / kStageBytes);
  static_assert(kStages >= 3, "token block too wide for a three-stage pipeline");
  static constexpr int kSmemBytes = kAlignSlack + kStages * kStageBytes + kEpilogueBytes + kMetaBytes + kBarrierBytes;

  static constexpr int kStageTxBytes = kTilesPerCta * (kWeightBytes + kScaleBytes + kZeroBytes);
  // One expect_tx arrival plus one cp.async arrival per producer thread.
  static constexpr int kFullArrivals = 1 + kProducerThreads;

  using SmemLayoutTokens = decltype(cute::tile_to_shape(
      cute::GMMA::Layout_K_SW128_Atom<bf16>{}, cute::Shape<cute::Int<kTokenBlock>, cute::Int<kTileK>>{}));
  using TiledMma = decltype(cute::make_tiled_mma(
      cute::SM90::GMMA::
          rs_op_selector<bf16, bf16, float, cute::Shape<cute::_64, cute::Int<kTokenBlock>, cute::_16>>()));
};

struct Params {
  const bf16* a;
  bf16* out;
  const uint32_t* qweight;
  const bf16* scales;
  const uint8_t* zeros;
  const int32_t* sorted_token_ids;
  const int32_t* expert_ids;
  const int32_t* num_tokens_post_padded;
  const float* topk_weights;  // nullptr: no per-row scaling
  int num_rows;               // routed rows of `out`; larger sorted ids are padding
  int a_row_divisor;
  int k;
  int n;
};

__device__ __forceinline__ void bulk_copy_g2s(void* dst, const void* src, int bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
                   device::ptx::to_shared(dst)),
               "l"(src),
               "r"(bytes),
               "r"(device::ptx::to_shared(bar))
               : "memory");
}

// Zero-fills the 16 bytes when `valid` is false, which keeps padding rows finite.
__device__ __forceinline__ void cp_async_16(void* dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(device::ptx::to_shared(dst)),
               "l"(src),
               "r"(valid ? 16 : 0)
               : "memory");
}

__device__ __forceinline__ void cp_async_arrive_noinc(uint64_t* bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(device::ptx::to_shared(bar)) : "memory");
}

__device__ __forceinline__ void named_barrier_sync(int id, int threads) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(threads) : "memory");
}

// Position in the stage ring: the slot and the parity of its current fill.
struct RingPos {
  int stage = 0;
  uint32_t phase = 0;
  __device__ __forceinline__ void advance(int num_stages) {
    if (++stage == num_stages) {
      stage = 0;
      phase ^= 1;
    }
  }
};

template <int kTokenBlock, bool kPingPong>
__device__ __forceinline__ void produce(const Params& p, uint8_t* stages, uint64_t* full, uint64_t* empty) {
  using Cfg = Config<kTokenBlock, kPingPong>;
  constexpr int kRowsPerThread = kTokenBlock * kChunksPerTokenRow / kProducerThreads;
  static_assert(kRowsPerThread >= 1, "every producer thread gathers at least one chunk");

  const int tid = threadIdx.x;
  const int chunk = tid % kChunksPerTokenRow;
  const int first_row = tid / kChunksPerTokenRow;
  constexpr int kRowStride = kProducerThreads / kChunksPerTokenRow;

  const int n_tiles = p.n / kTileN;
  const int n_pairs = n_tiles / kTilesPerCta;
  const int k_tiles = p.k / kTileK;
  const int num_tiles = (*p.num_tokens_post_padded / kTokenBlock) * n_pairs;

  // Global reads a tile needs before its first copy; loaded one tile ahead so
  // their latency overlaps the previous tile's stages.
  struct Meta {
    int expert = -1;  // negative: no local expert, the tile is skipped
    int ids[kRowsPerThread];
  };
  auto load_meta = [&](int tile) {
    Meta m;
    if (tile < num_tiles) {
      const int m_block = tile / n_pairs;
      m.expert = p.expert_ids[m_block];
#pragma unroll
      for (int i = 0; i < kRowsPerThread; ++i) {
        m.ids[i] = p.sorted_token_ids[m_block * kTokenBlock + first_row + i * kRowStride];
      }
    }
    return m;
  };

  RingPos pos;
  Meta cur = load_meta(blockIdx.x);
  for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
    const Meta next = load_meta(tile + gridDim.x);
    if (cur.expert < 0) {
      cur = next;
      continue;
    }

    const bf16* src[kRowsPerThread];
    bool valid[kRowsPerThread];
#pragma unroll
    for (int i = 0; i < kRowsPerThread; ++i) {
      valid[i] = cur.ids[i] < p.num_rows;
      src[i] = p.a + int64_t(valid[i] ? cur.ids[i] / p.a_row_divisor : 0) * p.k + chunk * 8;
    }
    const int64_t stage_index = (int64_t(cur.expert) * n_tiles + tile % n_pairs * kTilesPerCta) * k_tiles;

    for (int kt = 0; kt < k_tiles; ++kt, pos.advance(Cfg::kStages)) {
      const int s = pos.stage;
      device::ptx::mbar_wait_parity(&empty[s], pos.phase ^ 1);
      uint8_t* stage = stages + s * Cfg::kStageBytes;

      if (tid == 0) {
        device::ptx::mbar_arrive_expect_tx(&full[s], Cfg::kStageTxBytes);
#pragma unroll
        for (int w = 0; w < kTilesPerCta; ++w) {
          const int64_t block = stage_index + w * k_tiles + kt;
          bulk_copy_g2s(stage + w * kWeightBytes, p.qweight + block * kWordsPerStage, kWeightBytes, &full[s]);
          bulk_copy_g2s(stage + Cfg::kScaleOffset + w * kScaleBytes, p.scales + block * kTileN, kScaleBytes, &full[s]);
          bulk_copy_g2s(stage + Cfg::kZeroOffset + w * kZeroBytes, p.zeros + block * kTileN, kZeroBytes, &full[s]);
        }
      }

      auto tokens = cute::make_tensor(
          cute::make_smem_ptr(reinterpret_cast<bf16*>(stage + Cfg::kTokenOffset)), typename Cfg::SmemLayoutTokens{});
#pragma unroll
      for (int i = 0; i < kRowsPerThread; ++i) {
        cp_async_16(&tokens(first_row + i * kRowStride, chunk * 8), src[i] + kt * kTileK, valid[i]);
      }
      cp_async_arrive_noinc(&full[s]);
    }
    cur = next;
  }
}

template <int kTokenBlock, bool kZeroUnrouted, bool kPingPong>
__device__ __forceinline__ void
consume(const Params& p, uint8_t* stages, bf16* epilogue, uint8_t* meta, uint64_t* full, uint64_t* empty) {
  using namespace cute;
  using Cfg = Config<kTokenBlock, kPingPong>;

  const int wg = threadIdx.x / 128 - 1;
  const int tid = threadIdx.x % 128;
  const int warp = tid / 32;
  const int g = tid % 32 / 4;
  const int row_lo = warp * 16 + g;  // within a 64-row atom; the hi row is + 8

  typename Cfg::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tid);
  // Only the shape matters: it sizes the register A fragment.
  auto a_shape = make_tensor(
      make_smem_ptr(static_cast<bf16*>(nullptr)), Layout<Shape<_64, Int<kTileK>>, Stride<Int<kTileK>, _1>>{});
  Tensor frag_a_lo = thr_mma.partition_fragment_A(a_shape);  // ((2,2,2), 1, 8)
  Tensor frag_a_hi = thr_mma.partition_fragment_A(a_shape);
  Tensor acc_lo = partition_fragment_C(tiled_mma, Shape<_64, Int<kTokenBlock>>{});
  Tensor acc_hi = partition_fragment_C(tiled_mma, Shape<_64, Int<kTokenBlock>>{});
  // The pair's second weight tile, which only a ping-pong warpgroup computes.
  Tensor acc_lo2 = partition_fragment_C(tiled_mma, Shape<_64, Int<kTokenBlock>>{});
  Tensor acc_hi2 = partition_fragment_C(tiled_mma, Shape<_64, Int<kTokenBlock>>{});
  Tensor acc_coord = thr_mma.partition_C(make_identity_tensor(Shape<_64, Int<kTokenBlock>>{}));

  bf16* staging = epilogue + wg * kTokenBlock * Cfg::kEpilogueStride;
  const int n_tiles = p.n / kTileN;
  const int n_pairs = n_tiles / kTilesPerCta;
  const int k_tiles = p.k / kTileK;
  const int num_tiles = (*p.num_tokens_post_padded / kTokenBlock) * n_pairs;

  int32_t* tile_ids = reinterpret_cast<int32_t*>(meta) + wg * 2 * kTokenBlock;
  float* tile_weights = reinterpret_cast<float*>(tile_ids + kTokenBlock);

  // Thread `tid` < kTokenBlock owns routed row `tid` of the tile; loaded one tile
  // ahead, and its top-k weight is fetched before the mainloop it is used after.
  struct Meta {
    int expert = -1;  // negative: no local expert, the tile is skipped
    int id = 0;
  };
  auto load_meta = [&](int tile) {
    Meta m;
    if (tile < num_tiles) {
      const int m_block = tile / n_pairs;
      m.expert = p.expert_ids[m_block];
      if (tid < kTokenBlock) m.id = p.sorted_token_ids[m_block * kTokenBlock + tid];
    }
    return m;
  };

  // Dequantises 64-row atom `atom` of the weight tile in slot `slot` of stage `s`
  // into `frag_a` and issues its wgmma batch into `acc` without waiting on it.
  auto issue_atom = [&](auto& frag_a, auto& acc, int s, int slot, int atom) {
    const uint8_t* stage = stages + s * Cfg::kStageBytes;
    const int row = atom * kAtomRows + row_lo;
    const __nv_bfloat16* scales =
        reinterpret_cast<const __nv_bfloat16*>(stage + Cfg::kScaleOffset + slot * kScaleBytes);
    const uint8_t* zeros = stage + Cfg::kZeroOffset + slot * kZeroBytes;
    const nv_bfloat162 scale_lo = __bfloat162bfloat162(scales[row]);
    const nv_bfloat162 scale_hi = __bfloat162bfloat162(scales[row + 8]);
    const nv_bfloat162 zero_lo = biased_zero(zeros[row]);
    const nv_bfloat162 zero_hi = biased_zero(zeros[row + 8]);

    Tensor frag_a_words = recast<uint32_t>(frag_a);
    const uint4* words = reinterpret_cast<const uint4*>(stage + slot * kWeightBytes) + atom * 2 * 128 + tid;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const uint4 q = words[half * 128];
      const uint32_t slices[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint32_t frag[4];
        dequant_fragment(slices[j], zero_lo, zero_hi, scale_lo, scale_hi, frag);
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          frag_a_words((half * 4 + j) * 4 + r) = frag[r];
        }
      }
    }

    Tensor tokens = make_tensor(
        make_smem_ptr(reinterpret_cast<const bf16*>(stage + Cfg::kTokenOffset)), typename Cfg::SmemLayoutTokens{});
    Tensor frag_b = thr_mma.partition_fragment_B(tokens);  // descriptors, (1, 1, 8)

    warpgroup_fence_operand(frag_a);
    warpgroup_arrive();
#pragma unroll
    for (int k16 = 0; k16 < size<2>(frag_a); ++k16) {
      cute::gemm(tiled_mma, frag_a(_, _, k16), frag_b(_, _, k16), acc);
    }
    warpgroup_commit_batch();
  };

  RingPos pos;
  Meta cur = load_meta(blockIdx.x);
  for (int tile = blockIdx.x, item = 0; tile < num_tiles; tile += gridDim.x, ++item) {
    const Meta next = load_meta(tile + gridDim.x);
    // The producer streams nothing for a block routed to no local expert; with
    // kZeroUnrouted its rows are stored as zeros, so every listed row is written.
    const bool has_expert = cur.expert >= 0;
    if (!has_expert && !kZeroUnrouted) {
      cur = next;
      continue;
    }
    // A ping-pong warpgroup takes every other pair; it still steps the stage ring
    // past the other warpgroup's stages.
    if (kPingPong && item % 2 != wg) {
      if (has_expert) {
        for (int kt = 0; kt < k_tiles; ++kt)
          pos.advance(Cfg::kStages);
      }
      cur = next;
      continue;
    }
    const bool row_valid = tid < kTokenBlock && cur.id < p.num_rows;
    float row_weight = row_valid && has_expert ? 1.0f : 0.0f;
    if (p.topk_weights != nullptr && row_valid && has_expert) row_weight = p.topk_weights[cur.id];

    clear(acc_lo);
    clear(acc_hi);
    warpgroup_fence_operand(acc_lo);
    warpgroup_fence_operand(acc_hi);
    if constexpr (kPingPong) {
      clear(acc_lo2);
      clear(acc_hi2);
      warpgroup_fence_operand(acc_lo2);
      warpgroup_fence_operand(acc_hi2);
    }
    if (has_expert) {
      // One wgmma batch stays in flight while the next 64-row atom dequantises, so
      // each atom has its own A fragment, and a stage is released once the
      // following stage's first batch is issued.
      int prev_stage = -1;
      for (int kt = 0; kt < k_tiles; ++kt, pos.advance(Cfg::kStages)) {
        const int s = pos.stage;
        device::ptx::mbar_wait_parity(&full[s], pos.phase);
        if constexpr (kPingPong) {
          issue_atom(frag_a_lo, acc_lo, s, 0, 0);
          warpgroup_wait<1>();
          if (prev_stage >= 0) device::ptx::mbar_arrive(&empty[prev_stage]);
          issue_atom(frag_a_hi, acc_hi, s, 0, 1);
          warpgroup_wait<1>();
          issue_atom(frag_a_lo, acc_lo2, s, 1, 0);
          warpgroup_wait<1>();
          issue_atom(frag_a_hi, acc_hi2, s, 1, 1);
          warpgroup_wait<1>();
        } else {
          issue_atom(frag_a_lo, acc_lo, s, wg, 0);
          warpgroup_wait<1>();
          if (prev_stage >= 0) device::ptx::mbar_arrive(&empty[prev_stage]);
          issue_atom(frag_a_hi, acc_hi, s, wg, 1);
          warpgroup_wait<1>();
        }
        prev_stage = s;
      }
      warpgroup_wait<0>();
      warpgroup_fence_operand(acc_lo);
      warpgroup_fence_operand(acc_hi);
      if constexpr (kPingPong) {
        warpgroup_fence_operand(acc_lo2);
        warpgroup_fence_operand(acc_hi2);
      }
      if (prev_stage >= 0) device::ptx::mbar_arrive(&empty[prev_stage]);
    }

    if (tid < kTokenBlock) {
      tile_ids[tid] = row_valid ? cur.id : -1;
      tile_weights[tid] = row_weight;
    }
    named_barrier_sync(1 + wg, 128);

    // Transpose through shared memory so each routed row leaves as 16B stores.
#pragma unroll
    for (int i = 0; i < size(acc_lo); ++i) {
      const int row = get<0>(acc_coord(i));
      const int token = get<1>(acc_coord(i));
      const float weight = tile_weights[token];
      bf16* staged = staging + token * Cfg::kEpilogueStride + row;
      staged[0] = bf16(acc_lo(i) * weight);
      staged[kAtomRows] = bf16(acc_hi(i) * weight);
      if constexpr (kPingPong) {
        staged[kTileN] = bf16(acc_lo2(i) * weight);
        staged[kTileN + kAtomRows] = bf16(acc_hi2(i) * weight);
      }
    }
    named_barrier_sync(1 + wg, 128);

    const int col = (tile % n_pairs) * kTilesPerCta * kTileN + (kPingPong ? 0 : wg * kTileN);
    constexpr int kParts = Cfg::kWarpgroupRows / 8;
    for (int ci = tid; ci < kTokenBlock * kParts; ci += 128) {
      const int token = ci / kParts;
      const int part = ci % kParts;
      const int id = tile_ids[token];
      if (id >= 0) {
        *reinterpret_cast<uint4*>(p.out + int64_t(id) * p.n + col + part * 8) =
            *reinterpret_cast<const uint4*>(staging + token * Cfg::kEpilogueStride + part * 8);
      }
    }
    named_barrier_sync(1 + wg, 128);
    cur = next;
  }
}

template <int kTokenBlock, bool kZeroUnrouted, bool kPingPong>
__global__ void __launch_bounds__(kThreads, 1) w4a16_moe_sm90_kernel(const __grid_constant__ Params p) {
  using Cfg = Config<kTokenBlock, kPingPong>;
  extern __shared__ uint8_t smem_raw[];
  // Offset from smem_raw rather than a rebuilt integer address, so the compiler
  // keeps these pointers in the shared window and emits LDS, not generic loads.
  uint8_t* stages = smem_raw + ((1024 - (device::ptx::to_shared(smem_raw) & 1023)) & 1023);
  bf16* epilogue = reinterpret_cast<bf16*>(stages + Cfg::kStages * Cfg::kStageBytes);
  uint8_t* meta = reinterpret_cast<uint8_t*>(epilogue) + Cfg::kEpilogueBytes;
  uint64_t* full = reinterpret_cast<uint64_t*>(meta + Cfg::kMetaBytes);
  uint64_t* empty = full + Cfg::kStages;

  if (threadIdx.x == 0) {
    for (int s = 0; s < Cfg::kStages; ++s) {
      device::ptx::mbar_init(&full[s], Cfg::kFullArrivals);
      // Only the warpgroup that owns a stage releases it under kPingPong.
      device::ptx::mbar_init(&empty[s], kPingPong ? kConsumerThreads / 2 : kConsumerThreads);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();

  if (threadIdx.x < kProducerThreads) {
    cutlass::arch::warpgroup_reg_dealloc<40>();
    produce<kTokenBlock, kPingPong>(p, stages, full, empty);
  } else {
    cutlass::arch::warpgroup_reg_alloc<232>();
    consume<kTokenBlock, kZeroUnrouted, kPingPong>(p, stages, epilogue, meta, full, empty);
  }
}

}  // namespace w4a16_moe_sm90

template <int kTokenBlock, bool kZeroUnrouted, bool kPingPong>
struct W4A16MoeSm90Kernel {
  using Cfg = w4a16_moe_sm90::Config<kTokenBlock, kPingPong>;

  static void
  run(const tvm::ffi::TensorView a,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView qweight,
      const tvm::ffi::TensorView scales,
      const tvm::ffi::TensorView zeros,
      const tvm::ffi::TensorView sorted_token_ids,
      const tvm::ffi::TensorView expert_ids,
      const tvm::ffi::TensorView num_tokens_post_padded,
      const tvm::ffi::TensorView topk_weights,
      int64_t a_row_divisor) {
    using namespace host;
    using namespace w4a16_moe_sm90;

    auto num_experts = SymbolicSize{"num_experts"};
    auto n_tiles = SymbolicSize{"n_tiles"};
    auto k_tiles = SymbolicSize{"k_tiles"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({num_experts, n_tiles, k_tiles, kWordsPerStage})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(qweight);
    TensorMatcher({num_experts, n_tiles, k_tiles, kTileN}).with_dtype<bf16_t>().with_device(device).verify(scales);
    TensorMatcher({num_experts, n_tiles, k_tiles, kTileN}).with_dtype<uint8_t>().with_device(device).verify(zeros);
    const int64_t k = k_tiles.unwrap() * kTileK;
    const int64_t n = n_tiles.unwrap() * kTileN;
    TensorMatcher({-1, k}).with_dtype<bf16_t>().with_device(device).verify(a);
    TensorMatcher({-1, n}).with_dtype<bf16_t>().with_device(device).verify(out);
    TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).verify(sorted_token_ids);
    TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).verify(expert_ids);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(num_tokens_post_padded);
    TensorMatcher({-1}).with_dtype<fp32_t>().with_device(device).verify(topk_weights);

    CHECK_HOST(n_tiles.unwrap() % kTilesPerCta == 0)
        << "w4a16_moe_sm90 needs N divisible by " << kTilesPerCta * kTileN << ", got " << n;
    const int64_t num_rows = out.size(0);
    CHECK_HOST(topk_weights.size(0) == 0 || topk_weights.size(0) == num_rows)
        << "topk_weights must be empty or hold one weight per routed row";
    CHECK_HOST(a_row_divisor >= 1) << "a_row_divisor must be positive, got " << a_row_divisor;

    const Params params{
        static_cast<const bf16*>(a.data_ptr()),
        static_cast<bf16*>(out.data_ptr()),
        static_cast<const uint32_t*>(qweight.data_ptr()),
        static_cast<const bf16*>(scales.data_ptr()),
        static_cast<const uint8_t*>(zeros.data_ptr()),
        static_cast<const int32_t*>(sorted_token_ids.data_ptr()),
        static_cast<const int32_t*>(expert_ids.data_ptr()),
        static_cast<const int32_t*>(num_tokens_post_padded.data_ptr()),
        topk_weights.size(0) == 0 ? nullptr : static_cast<const float*>(topk_weights.data_ptr()),
        static_cast<int>(num_rows),
        static_cast<int>(a_row_divisor),
        static_cast<int>(k),
        static_cast<int>(n),
    };

    const DLDevice dev = device.unwrap();
    int sms = 0;
    RuntimeDeviceCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev.device_id));
    // The tile count lives on the device; a CTA without work exits after setup.
    const int64_t max_tiles = sorted_token_ids.size(0) / kTokenBlock * (n_tiles.unwrap() / kTilesPerCta);
    const int grid = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(sms, max_tiles)));

    constexpr auto kernel = w4a16_moe_sm90_kernel<kTokenBlock, kZeroUnrouted, kPingPong>;
    [[maybe_unused]] static const auto _ = [] {
      RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg::kSmemBytes));
      return 0;
    }();
    LaunchKernel(grid, kThreads, dev, Cfg::kSmemBytes)(kernel, params);
  }
};

}  // namespace sglang
