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
// The dequantised weight tile is the wgmma A operand (register-sourced, 64 output
// rows per consumer warpgroup); the routed tokens of one moe_align block are the
// wgmma N operand, gathered row by row into 128B-swizzled shared memory. A producer
// warpgroup streams weights, scales and zeros with bulk async copies and gathers
// tokens with cp.async; consumers dequantise in registers. Work is persistent over
// (token block, 128-row tile) pairs, n-tile fastest so concurrent CTAs share the
// block's tokens in L2.
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
inline constexpr int kWarpgroupRows = 64;
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

template <int kTokenBlock>
struct Config {
  static_assert(kTokenBlock % 8 == 0 && kTokenBlock <= 256, "wgmma N must be a multiple of 8 up to 256");

  static constexpr int kTokenBytes = kTokenBlock * kTileK * 2;
  // The token tile is the swizzled wgmma B operand and needs 1024B alignment.
  static constexpr int kTokenOffset = kWeightBytes;
  static constexpr int kScaleOffset = kTokenOffset + kTokenBytes;
  static constexpr int kZeroOffset = kScaleOffset + kScaleBytes;
  static constexpr int kStageBytes = round_up(kZeroOffset + kZeroBytes, 1024);

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

  // A stage is exactly one AWQ group, so a row's scale is constant within it:
  // wgmma accumulates exact q - z products into a per-stage partial, folded into
  // the result with one fp32 FMA per value. That replaces the per-weight bf16
  // scale multiply; wider blocks lack registers for the double-buffered partial.
  static constexpr bool kScaleOnAccumulator = kTokenBlock <= 32;

  static constexpr int kStageTxBytes = kWeightBytes + kScaleBytes + kZeroBytes;
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

template <int kTokenBlock>
__device__ __forceinline__ void produce(const Params& p, uint8_t* stages, uint64_t* full, uint64_t* empty) {
  using Cfg = Config<kTokenBlock>;
  constexpr int kRowsPerThread = kTokenBlock * kChunksPerTokenRow / kProducerThreads;
  static_assert(kRowsPerThread >= 1, "every producer thread gathers at least one chunk");

  const int tid = threadIdx.x;
  const int chunk = tid % kChunksPerTokenRow;
  const int first_row = tid / kChunksPerTokenRow;
  constexpr int kRowStride = kProducerThreads / kChunksPerTokenRow;

  const int n_tiles = p.n / kTileN;
  const int k_tiles = p.k / kTileK;
  const int num_tiles = (*p.num_tokens_post_padded / kTokenBlock) * n_tiles;

  // Global reads a tile needs before its first copy; loaded one tile ahead so
  // their latency overlaps the previous tile's stages.
  struct Meta {
    int expert = -1;  // negative: no local expert, the tile is skipped
    int ids[kRowsPerThread];
  };
  auto load_meta = [&](int tile) {
    Meta m;
    if (tile < num_tiles) {
      const int m_block = tile / n_tiles;
      m.expert = p.expert_ids[m_block];
#pragma unroll
      for (int i = 0; i < kRowsPerThread; ++i) {
        m.ids[i] = p.sorted_token_ids[m_block * kTokenBlock + first_row + i * kRowStride];
      }
    }
    return m;
  };

  int it = 0;
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
    const int64_t stage_index = (int64_t(cur.expert) * n_tiles + tile % n_tiles) * k_tiles;

    for (int kt = 0; kt < k_tiles; ++kt, ++it) {
      const int s = it % Cfg::kStages;
      device::ptx::mbar_wait_parity(&empty[s], ((it / Cfg::kStages) & 1) ^ 1);
      uint8_t* stage = stages + s * Cfg::kStageBytes;

      if (tid == 0) {
        const int64_t block = stage_index + kt;
        device::ptx::mbar_arrive_expect_tx(&full[s], Cfg::kStageTxBytes);
        bulk_copy_g2s(stage, p.qweight + block * kWordsPerStage, kWeightBytes, &full[s]);
        bulk_copy_g2s(stage + Cfg::kScaleOffset, p.scales + block * kTileN, kScaleBytes, &full[s]);
        bulk_copy_g2s(stage + Cfg::kZeroOffset, p.zeros + block * kTileN, kZeroBytes, &full[s]);
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

template <int kTokenBlock>
__device__ __forceinline__ void
consume(const Params& p, uint8_t* stages, bf16* epilogue, uint8_t* meta, uint64_t* full, uint64_t* empty) {
  using namespace cute;
  using Cfg = Config<kTokenBlock>;

  const int wg = threadIdx.x / 128 - 1;
  const int tid = threadIdx.x % 128;
  const int warp = tid / 32;
  const int g = tid % 32 / 4;
  const int row_lo = wg * kWarpgroupRows + warp * 16 + g;

  typename Cfg::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tid);
  // Only the shape matters: it sizes the register A fragment.
  auto a_shape = make_tensor(
      make_smem_ptr(static_cast<bf16*>(nullptr)), Layout<Shape<_64, Int<kTileK>>, Stride<Int<kTileK>, _1>>{});
  Tensor frag_a_even = thr_mma.partition_fragment_A(a_shape);  // ((2,2,2), 1, 8)
  Tensor frag_a_odd = thr_mma.partition_fragment_A(a_shape);
  Tensor acc = partition_fragment_C(tiled_mma, Shape<_64, Int<kTokenBlock>>{});
  Tensor part_even = partition_fragment_C(tiled_mma, Shape<_64, Int<kTokenBlock>>{});
  Tensor part_odd = partition_fragment_C(tiled_mma, Shape<_64, Int<kTokenBlock>>{});
  Tensor acc_coord = thr_mma.partition_C(make_identity_tensor(Shape<_64, Int<kTokenBlock>>{}));

  bf16* staging = epilogue + wg * kTokenBlock * Cfg::kEpilogueStride;
  const int n_tiles = p.n / kTileN;
  const int k_tiles = p.k / kTileK;
  const int num_tiles = (*p.num_tokens_post_padded / kTokenBlock) * n_tiles;

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
      const int m_block = tile / n_tiles;
      m.expert = p.expert_ids[m_block];
      if (tid < kTokenBlock) m.id = p.sorted_token_ids[m_block * kTokenBlock + tid];
    }
    return m;
  };

  int it = 0;
  Meta cur = load_meta(blockIdx.x);
  for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
    const Meta next = load_meta(tile + gridDim.x);
    if (cur.expert < 0) {
      cur = next;
      continue;
    }
    const bool row_valid = tid < kTokenBlock && cur.id < p.num_rows;
    float row_weight = row_valid ? 1.0f : 0.0f;
    if (p.topk_weights != nullptr && row_valid) row_weight = p.topk_weights[cur.id];

    // Dequantises stage `s` into `frag_a` and issues its wgmma batch without
    // waiting on it, into `part` with the row scales returned in `scale`, or
    // straight into `acc` with the weights already scaled.
    auto issue_stage = [&](auto& frag_a, auto& part, float* scale, int s) {
      const uint8_t* stage = stages + s * Cfg::kStageBytes;
      const __nv_bfloat16* scales = reinterpret_cast<const __nv_bfloat16*>(stage + Cfg::kScaleOffset);
      const uint8_t* zeros = stage + Cfg::kZeroOffset;
      const nv_bfloat162 zero_lo = biased_zero(zeros[row_lo]);
      const nv_bfloat162 zero_hi = biased_zero(zeros[row_lo + 8]);
      const __nv_bfloat16 s_lo = scales[row_lo];
      const __nv_bfloat16 s_hi = scales[row_lo + 8];
      scale[0] = __bfloat162float(s_lo);
      scale[1] = __bfloat162float(s_hi);

      Tensor frag_a_words = recast<uint32_t>(frag_a);
      const uint4* words = reinterpret_cast<const uint4*>(stage) + wg * 2 * 128 + tid;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const uint4 q = words[half * 128];
        const uint32_t slices[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t frag[4];
          if constexpr (Cfg::kScaleOnAccumulator) {
            dequant_fragment_codes(slices[j], zero_lo, zero_hi, reinterpret_cast<nv_bfloat162*>(frag));
          } else {
            dequant_fragment(slices[j], zero_lo, zero_hi, __bfloat162bfloat162(s_lo), __bfloat162bfloat162(s_hi), frag);
          }
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            frag_a_words((half * 4 + j) * 4 + r) = frag[r];
          }
        }
      }

      Tensor tokens = make_tensor(
          make_smem_ptr(reinterpret_cast<const bf16*>(stage + Cfg::kTokenOffset)), typename Cfg::SmemLayoutTokens{});
      Tensor frag_b = thr_mma.partition_fragment_B(tokens);  // descriptors, (1, 1, 8)

      auto& target = [&]() -> auto& {
        if constexpr (Cfg::kScaleOnAccumulator) {
          return part;
        } else {
          return acc;
        }
      }();
      warpgroup_fence_operand(frag_a);
      warpgroup_fence_operand(target);
      warpgroup_arrive();
      tiled_mma.accumulate_ = Cfg::kScaleOnAccumulator ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
#pragma unroll
      for (int k16 = 0; k16 < size<2>(frag_a); ++k16) {
        cute::gemm(tiled_mma, frag_a(_, _, k16), frag_b(_, _, k16), target);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();
    };

    // acc += scale[row] * part; C values alternate row g, g + 8 every second element.
    auto fold = [&](auto& part, const float* scale) {
      if constexpr (Cfg::kScaleOnAccumulator) {
        warpgroup_fence_operand(part);
#pragma unroll
        for (int i = 0; i < size(acc); ++i) {
          acc(i) = fmaf(scale[(i >> 1) & 1], part(i), acc(i));
        }
      }
    };

    clear(acc);
    warpgroup_fence_operand(acc);
    // One wgmma batch stays in flight while the next stage dequantises, so the
    // register A fragments and partials alternate and a stage is released one
    // step late.
    float scale_even[2], scale_odd[2];
    int prev_stage = -1;
    for (int kt = 0; kt < k_tiles; ++kt, ++it) {
      const int s = it % Cfg::kStages;
      device::ptx::mbar_wait_parity(&full[s], (it / Cfg::kStages) & 1);
      if (it & 1) {
        issue_stage(frag_a_odd, part_odd, scale_odd, s);
      } else {
        issue_stage(frag_a_even, part_even, scale_even, s);
      }
      warpgroup_wait<1>();
      if (prev_stage >= 0) {
        if (it & 1) {
          fold(part_even, scale_even);
        } else {
          fold(part_odd, scale_odd);
        }
        device::ptx::mbar_arrive(&empty[prev_stage]);
      }
      prev_stage = s;
    }
    warpgroup_wait<0>();
    if (prev_stage >= 0) {
      if ((it - 1) & 1) {
        fold(part_odd, scale_odd);
      } else {
        fold(part_even, scale_even);
      }
      device::ptx::mbar_arrive(&empty[prev_stage]);
    }
    warpgroup_fence_operand(acc);

    if (tid < kTokenBlock) {
      tile_ids[tid] = row_valid ? cur.id : -1;
      tile_weights[tid] = row_weight;
    }
    named_barrier_sync(1 + wg, 128);

    // Transpose through shared memory so each routed row leaves as 16B stores.
#pragma unroll
    for (int i = 0; i < size(acc); ++i) {
      const int row = get<0>(acc_coord(i));
      const int token = get<1>(acc_coord(i));
      staging[token * Cfg::kEpilogueStride + row] = bf16(acc(i) * tile_weights[token]);
    }
    named_barrier_sync(1 + wg, 128);

    const int col = (tile % n_tiles) * kTileN + wg * kWarpgroupRows;
    for (int ci = tid; ci < kTokenBlock * kWarpgroupRows / 8; ci += 128) {
      const int token = ci / (kWarpgroupRows / 8);
      const int part = ci % (kWarpgroupRows / 8);
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

template <int kTokenBlock>
__global__ void __launch_bounds__(kThreads, 1) w4a16_moe_sm90_kernel(const __grid_constant__ Params p) {
  using Cfg = Config<kTokenBlock>;
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
      device::ptx::mbar_init(&empty[s], kConsumerThreads);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();

  if (threadIdx.x < kProducerThreads) {
    cutlass::arch::warpgroup_reg_dealloc<40>();
    produce<kTokenBlock>(p, stages, full, empty);
  } else {
    cutlass::arch::warpgroup_reg_alloc<232>();
    consume<kTokenBlock>(p, stages, epilogue, meta, full, empty);
  }
}

}  // namespace w4a16_moe_sm90

template <int kTokenBlock>
struct W4A16MoeSm90Kernel {
  using Cfg = w4a16_moe_sm90::Config<kTokenBlock>;

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
    const int64_t max_tiles = sorted_token_ids.size(0) / kTokenBlock * n_tiles.unwrap();
    const int grid = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(sms, max_tiles)));

    constexpr auto kernel = w4a16_moe_sm90_kernel<kTokenBlock>;
    [[maybe_unused]] static const auto _ = [] {
      RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg::kSmemBytes));
      return 0;
    }();
    LaunchKernel(grid, kThreads, dev, Cfg::kSmemBytes)(kernel, params);
  }
};

}  // namespace sglang
