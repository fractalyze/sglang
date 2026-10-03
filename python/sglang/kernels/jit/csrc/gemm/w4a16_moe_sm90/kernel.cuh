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
  static constexpr int kBarrierBytes = 2 * kMaxStages * 8;
  // Dynamic smem is only 16B-aligned; the stages are realigned to 1024B in-kernel.
  static constexpr int kAlignSlack = 1024;

  static constexpr int kStages =
      std::min(kMaxStages, (kSmemLimit - kEpilogueBytes - kBarrierBytes - kAlignSlack) / kStageBytes);
  static_assert(kStages >= 3, "token block too wide for a three-stage pipeline");
  static constexpr int kSmemBytes = kAlignSlack + kStages * kStageBytes + kEpilogueBytes + kBarrierBytes;

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

struct TileCoord {
  int m_block;
  int n_tile;
  int expert;  // negative: the block belongs to no local expert and is skipped
};

__device__ __forceinline__ TileCoord tile_coord(const Params& p, int tile, int n_tiles) {
  TileCoord c;
  c.m_block = tile / n_tiles;
  c.n_tile = tile % n_tiles;
  c.expert = p.expert_ids[c.m_block];
  return c;
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

  int it = 0;
  for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
    const TileCoord c = tile_coord(p, tile, n_tiles);
    if (c.expert < 0) continue;

    const bf16* src[kRowsPerThread];
    bool valid[kRowsPerThread];
#pragma unroll
    for (int i = 0; i < kRowsPerThread; ++i) {
      const int id = p.sorted_token_ids[c.m_block * kTokenBlock + first_row + i * kRowStride];
      valid[i] = id < p.num_rows;
      src[i] = p.a + int64_t(valid[i] ? id / p.a_row_divisor : 0) * p.k + chunk * 8;
    }
    const int64_t stage_index = (int64_t(c.expert) * n_tiles + c.n_tile) * k_tiles;

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
  }
}

template <int kTokenBlock>
__device__ __forceinline__ void
consume(const Params& p, uint8_t* stages, bf16* epilogue, uint64_t* full, uint64_t* empty) {
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
  Tensor frag_a = thr_mma.partition_fragment_A(a_shape);  // ((2,2,2), 1, 8)
  Tensor frag_a_words = recast<uint32_t>(frag_a);
  Tensor acc = partition_fragment_C(tiled_mma, Shape<_64, Int<kTokenBlock>>{});
  Tensor acc_coord = thr_mma.partition_C(make_identity_tensor(Shape<_64, Int<kTokenBlock>>{}));

  bf16* staging = epilogue + wg * kTokenBlock * Cfg::kEpilogueStride;
  const int n_tiles = p.n / kTileN;
  const int k_tiles = p.k / kTileK;
  const int num_tiles = (*p.num_tokens_post_padded / kTokenBlock) * n_tiles;

  int it = 0;
  for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
    const TileCoord c = tile_coord(p, tile, n_tiles);
    if (c.expert < 0) continue;

    clear(acc);
    for (int kt = 0; kt < k_tiles; ++kt, ++it) {
      const int s = it % Cfg::kStages;
      device::ptx::mbar_wait_parity(&full[s], (it / Cfg::kStages) & 1);
      const uint8_t* stage = stages + s * Cfg::kStageBytes;

      const __nv_bfloat16* scales = reinterpret_cast<const __nv_bfloat16*>(stage + Cfg::kScaleOffset);
      const uint8_t* zeros = stage + Cfg::kZeroOffset;
      const nv_bfloat162 scale_lo = __bfloat162bfloat162(scales[row_lo]);
      const nv_bfloat162 scale_hi = __bfloat162bfloat162(scales[row_lo + 8]);
      const nv_bfloat162 zero_lo = biased_zero(zeros[row_lo]);
      const nv_bfloat162 zero_hi = biased_zero(zeros[row_lo + 8]);

      const uint4* words = reinterpret_cast<const uint4*>(stage) + wg * 2 * 128 + tid;
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
      warpgroup_fence_operand(acc);
      warpgroup_arrive();
#pragma unroll
      for (int k16 = 0; k16 < size<2>(frag_a); ++k16) {
        cute::gemm(tiled_mma, frag_a(_, _, k16), frag_b(_, _, k16), acc);
      }
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(acc);
      warpgroup_fence_operand(frag_a);
      device::ptx::mbar_arrive(&empty[s]);
    }

    // Transpose through shared memory so each routed row leaves as 16B stores.
    const int block_base = c.m_block * kTokenBlock;
#pragma unroll
    for (int i = 0; i < size(acc); ++i) {
      const int row = get<0>(acc_coord(i));
      const int token = get<1>(acc_coord(i));
      float v = acc(i);
      if (p.topk_weights != nullptr) {
        const int id = p.sorted_token_ids[block_base + token];
        v *= id < p.num_rows ? p.topk_weights[id] : 0.0f;
      }
      staging[token * Cfg::kEpilogueStride + row] = bf16(v);
    }
    named_barrier_sync(1 + wg, 128);

    const int col = c.n_tile * kTileN + wg * kWarpgroupRows;
    for (int ci = tid; ci < kTokenBlock * kWarpgroupRows / 8; ci += 128) {
      const int token = ci / (kWarpgroupRows / 8);
      const int part = ci % (kWarpgroupRows / 8);
      const int id = p.sorted_token_ids[block_base + token];
      if (id < p.num_rows) {
        *reinterpret_cast<uint4*>(p.out + int64_t(id) * p.n + col + part * 8) =
            *reinterpret_cast<const uint4*>(staging + token * Cfg::kEpilogueStride + part * 8);
      }
    }
    named_barrier_sync(1 + wg, 128);
  }
}

template <int kTokenBlock>
__global__ void __launch_bounds__(kThreads, 1) w4a16_moe_sm90_kernel(const __grid_constant__ Params p) {
  using Cfg = Config<kTokenBlock>;
  extern __shared__ uint8_t smem_raw[];
  uint8_t* stages = reinterpret_cast<uint8_t*>(round_up(reinterpret_cast<uintptr_t>(smem_raw), 1024));
  bf16* epilogue = reinterpret_cast<bf16*>(stages + Cfg::kStages * Cfg::kStageBytes);
  uint64_t* full = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(epilogue) + Cfg::kEpilogueBytes);
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
    consume<kTokenBlock>(p, stages, epilogue, full, empty);
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
