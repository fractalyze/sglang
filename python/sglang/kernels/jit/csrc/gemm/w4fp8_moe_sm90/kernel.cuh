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

// Hopper grouped W4A8 MoE GEMM for large token blocks, swap-AB:
// out[r, n] = sum_g s_w[e(r)][n, g] * s_a[r / div, g] * sum_{k in g} (q - z)[e(r)][n, k] * a8[r / div, k].
//
// Reads the w4a16_moe_sm90 weight layout unchanged. Each consumer warpgroup turns
// q - z into exact e4m3 register A fragments (fragment.cuh) and runs FP8 wgmma
// against the routed tokens' e4m3 rows, which a producer warpgroup gathers into
// 128B-swizzled shared memory together with their per-group scales. One pipeline
// stage is one AWQ group, so every stage's FP32 partial is promoted into the
// running sum with the weight and token scales of that group.
//
// Activations come from w4fp8_moe_sm90.quantize_activations, in K_PERMUTE16 order.

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
#include <limits>

namespace sglang {

namespace w4fp8_moe_sm90 {

using bf16 = cute::bfloat16_t;
using e4m3 = cute::float_e4m3_t;

// The w4a16_moe_sm90 stage geometry; see W4A16MoeWeights in w4a16_moe_sm90.py.
inline constexpr int kTileN = 128;
inline constexpr int kTileK = 128;
inline constexpr int kWarpgroupRows = 64;
inline constexpr int kWordsPerStage = kTileN * kTileK / 8;

inline constexpr int kWeightBytes = kWordsPerStage * 4;
inline constexpr int kScaleBytes = kTileN * 2;
inline constexpr int kZeroBytes = kTileN;
inline constexpr int kChunksPerTokenRow = kTileK / 16;

inline constexpr int kProducerThreads = 128;
inline constexpr int kConsumerThreads = 256;
inline constexpr int kThreads = kProducerThreads + kConsumerThreads;
// Hardware limit for one CTA's dynamic shared memory on sm90.
inline constexpr int kSmemLimit = 227 * 1024;
// Arbitrary cap, as in w4a16_moe_sm90.
inline constexpr int kMaxStages = 12;

constexpr int round_up(int x, int m) {
  return (x + m - 1) / m * m;
}

template <int kTokenBlock>
struct Config {
  static_assert(kTokenBlock % 16 == 0 && kTokenBlock <= 128, "a token block is 16 to 128 rows");

  static constexpr int kTokenBytes = kTokenBlock * kTileK;
  // The token tile is the swizzled wgmma B operand and needs 1024B alignment.
  static constexpr int kTokenOffset = kWeightBytes;
  static constexpr int kTokenScaleOffset = kTokenOffset + kTokenBytes;
  static constexpr int kScaleOffset = kTokenScaleOffset + kTokenBlock * 4;
  static constexpr int kZeroOffset = kScaleOffset + kScaleBytes;
  static constexpr int kStageBytes = round_up(kZeroOffset + kZeroBytes, 1024);

  // Padded so a warp's column-wise accumulator stores spread over banks.
  static constexpr int kEpilogueStride = kWarpgroupRows + 8;
  static constexpr int kEpilogueBytes = 2 * kTokenBlock * kEpilogueStride * 2;
  // Per consumer warpgroup: the tile's routed row ids, then their top-k weights.
  static constexpr int kRowMetaBytes = 2 * kTokenBlock * 8;
  static constexpr int kBarrierBytes = 2 * kMaxStages * 8;
  // Dynamic smem is only 16B-aligned; the stages are realigned to 1024B in-kernel.
  static constexpr int kAlignSlack = 1024;

  static constexpr int kStages =
      std::min(kMaxStages, (kSmemLimit - kEpilogueBytes - kRowMetaBytes - kBarrierBytes - kAlignSlack) / kStageBytes);
  static_assert(kStages >= 3, "token block too wide for a three-stage pipeline");
  static constexpr int kSmemBytes =
      kAlignSlack + kStages * kStageBytes + kEpilogueBytes + kRowMetaBytes + kBarrierBytes;

  static constexpr int kStageTxBytes = kWeightBytes + kScaleBytes + kZeroBytes;
  // One expect_tx arrival plus one cp.async arrival per producer thread.
  static constexpr int kFullArrivals = 1 + kProducerThreads;

  using SmemLayoutTokens = decltype(cute::tile_to_shape(
      cute::GMMA::Layout_K_SW128_Atom<e4m3>{}, cute::Shape<cute::Int<kTokenBlock>, cute::Int<kTileK>>{}));
  using TiledMma = decltype(cute::make_tiled_mma(
      cute::SM90::GMMA::
          rs_op_selector<e4m3, e4m3, float, cute::Shape<cute::_64, cute::Int<kTokenBlock>, cute::_32>>()));
};

struct Params {
  const e4m3* a;
  const float* a_scales;  // [a rows, k / kTileK]
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

// Zero-fills when `valid` is false, so a padding row's partial is scaled to zero.
__device__ __forceinline__ void cp_async_4(void* dst, const void* src, bool valid) {
  asm volatile(
      "cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(device::ptx::to_shared(dst)), "l"(src), "r"(valid ? 4 : 0)
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
  constexpr int kRowStride = kProducerThreads / kChunksPerTokenRow;
  constexpr int kRowsPerThread = kTokenBlock / kRowStride;

  const int tid = threadIdx.x;
  const int chunk = tid % kChunksPerTokenRow;
  const int first_row = tid / kChunksPerTokenRow;
  // 128B-swizzled K-major rows are 128 bytes, with 16B chunk c of row r stored at
  // chunk c ^ (r % 8); this thread's rows step by kRowStride, a multiple of 8.
  static_assert(kTileK == 128 && kRowStride % 8 == 0, "one swizzle phase per thread");
  const int token_chunk_offset = (chunk ^ (first_row % 8)) * 16;

  const int n_tiles = p.n / kTileN;
  const int k_tiles = p.k / kTileK;
  const int num_tiles = (*p.num_tokens_post_padded / kTokenBlock) * n_tiles;

  int it = 0;
  for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
    const TileCoord c = tile_coord(p, tile, n_tiles);
    if (c.expert < 0) continue;

    // 32-bit row indices and a validity mask keep the gather state inside the
    // producer's register budget; the host checks that row offsets fit in int32.
    int a_row[kRowsPerThread];
    uint32_t valid = 0;
#pragma unroll
    for (int i = 0; i < kRowsPerThread; ++i) {
      const int id = p.sorted_token_ids[c.m_block * kTokenBlock + first_row + i * kRowStride];
      const bool in_range = id < p.num_rows;
      valid |= uint32_t(in_range) << i;
      a_row[i] = in_range ? id / p.a_row_divisor : 0;
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

      uint8_t* tokens = stage + Cfg::kTokenOffset + first_row * kTileK + token_chunk_offset;
      float* token_scales = reinterpret_cast<float*>(stage + Cfg::kTokenScaleOffset) + first_row;
      const e4m3* a = p.a + kt * kTileK + chunk * 16;
#pragma unroll
      for (int i = 0; i < kRowsPerThread; ++i) {
        const bool row_valid = (valid >> i) & 1;
        cp_async_16(tokens + i * kRowStride * kTileK, a + a_row[i] * p.k, row_valid);
        if (chunk == 0) {
          cp_async_4(token_scales + i * kRowStride, p.a_scales + a_row[i] * k_tiles + kt, row_valid);
        }
      }
      cp_async_arrive_noinc(&full[s]);
    }
  }
}

// Dequantises the stage's weights into this thread's e4m3 A fragments.
template <typename Fragment>
__device__ __forceinline__ void
dequant_stage(const uint8_t* stage, int zero_offset, int wg, int tid, int row_lo, Fragment& frag_a) {
  const half2 zero_lo = biased_zero(stage[zero_offset + row_lo]);
  const half2 zero_hi = biased_zero(stage[zero_offset + row_lo + 8]);
  auto frag_a_words = cute::recast<uint32_t>(frag_a);
  const uint4* words = reinterpret_cast<const uint4*>(stage) + wg * 2 * 128 + tid;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const uint4 q = words[half * 128];
    const uint32_t slices[4] = {q.x, q.y, q.z, q.w};
    // A k32 fragment is two consecutive k16 slices; PTX orders its words
    // (row g, row g + 8) of the first slice, then of the second.
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      uint32_t frag[2];
      dequant_fragment(slices[j], zero_lo, zero_hi, frag);
      const int word = (half * 2 + j / 2) * 4 + (j % 2) * 2;
      frag_a_words(word) = frag[0];
      frag_a_words(word + 1) = frag[1];
    }
  }
}

template <int kTokenBlock>
__device__ __forceinline__ void
consume(const Params& p, uint8_t* stages, bf16* epilogue, uint8_t* row_meta, uint64_t* full, uint64_t* empty) {
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
      make_smem_ptr(static_cast<e4m3*>(nullptr)), Layout<Shape<_64, Int<kTileK>>, Stride<Int<kTileK>, _1>>{});
  Tensor frag_a_even = thr_mma.partition_fragment_A(a_shape);  // ((4,2,2), 1, 4)
  Tensor frag_a_odd = thr_mma.partition_fragment_A(a_shape);
  Tensor partial = partition_fragment_C(tiled_mma, Shape<_64, Int<kTokenBlock>>{});
  Tensor acc = partition_fragment_C(tiled_mma, Shape<_64, Int<kTokenBlock>>{});
  Tensor acc_coord = thr_mma.partition_C(make_identity_tensor(Shape<_64, Int<kTokenBlock>>{}));

  bf16* staging = epilogue + wg * kTokenBlock * Cfg::kEpilogueStride;
  int* row_ids = reinterpret_cast<int*>(row_meta + wg * kTokenBlock * 8);
  float* row_weights = reinterpret_cast<float*>(row_ids + kTokenBlock);
  const int n_tiles = p.n / kTileN;
  const int k_tiles = p.k / kTileK;
  const int num_tiles = (*p.num_tokens_post_padded / kTokenBlock) * n_tiles;

  auto stage_ptr = [&](int it) { return stages + (it % Cfg::kStages) * Cfg::kStageBytes; };
  auto wait_full = [&](int it) { device::ptx::mbar_wait_parity(&full[it % Cfg::kStages], (it / Cfg::kStages) & 1); };

  // Issues stage `it`'s wgmma batch from `cur`, dequantises the next stage into
  // `next` while it runs, then promotes the partial with the group's scales.
  auto step = [&](auto& cur, auto& next, int it, bool has_next) {
    const uint8_t* stage = stage_ptr(it);
    Tensor tokens = make_tensor(
        make_smem_ptr(reinterpret_cast<const e4m3*>(stage + Cfg::kTokenOffset)), typename Cfg::SmemLayoutTokens{});
    Tensor frag_b = thr_mma.partition_fragment_B(tokens);  // descriptors, (1, 1, 4)

    warpgroup_fence_operand(cur);
    warpgroup_fence_operand(partial);
    warpgroup_arrive();
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
#pragma unroll
    for (int k32 = 0; k32 < size<2>(cur); ++k32) {
      cute::gemm(tiled_mma, cur(_, _, k32), frag_b(_, _, k32), partial);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_commit_batch();

    if (has_next) {
      wait_full(it + 1);
      dequant_stage(stage_ptr(it + 1), Cfg::kZeroOffset, wg, tid, row_lo, next);
    }

    const __nv_bfloat16* weight_scales = reinterpret_cast<const __nv_bfloat16*>(stage + Cfg::kScaleOffset);
    const float scale_lo = __bfloat162float(weight_scales[row_lo]);
    const float scale_hi = __bfloat162float(weight_scales[row_lo + 8]);
    const float* token_scales = reinterpret_cast<const float*>(stage + Cfg::kTokenScaleOffset);
    warpgroup_wait<0>();
    warpgroup_fence_operand(partial);
    warpgroup_fence_operand(cur);
#pragma unroll
    for (int i = 0; i < size(acc); ++i) {
      // A thread's accumulator rows are only row_lo and row_lo + 8 of its warpgroup.
      const float s_w = get<0>(acc_coord(i)) % 16 < 8 ? scale_lo : scale_hi;
      acc(i) += partial(i) * (s_w * token_scales[get<1>(acc_coord(i))]);
    }
    device::ptx::mbar_arrive(&empty[it % Cfg::kStages]);
  };

  int it = 0;
  for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
    const TileCoord c = tile_coord(p, tile, n_tiles);
    if (c.expert < 0) continue;

    // Issued before the main loop so the global loads land while it runs.
    const int block_base = c.m_block * kTokenBlock;
    if (tid < kTokenBlock) {
      const int id = p.sorted_token_ids[block_base + tid];
      const bool routed = id < p.num_rows;
      row_ids[tid] = routed ? id : -1;
      row_weights[tid] = p.topk_weights == nullptr ? 1.0f : (routed ? p.topk_weights[id] : 0.0f);
    }

    clear(acc);
    wait_full(it);
    dequant_stage(stage_ptr(it), Cfg::kZeroOffset, wg, tid, row_lo, frag_a_even);
    for (int kt = 0; kt < k_tiles; ++kt, ++it) {
      const bool has_next = kt + 1 < k_tiles;
      if (kt & 1) {
        step(frag_a_odd, frag_a_even, it, has_next);
      } else {
        step(frag_a_even, frag_a_odd, it, has_next);
      }
    }

    // Transpose through shared memory so each routed row leaves as 16B stores.
    named_barrier_sync(1 + wg, 128);
#pragma unroll
    for (int i = 0; i < size(acc); ++i) {
      const int row = get<0>(acc_coord(i));
      const int token = get<1>(acc_coord(i));
      staging[token * Cfg::kEpilogueStride + row] = bf16(acc(i) * row_weights[token]);
    }
    named_barrier_sync(1 + wg, 128);

    const int col = c.n_tile * kTileN + wg * kWarpgroupRows;
#pragma unroll
    for (int ci = tid; ci < kTokenBlock * kWarpgroupRows / 8; ci += 128) {
      const int token = ci / (kWarpgroupRows / 8);
      const int part = ci % (kWarpgroupRows / 8);
      const int id = row_ids[token];
      if (id >= 0) {
        *reinterpret_cast<uint4*>(p.out + int64_t(id) * p.n + col + part * 8) =
            *reinterpret_cast<const uint4*>(staging + token * Cfg::kEpilogueStride + part * 8);
      }
    }
    named_barrier_sync(1 + wg, 128);
  }
}

template <int kTokenBlock>
__global__ void __launch_bounds__(kThreads, 1) w4fp8_moe_sm90_kernel(const __grid_constant__ Params p) {
  using Cfg = Config<kTokenBlock>;
  extern __shared__ uint8_t smem_raw[];
  // Pointer arithmetic, not an integer round trip, so loads stay in the shared space.
  uint8_t* stages = smem_raw + ((1024 - device::ptx::to_shared(smem_raw) % 1024) % 1024);
  bf16* epilogue = reinterpret_cast<bf16*>(stages + Cfg::kStages * Cfg::kStageBytes);
  uint8_t* row_meta = reinterpret_cast<uint8_t*>(epilogue) + Cfg::kEpilogueBytes;
  uint64_t* full = reinterpret_cast<uint64_t*>(row_meta + Cfg::kRowMetaBytes);
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
    consume<kTokenBlock>(p, stages, epilogue, row_meta, full, empty);
  }
}

}  // namespace w4fp8_moe_sm90

template <int kTokenBlock>
struct W4Fp8MoeSm90Kernel {
  using Cfg = w4fp8_moe_sm90::Config<kTokenBlock>;

  static void
  run(const tvm::ffi::TensorView a,
      const tvm::ffi::TensorView a_scales,
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
    using namespace w4fp8_moe_sm90;

    auto num_experts = SymbolicSize{"num_experts"};
    auto n_tiles = SymbolicSize{"n_tiles"};
    auto k_tiles = SymbolicSize{"k_tiles"};
    auto a_rows = SymbolicSize{"a_rows"};
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
    TensorMatcher({a_rows, k}).with_dtype<fp8_e4m3_t>().with_device(device).verify(a);
    TensorMatcher({a_rows, k_tiles}).with_dtype<fp32_t>().with_device(device).verify(a_scales);
    TensorMatcher({-1, n}).with_dtype<bf16_t>().with_device(device).verify(out);
    TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).verify(sorted_token_ids);
    TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).verify(expert_ids);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(num_tokens_post_padded);
    TensorMatcher({-1}).with_dtype<fp32_t>().with_device(device).verify(topk_weights);

    const int64_t num_rows = out.size(0);
    CHECK_HOST(topk_weights.size(0) == 0 || topk_weights.size(0) == num_rows)
        << "topk_weights must be empty or hold one weight per routed row";
    CHECK_HOST(a_row_divisor >= 1) << "a_row_divisor must be positive, got " << a_row_divisor;
    CHECK_HOST(a_rows.unwrap() * k <= std::numeric_limits<int32_t>::max())
        << "activation offsets must fit in int32, got " << a_rows.unwrap() << " x " << k;

    const Params params{
        static_cast<const e4m3*>(a.data_ptr()),
        static_cast<const float*>(a_scales.data_ptr()),
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

    constexpr auto kernel = w4fp8_moe_sm90_kernel<kTokenBlock>;
    [[maybe_unused]] static const auto _ = [] {
      RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg::kSmemBytes));
      return 0;
    }();
    LaunchKernel(grid, kThreads, dev, Cfg::kSmemBytes)(kernel, params);
  }
};

}  // namespace sglang
