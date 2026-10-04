/// \file w4a16_sm90.cuh
/// \brief W4A16 GEMM for M <= 192 over Marlin-repacked AWQ weights (4-bit,
/// group 128, zero point): `out[m, n] = sum_k a[m, k] * dequant(w)[k, n]`, SM90.
///
/// It reads the tensors `AWQMarlinLinearKernel` already holds -- the
/// `awq_marlin_repack` weight, `marlin_permute_scales` scales and
/// `awq_to_marlin_zero_points` zeros -- and dequantizes with Marlin's own
/// `dequant`, so the packed layout keeps one definition.
///
/// Swap-AB wgmma: the weights are the A operand (M axis = 64 output columns),
/// taken from registers; the activations are B (N axis = tokens, padded to the
/// token tile) from shared memory. Marlin packs one 16x64 k-tile per 32 lanes so
/// that word j of lane l, dequantized, is lane l's mma B-fragment for column
/// subtile j; read as rows, the two halves of that fragment are exactly lane l's
/// slice of the wgmma m64k16 register-A fragment of warp j. So warp j of a
/// warpgroup takes word j and issues the warpgroup's MMA with no shuffle.
///
/// Schedules, chosen per shape by kClusterK. The work is (column unit,
/// quantization group) pairs, a column unit being kTiles Marlin tiles that share
/// one activation tile.
///   - Unsplit (1): one CTA per unit, in as many waves as it takes.
///   - Cluster (2-8): one cluster per unit; its CTAs split the groups and reduce
///     through distributed shared memory, each rank owning a slice of columns.
///   - Stream-K (0): a persistent grid splits the pairs into one contiguous range
///     per CTA. A unit spanning several CTAs is finished by the CTA holding its
///     last group: every other CTA on it writes an fp32 partial to a workspace
///     slot and raises its flag, and the finisher sums the partials. Each CTA
///     walks its range back to front, so the segment it finishes comes last and
///     it only ever waits on lower-numbered CTAs, which never wait on it.
/// Every reduction sums in a fixed order, so results are deterministic.
///
/// Per CTA: one producer warp streams groups (the unit's weight k-tiles, scales
/// and zeros, and the matching 128 activation columns) into an mbarrier ring;
/// each column tile has kPing consumer warpgroups taking alternate groups, so one
/// dequantizes while another's wgmmas run.

#pragma once

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/cluster.cuh>
#include <sgl_kernel/mbarrier.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tma.cuh>
#include <sgl_kernel/utils.cuh>

#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/arch/mma_sm90_gmma_ext.hpp>
#include <dlpack/dlpack.h>
#include <sgl_kernel/scalar_type.hpp>
#include <tvm/ffi/container/tensor.h>

#include "marlin/dequant.h"
#include <algorithm>
#include <cstdint>
#include <cuda.h>
#include <type_traits>
#include <utility>

namespace sglang::w4a16_sm90 {

// AWQ group 128 = 8 Marlin k-tiles of 16.
inline constexpr int kGroupSize = 128;
inline constexpr int kGroupKTiles = kGroupSize / 16;
// Marlin's repack tile: 16 k-rows x 64 columns, one int4 per lane.
inline constexpr int kTileN = 64;
inline constexpr int kTileBytes = 512;
inline constexpr int kWarpGroupThreads = 128;

template <int kN>
struct Gmma;
#define SGL_W4A16_GMMA(N)                                                                                         \
  template <>                                                                                                     \
  struct Gmma<N> {                                                                                                \
    using Op =                                                                                                    \
        cute::SM90::GMMA::MMA_64x##N##x16_F32BF16BF16_RS<cute::SM90::GMMA::Major::K, cute::SM90::GMMA::Major::K>; \
  };
SGL_W4A16_GMMA(8)
SGL_W4A16_GMMA(16)
SGL_W4A16_GMMA(32)
SGL_W4A16_GMMA(48)
SGL_W4A16_GMMA(64)
SGL_W4A16_GMMA(96)
SGL_W4A16_GMMA(128)
SGL_W4A16_GMMA(192)
#undef SGL_W4A16_GMMA

/// \brief Compile-time shape of one launch.
///
/// \tparam kN_      token tile (wgmma N), a multiple of 8
/// \tparam kTiles_  Marlin column tiles per CTA work unit; they share the
///                  unit's activation tile
/// \tparam kPing_   consumer warpgroups per column tile, taking alternate
///                  groups so one dequantizes while another's wgmmas run
/// \tparam kClusterK_ split-K schedule: 0 is a stream-K grid with a global fixup;
///                    1 is one CTA per unit, unsplit; above 1, the CTAs of a
///                    cluster split one unit's groups and reduce through
///                    distributed shared memory
/// \tparam kStages_ quantization groups in flight per CTA
template <int kN_, int kTiles_, int kPing_, int kClusterK_, int kStages_>
struct Trait {
  static constexpr int kN = kN_;
  static constexpr int kTiles = kTiles_;
  static constexpr int kPing = kPing_;
  static constexpr int kClusterK = kClusterK_;
  static constexpr bool kStreamK = kClusterK == 0;
  static constexpr bool kClusterReduce = kClusterK > 1;
  static constexpr int kClusterDim = kStreamK ? 1 : kClusterK;
  static constexpr int kStages = kStages_;
  static constexpr int kAccRegs = kN / 2;
  static constexpr int kCtaN = kTileN * kTiles;

  static constexpr int kConsumerThreads = kTiles * kPing * kWarpGroupThreads;
  // Threads of the warpgroups that finish segments, one per column tile.
  static constexpr int kFinisherThreads = kTiles * kWarpGroupThreads;
  static constexpr int kProducerWarp = kConsumerThreads / device::kWarpThreads;
  static constexpr int kThreads = kConsumerThreads + device::kWarpThreads;

  // One stage = one quantization group: activations as two 128B-swizzled
  // K-major atoms [k / 64][token][64] bf16 (first, since swizzled TMA
  // destinations must be 1024-byte aligned), the unit's weight k-tiles
  // [k-tile][column tile][512 B], scales [kCtaN] bf16 and zeros [kCtaN / 8] int32.
  static constexpr uint32_t kActAtomK = 64;
  static constexpr uint32_t kActAtomBytes = kN * kActAtomK * 2;
  static constexpr uint32_t kActBytes = (kGroupSize / kActAtomK) * kActAtomBytes;
  static constexpr uint32_t kWeightBytes = kGroupKTiles * kTiles * kTileBytes;
  static constexpr uint32_t kScaleBytes = kCtaN * 2;
  static constexpr uint32_t kZeroBytes = kCtaN / 2;
  static constexpr uint32_t kWeightOffset = kActBytes;
  static constexpr uint32_t kScaleOffset = kWeightOffset + kWeightBytes;
  static constexpr uint32_t kZeroOffset = kScaleOffset + kScaleBytes;
  static constexpr uint32_t kStageTxBytes = kZeroOffset + kZeroBytes;
  static constexpr uint32_t kStageBytes = (kStageTxBytes + 1023) / 1024 * 1024;

  // Where a column tile's second warpgroup hands its accumulators to the first,
  // [column tile][register][thread] so both sides access it conflict-free.
  static constexpr uint32_t kHandoffBytes = kPing > 1 ? kTiles* kAccRegs* kWarpGroupThreads * sizeof(float) : 0;
  // Cluster mode: CTA `rank` owns 1/kClusterK of the unit's columns and gathers
  // every rank's partial for them, [rank][owned column][token] fp32.
  static constexpr uint32_t kOwnedCols = kCtaN / kClusterDim;
  static constexpr uint32_t kInboxBytes = kClusterReduce ? kCtaN * kN * sizeof(float) : 0;
  static constexpr uint32_t kBarrierBytes = 1024;
  static constexpr uint32_t kSmemBytes = kBarrierBytes + kStages * kStageBytes + kHandoffBytes + kInboxBytes;
  // One CTA's partial unit in the workspace, in the same layout as the handoff.
  static constexpr uint32_t kPartialFloats = kTiles * kAccRegs * kWarpGroupThreads;

  static_assert(kN % 8 == 0 && kN <= 192);
  static_assert(kPing == 1 || kPing == 2);
  static_assert(kClusterK >= 0 && kClusterK <= 8 && kOwnedCols % 8 == 0 && kCtaN % kClusterDim == 0);
  static_assert(kThreads <= 1024);
  // Accumulators plus one group's A fragments and addressing per thread; the
  // launch bounds hold ptxas to it so that small CTAs stay several to an SM.
  // Larger CTAs are held to one or two per SM by shared memory anyway, and a
  // tighter cap only makes them spill.
  static constexpr int kRegBudget = kAccRegs + 64;
  static_assert(kRegBudget <= 65536 / kThreads, "token tile too wide for this many warpgroups");
  static constexpr int kMinBlocksPerSm = kThreads >= 256 ? 1 : 65536 / (kThreads * kRegBudget);
  static_assert((2 * kStages + 1) * sizeof(uint64_t) <= kBarrierBytes);
  // mbarrier tx-count is 20 bits.
  static_assert(kStageTxBytes < (1u << 20) && kInboxBytes < (1u << 20));
  static_assert(kSmemBytes <= 227 * 1024);
};

struct Params {
  bf16_t* __restrict__ out;           // [m, n], contiguous
  const int4* __restrict__ b;         // [k / 16, 2 * n] int32, awq_marlin_repack layout
  const bf16_t* __restrict__ scales;  // [k / 128, n] bf16, marlin_permute_scales layout
  const int32_t* __restrict__ zeros;  // [k / 128, n / 8] int32, awq_to_marlin_zero_points layout
  float* __restrict__ partials;       // [grid][kPartialFloats], fp32
  int32_t* __restrict__ flags;        // [grid], zero between launches
  uint32_t m;
  uint32_t n;
  uint32_t k;
};

/// \brief One CTA's work: units [begin, end) of the unit-major (column unit,
/// group) order, where a column unit is kTiles Marlin tiles.
struct WorkRange {
  uint32_t begin;
  uint32_t end;

  SGL_DEVICE_HOST static WorkRange of(uint32_t cta, uint32_t num_ctas, uint32_t num_units) {
    return {
        static_cast<uint32_t>(uint64_t{cta} * num_units / num_ctas),
        static_cast<uint32_t>(uint64_t{cta + 1} * num_units / num_ctas)};
  }
};

/// \brief A run of groups [first, last) of one column unit, inside one CTA's range.
struct Segment {
  uint32_t unit;
  uint32_t first;
  uint32_t last;
};

/// \brief The segment of `range` that ends at unit `end`, which is the last
/// unprocessed one when walking the range back to front.
SGL_DEVICE Segment segment_ending_at(uint32_t end, uint32_t range_begin, uint32_t num_groups) {
  const uint32_t unit = (end - 1) / num_groups;
  const uint32_t unit_begin = unit * num_groups;
  return {unit, max(range_begin, unit_begin) - unit_begin, end - unit_begin};
}

// PTX matrix descriptor for a K-major tile in 128B-swizzle atoms (8 rows x
// 128 B, 1024 B apart). A k16 step inside an atom advances the start address by
// 32 B; the leading-byte offset is unused for swizzled K-major layouts.
SGL_DEVICE uint64_t smem_desc_kmajor_sw128(uint32_t addr) {
  constexpr uint64_t kLayoutSwizzle128B = 1;
  constexpr uint32_t kAtomStrideBytes = 1024;
  return static_cast<uint64_t>((addr & 0x3FFFF) >> 4) | uint64_t{1} << 16 |
         static_cast<uint64_t>(kAtomStrideBytes >> 4) << 32 | kLayoutSwizzle128B << 62;
}

template <int kN, size_t... I>
SGL_DEVICE void gmma_impl(const uint32_t (&a)[4], uint64_t desc_b, float (&d)[kN / 2], std::index_sequence<I...>) {
  Gmma<kN>::Op::fma(a[0], a[1], a[2], a[3], desc_b, d[I]...);
}

template <int kN>
SGL_DEVICE void gmma(const uint32_t (&a)[4], uint64_t desc_b, float (&d)[kN / 2]) {
  gmma_impl<kN>(a, desc_b, d, std::make_index_sequence<kN / 2>{});
}

SGL_DEVICE void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}
SGL_DEVICE void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}
SGL_DEVICE void wgmma_wait_all() {
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
}

// Registers an in-flight wgmma reads or writes must stay where they are until
// its wait_group; this pins them across the asynchronous window.
template <typename T, int kCount>
SGL_DEVICE void fence_registers(T (&regs)[kCount]) {
#pragma unroll
  for (int i = 0; i < kCount; ++i) {
    if constexpr (std::is_same_v<T, float>) {
      asm volatile("" : "+f"(regs[i])::"memory");
    } else {
      asm volatile("" : "+r"(regs[i])::"memory");
    }
  }
}

// Named barriers: 0 is __syncthreads; these sync consumer warpgroups only.
enum NamedBarrier : uint32_t { kHandoffWritten = 1, kHandoffRead = 2, kFinisherReady = 3 };
SGL_DEVICE void named_barrier_sync(NamedBarrier id, uint32_t threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(static_cast<uint32_t>(id)), "r"(threads) : "memory");
}

SGL_DEVICE void flag_release(int32_t* flag) {
  asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(flag), "r"(1) : "memory");
}
SGL_DEVICE void flag_wait(const int32_t* flag) {
  int32_t v;
  do {
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(flag) : "memory");
  } while (v == 0);
}

template <typename T, bool kUsePDL>
__global__ void __launch_bounds__(T::kThreads, T::kMinBlocksPerSm)
    w4a16_sm90_kernel(const __grid_constant__ CUtensorMap act_map, const Params params) {
  using namespace device;
  using bf162 = nv_bfloat162;
  constexpr auto kU4 = host::kU4.id();
  constexpr int kN = T::kN;
  constexpr int kStages = T::kStages;

  extern __shared__ __align__(1024) uint8_t smem[];
  auto* full = reinterpret_cast<uint64_t*>(smem);
  auto* empty = full + kStages;
  auto* inbox_bar = empty + kStages;
  uint8_t* stages = smem + T::kBarrierBytes;
  auto* handoff = reinterpret_cast<float*>(stages + kStages * T::kStageBytes);
  auto* inbox = reinterpret_cast<float*>(stages + kStages * T::kStageBytes + T::kHandoffBytes);

  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t lane = threadIdx.x % kWarpThreads;
  const uint32_t num_groups = params.k / kGroupSize;
  const uint32_t n_tiles = params.n / kTileN;
  const uint32_t num_units = params.n / T::kCtaN * num_groups;
  // Unsplit and cluster modes: cluster c is unit c, and rank r takes its r-th
  // share of groups.
  const uint32_t rank = T::kClusterReduce ? ptx::cluster_ctarank() : 0;
  const WorkRange range = [&] {
    if constexpr (T::kStreamK) {
      return WorkRange::of(blockIdx.x, gridDim.x, num_units);
    } else {
      const uint32_t unit_begin = blockIdx.x / T::kClusterDim * num_groups;
      const WorkRange share = WorkRange::of(rank, T::kClusterDim, num_groups);
      return WorkRange{unit_begin + share.begin, unit_begin + share.end};
    }
  }();

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      ptx::mbar_init(&full[s], 1);
      // Each stage is consumed by one warpgroup per column tile.
      ptx::mbar_init(&empty[s], 4 * T::kTiles);
    }
    if constexpr (T::kClusterReduce) {
      ptx::mbar_init(inbox_bar, 1);
      // Every inbox slot is pushed, padding tokens included, so the byte count
      // is a constant.
      ptx::mbar_arrive_expect_tx(inbox_bar, T::kInboxBytes);
    }
    ptx::fence_mbarrier_init_release_cluster();
  }
  __syncthreads();
  // Each thread arrives here and waits once, right before it is done with the
  // cluster: a CTA must stay alive while peers may still push into its inbox.
  if constexpr (T::kClusterReduce) ptx::cluster_arrive_relaxed();

  auto stage_ptr = [&](uint32_t s) { return stages + s * T::kStageBytes; };

  if (warp_id == T::kProducerWarp) {
    // ---- Producer: the range back to front, each segment's groups in order ---
    if (lane == 0) {
      ptx::prefetch_tensormap(&act_map);
      uint32_t i = 0;
      for (uint32_t end = range.end; end > range.begin;) {
        const Segment seg = segment_ending_at(end, range.begin, num_groups);
        for (uint32_t g = seg.first; g < seg.last; ++g, ++i) {
          const uint32_t s = i % kStages;
          ptx::mbar_wait_parity(&empty[s], ((i / kStages) & 1) ^ 1);
          ptx::mbar_arrive_expect_tx(&full[s], T::kStageTxBytes);
          const uint32_t base = ptx::to_shared(stage_ptr(s));
#pragma unroll
          for (int kt = 0; kt < kGroupKTiles; ++kt) {
            const int4* src =
                params.b + (static_cast<size_t>(g * kGroupKTiles + kt) * n_tiles + seg.unit * T::kTiles) * 32;
            ptx::cp_async_bulk(
                base + T::kWeightOffset + kt * T::kTiles * kTileBytes, src, T::kTiles * kTileBytes, &full[s]);
          }
          ptx::cp_async_bulk(
              base + T::kScaleOffset,
              params.scales + static_cast<size_t>(g) * params.n + seg.unit * T::kCtaN,
              T::kScaleBytes,
              &full[s]);
          ptx::cp_async_bulk(
              base + T::kZeroOffset,
              params.zeros + static_cast<size_t>(g) * (params.n / 8) + seg.unit * (T::kCtaN / 8),
              T::kZeroBytes,
              &full[s]);
          // The weights are not the producer kernel's output, so the first
          // group's weights are in flight before the PDL wait.
          if (i == 0) PDLWaitPrimary<kUsePDL>();
#pragma unroll
          for (uint32_t c = 0; c < kGroupSize / T::kActAtomK; ++c) {
            ptx::cp_async_bulk_tensor_2d(
                base + c * T::kActAtomBytes, &act_map, g * kGroupSize + c * T::kActAtomK, 0, &full[s]);
          }
        }
        end -= seg.last - seg.first;
      }
    }
    __syncwarp();
    if constexpr (T::kClusterReduce) ptx::cluster_wait_acquire();
    return;
  }

  // ---- Consumers ---------------------------------------------------------------
  const uint32_t slot = warp_id / 4 % T::kTiles;  // the unit's column tile this warpgroup owns
  const uint32_t ping = warp_id / 4 / T::kTiles;
  const uint32_t wi = warp_id % 4;  // == the Marlin column subtile this warp owns
  const uint32_t tid = threadIdx.x % kWarpGroupThreads;
  const uint32_t frag_g = lane / 4;
  const uint32_t frag_t = lane % 4;

  float acc[T::kAccRegs];
#pragma unroll
  for (int r = 0; r < T::kAccRegs; ++r)
    acc[r] = 0.0f;

  // Each group's wgmmas retire before the same warpgroup's next dequant: ptxas
  // serializes every wgmma whose A registers are written while an earlier wgmma
  // is in flight, so the overlap comes from the tile's other warpgroup.
  auto run_group = [&](uint32_t i) {
    const uint32_t s = i % kStages;
    ptx::mbar_wait_parity(&full[s], (i / kStages) & 1);
    const uint8_t* st = stage_ptr(s);

    // Zero points per Marlin's dequant: nibbles {0, 4} and {1, 5} feed
    // subtiles 0 and 1, the next byte subtiles 2 and 3.
    const int32_t zq = *reinterpret_cast<const int32_t*>(st + T::kZeroOffset + slot * 32 + frag_g * 4);
    bf162 zp[4];
    marlin::dequant<bf162, kU4, true>(zq, &zp[0]);
    marlin::dequant<bf162, kU4, true>(zq >> 8, &zp[2]);
    const bf162 scale = reinterpret_cast<const bf162*>(st + T::kScaleOffset + slot * 128 + frag_g * 16)[wi];
    const bf162 z0 = __bfloat162bfloat162(zp[wi].x), z1 = __bfloat162bfloat162(zp[wi].y);
    const bf162 s0 = __bfloat162bfloat162(scale.x), s1 = __bfloat162bfloat162(scale.y);

    uint32_t a[kGroupKTiles][4];
#pragma unroll
    for (int kt = 0; kt < kGroupKTiles; ++kt) {
      const int q = *reinterpret_cast<const int*>(
          st + T::kWeightOffset + (kt * T::kTiles + slot) * kTileBytes + lane * 16 + wi * 4);
      bf162 b0[2], b1[2];
      marlin::dequant<bf162, kU4, true>(q, b0);
      marlin::dequant<bf162, kU4, true>(q >> 8, b1);
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        // (w - z) * s, the same bf16 ops as Marlin's sub_zp + scale.
        b0[e] = __hmul2(__hsub2(b0[e], z0), s0);
        b1[e] = __hmul2(__hsub2(b1[e], z1), s1);
      }
      // b0 holds rows g, b1 rows g + 8 of the subtile; [0] is k 2t..2t+1, [1] k 2t+8..2t+9.
      a[kt][0] = reinterpret_cast<const uint32_t&>(b0[0]);
      a[kt][1] = reinterpret_cast<const uint32_t&>(b1[0]);
      a[kt][2] = reinterpret_cast<const uint32_t&>(b0[1]);
      a[kt][3] = reinterpret_cast<const uint32_t&>(b1[1]);
    }

    const uint32_t act = ptx::to_shared(st);
    wgmma_fence();
#pragma unroll
    for (int kt = 0; kt < kGroupKTiles; ++kt) {
      const uint32_t atom = kt * 16 / T::kActAtomK;
      const uint32_t k_in_atom = kt * 16 % T::kActAtomK;
      gmma<kN>(a[kt], smem_desc_kmajor_sw128(act + atom * T::kActAtomBytes + k_in_atom * 2), acc);
    }
    wgmma_commit();
    wgmma_wait_all();
#pragma unroll
    for (int kt = 0; kt < kGroupKTiles; ++kt) {
      fence_registers(a[kt]);
    }
    fence_registers(acc);
    if (lane == 0) ptx::mbar_arrive(&empty[s]);
  };

  // Accumulator (wgmma m64nN): acc[4j + {0, 1}] is row 16 wi + g, tokens
  // 8j + 2t + {0, 1}; acc[4j + {2, 3}] is row 16 wi + g + 8, same tokens.
  auto store_unit = [&](uint32_t unit) {
    const uint32_t col = unit * T::kCtaN + slot * kTileN + wi * 16 + frag_g;
    auto store = [&](uint32_t tok, uint32_t c, float v) {
      if (tok < params.m) params.out[static_cast<size_t>(tok) * params.n + c] = __float2bfloat16_rn(v);
    };
#pragma unroll
    for (int j = 0; j < kN / 8; ++j) {
      const uint32_t tok = 8 * j + 2 * frag_t;
      store(tok, col, acc[4 * j + 0]);
      store(tok + 1, col, acc[4 * j + 1]);
      store(tok, col + 8, acc[4 * j + 2]);
      store(tok + 1, col + 8, acc[4 * j + 3]);
    }
  };

  // Cluster mode: push this rank's partial rows to their owners, then sum the
  // rows this rank owns over all ranks in rank order, so the result is fixed.
  auto reduce_in_cluster = [&](uint32_t unit) {
    ptx::cluster_wait_acquire();
    const uint32_t inbox_addr = ptx::to_shared(inbox);
    const uint32_t bar_addr = ptx::to_shared(inbox_bar);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t row = slot * kTileN + wi * 16 + frag_g + 8 * h;
      const uint32_t owner = row / T::kOwnedCols;
      const uint32_t dst = ptx::mapa(inbox_addr, owner);
      const uint32_t bar = ptx::mapa(bar_addr, owner);
      const uint32_t row_base = (rank * T::kOwnedCols + row % T::kOwnedCols) * kN;
#pragma unroll
      for (int j = 0; j < kN / 8; ++j) {
        const uint32_t idx = row_base + 8 * j + 2 * frag_t;
        ptx::st_async_v2_b32(dst + idx * 4, acc[4 * j + 2 * h], acc[4 * j + 2 * h + 1], bar);
      }
    }
    // Unconditional: this is also what keeps the CTA alive until every peer's
    // push into its inbox has landed.
    ptx::mbar_wait_parity(inbox_bar, 0);
    constexpr uint32_t kPairs = T::kOwnedCols * kN / 2;
    for (uint32_t p = slot * kWarpGroupThreads + tid; p < kPairs; p += T::kFinisherThreads) {
      const uint32_t local = p / (kN / 2);
      const uint32_t tok = (p % (kN / 2)) * 2;
      float2 sum = {0.0f, 0.0f};
#pragma unroll
      for (int r = 0; r < T::kClusterK; ++r) {
        const auto v = *reinterpret_cast<const float2*>(&inbox[(r * T::kOwnedCols + local) * kN + tok]);
        sum.x += v.x;
        sum.y += v.y;
      }
      const uint32_t col = unit * T::kCtaN + rank * T::kOwnedCols + local;
      if (tok < params.m) params.out[static_cast<size_t>(tok) * params.n + col] = __float2bfloat16_rn(sum.x);
      if (tok + 1 < params.m) params.out[static_cast<size_t>(tok + 1) * params.n + col] = __float2bfloat16_rn(sum.y);
    }
  };

  uint32_t i = 0;
  for (uint32_t end = range.end; end > range.begin;) {
    const Segment seg = segment_ending_at(end, range.begin, num_groups);
    for (uint32_t g = seg.first; g < seg.last; ++g, ++i) {
      if (i % T::kPing == ping) run_group(i);
    }
    end -= seg.last - seg.first;

    // Each column tile's partial sums of this segment meet in its first warpgroup.
    float* const tile_handoff = handoff + slot * T::kAccRegs * kWarpGroupThreads;
    if constexpr (T::kPing > 1) {
      if (ping == 1) {
#pragma unroll
        for (int r = 0; r < T::kAccRegs; ++r) {
          tile_handoff[r * kWarpGroupThreads + tid] = acc[r];
          acc[r] = 0.0f;
        }
      }
      named_barrier_sync(kHandoffWritten, T::kConsumerThreads);
      if (ping == 0) {
#pragma unroll
        for (int r = 0; r < T::kAccRegs; ++r)
          acc[r] += tile_handoff[r * kWarpGroupThreads + tid];
      }
      named_barrier_sync(kHandoffRead, T::kConsumerThreads);
      if (ping == 1) continue;
    }

    // A CTA that has not loaded activations yet has read nothing the producer
    // kernel wrote; the output may still alias its inputs.
    PDLWaitPrimary<kUsePDL>();
    if constexpr (T::kClusterReduce) {
      reduce_in_cluster(seg.unit);
    } else if constexpr (!T::kStreamK) {
      store_unit(seg.unit);
    } else {
      const bool finishes_tile = seg.last == num_groups;
      if (!finishes_tile) {
        // Only the first segment walked can stop short of its tile's end, so this
        // CTA has one partial to hand over. Partials bypass L1 on both sides: the
        // reader is another SM.
        float* partial = params.partials + static_cast<size_t>(blockIdx.x) * T::kPartialFloats +
                         slot * T::kAccRegs * kWarpGroupThreads;
#pragma unroll
        for (int r = 0; r < T::kAccRegs; ++r)
          __stcg(&partial[r * kWarpGroupThreads + tid], acc[r]);
        named_barrier_sync(kFinisherReady, T::kFinisherThreads);
        if (slot == 0 && tid == 0) flag_release(&params.flags[blockIdx.x]);
      } else {
        // Lower-numbered CTAs hold the unit's earlier groups. Wait for all of
        // them first, so their partials load concurrently rather than one round
        // trip per CTA; add them nearest first, so the summation order is fixed.
        const uint32_t unit_begin = seg.unit * num_groups;
        uint32_t first_cta = blockIdx.x;
        while (seg.first > 0 && WorkRange::of(first_cta, gridDim.x, num_units).begin > unit_begin)
          --first_cta;
        if (slot == 0 && tid == 0) {
          for (uint32_t cta = first_cta; cta < blockIdx.x; ++cta)
            flag_wait(&params.flags[cta]);
        }
        named_barrier_sync(kFinisherReady, T::kFinisherThreads);
        for (uint32_t cta = blockIdx.x; cta > first_cta;) {
          --cta;
          const float* partial =
              params.partials + static_cast<size_t>(cta) * T::kPartialFloats + slot * T::kAccRegs * kWarpGroupThreads;
#pragma unroll
          for (int r = 0; r < T::kAccRegs; ++r)
            acc[r] += __ldcg(&partial[r * kWarpGroupThreads + tid]);
          if (slot == 0 && tid == 0) params.flags[cta] = 0;
        }
        store_unit(seg.unit);
      }
    }
#pragma unroll
    for (int r = 0; r < T::kAccRegs; ++r)
      acc[r] = 0.0f;
  }
  // The tile's second warpgroups skipped the epilogue above.
  if constexpr (T::kClusterReduce) {
    if (ping == 1) ptx::cluster_wait_acquire();
  }
  PDLTriggerSecondary<kUsePDL>();
}

}  // namespace sglang::w4a16_sm90

namespace sglang {

/**
 * \brief Validate the Marlin-layout AWQ operands and launch the W4A16 GEMM.
 *
 * \tparam kN        token tile (wgmma N), at least M
 * \tparam kTiles    Marlin column tiles per work unit
 * \tparam kPing     consumer warpgroups per column tile
 * \tparam kClusterK 0: stream-K; 1: one CTA per unit; above 1: cluster split-K
 * \tparam kStages   quantization groups in flight per CTA
 * \tparam kUsePDL  launch with programmatic dependent launch
 * \param out        [M, N] bf16, contiguous
 * \param a          [M, K] bf16, unit inner stride, row stride a multiple of 8
 * \param b_q_weight [K / 16, 2 * N] int32 from awq_marlin_repack
 * \param b_scales   [K / 128, N] bf16 from marlin_permute_scales
 * \param b_zeros    [K / 128, N / 8] int32 from awq_to_marlin_zero_points
 * \param flags      [>= SMs * resident CTAs per SM] int32, all zero; the kernel
 *                   leaves it zero, so one buffer serves every launch on a stream
 * \param min_groups_per_cta stream-K only: the floor on groups per CTA
 */
template <int kN, int kTiles, int kPing, int kClusterK, int kStages, bool kUsePDL>
void w4a16_sm90_gemm(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b_q_weight,
    tvm::ffi::TensorView b_scales,
    tvm::ffi::TensorView b_zeros,
    tvm::ffi::TensorView flags,
    int64_t min_groups_per_cta) {
  using namespace host;
  using T = w4a16_sm90::Trait<kN, kTiles, kPing, kClusterK, kStages>;
  constexpr int kGroupSize = w4a16_sm90::kGroupSize;

  SymbolicSize M = {"m"};
  SymbolicSize N = {"n"};
  SymbolicSize K = {"k"};
  SymbolicSize LDA = {"lda"};
  SymbolicSize KT = {"k_tiles"};
  SymbolicSize G = {"num_groups"};
  SymbolicSize N2 = {"2n"};
  SymbolicSize N8 = {"n_div_8"};
  SymbolicSize F = {"num_flags"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  // TMA needs a 16-byte aligned base and row stride.
  TensorMatcher({M, K})  //
      .with_strides({LDA, 1})
      .with_dtype<bf16_t>()
      .with_device<kDLCUDA>(device_)
      .ensure_alignment(16)
      .verify(a);
  TensorMatcher({M, N})  //
      .with_dtype<bf16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(out);
  TensorMatcher({KT, N2})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .ensure_alignment(16)
      .verify(b_q_weight);
  TensorMatcher({G, N})  //
      .with_dtype<bf16_t>()
      .with_device<kDLCUDA>(device_)
      .ensure_alignment(16)
      .verify(b_scales);
  TensorMatcher({G, N8})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .ensure_alignment(16)
      .verify(b_zeros);
  TensorMatcher({F})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(flags);

  const int64_t m = M.unwrap();
  const int64_t n = N.unwrap();
  const int64_t k = K.unwrap();
  // With a single group marlin_permute_scales switches to its channelwise
  // permutation, which this kernel does not read.
  CHECK_HOST(k >= 2 * kGroupSize && k % kGroupSize == 0)
      << "w4a16_sm90: K must be a multiple of " << kGroupSize << " spanning at least two groups, got " << k;
  CHECK_HOST(n % T::kCtaN == 0) << "w4a16_sm90: N must be a multiple of " << T::kCtaN << ", got " << n;
  CHECK_HOST(KT.unwrap() * 16 == k) << "w4a16_sm90: b_q_weight has " << KT.unwrap() << " k-tiles for K = " << k;
  CHECK_HOST(N2.unwrap() == 2 * n) << "w4a16_sm90: b_q_weight inner dim must be 2 * N";
  CHECK_HOST(G.unwrap() * kGroupSize == k) << "w4a16_sm90: expected group size " << kGroupSize;
  CHECK_HOST(N8.unwrap() * 8 == n) << "w4a16_sm90: b_zeros inner dim must be N / 8";
  CHECK_HOST(m <= kN) << "w4a16_sm90: token tile " << kN << " is smaller than M = " << m;
  CHECK_HOST(n <= UINT32_MAX && k <= UINT32_MAX) << "w4a16_sm90: dims exceed 32 bits";
  if (m == 0) return;

  constexpr auto kernel = w4a16_sm90::w4a16_sm90_kernel<T, kUsePDL>;
  [[maybe_unused]] static const auto _ = [] {
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, T::kSmemBytes));
    return 0;
  }();
  // Unsplit and cluster modes: one cluster per unit, in as many waves as it
  // takes. Stream-K: a persistent grid -- every CTA resident at once -- which is
  // what lets a CTA wait on the partials of lower-numbered ones; the floor on
  // groups per CTA bounds how many partials a finisher sums.
  const DLDevice device = device_.unwrap();
  const uint32_t num_groups = static_cast<uint32_t>(k / kGroupSize);
  const uint32_t num_column_units = static_cast<uint32_t>(n / T::kCtaN);
  uint32_t grid;
  if constexpr (!T::kStreamK) {
    CHECK_HOST(num_groups >= kClusterK) << "w4a16_sm90: " << kClusterK << " K splits for " << num_groups << " groups";
    grid = num_column_units * kClusterK;
  } else {
    static const uint32_t blocks_per_sm = runtime::get_blocks_per_sm(kernel, T::kThreads, T::kSmemBytes);
    const uint32_t resident = runtime::get_sm_count(device.device_id) * blocks_per_sm;
    const uint32_t num_units = num_column_units * num_groups;
    const uint32_t min_groups = static_cast<uint32_t>(std::max<int64_t>(min_groups_per_cta, kPing));
    grid = std::max(1u, std::min(resident, num_units / min_groups));
    CHECK_HOST(F.unwrap() >= grid) << "w4a16_sm90: " << F.unwrap() << " flags for a grid of " << grid;
  }

  // Rows past M are zero-filled by TMA, which is how the token tile is padded.
  const CUtensorMap act_map = make_tma_map_2d(
      a.data_ptr(),
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      k,
      m,
      LDA.unwrap() * sizeof(bf16_t),
      T::kActAtomK,
      kN,
      CU_TENSOR_MAP_SWIZZLE_128B);
  // Freed on return; the caching allocator keeps it alive for this stream's work.
  const auto partials =
      ffi::alloc_workspace_tensor(T::kStreamK ? size_t{grid} * T::kPartialFloats * sizeof(float) : 0, device);
  const auto params = w4a16_sm90::Params{
      .out = static_cast<bf16_t*>(out.data_ptr()),
      .b = static_cast<const int4*>(b_q_weight.data_ptr()),
      .scales = static_cast<const bf16_t*>(b_scales.data_ptr()),
      .zeros = static_cast<const int32_t*>(b_zeros.data_ptr()),
      .partials = T::kStreamK ? static_cast<float*>(partials.data_ptr()) : nullptr,
      .flags = static_cast<int32_t*>(flags.data_ptr()),
      .m = static_cast<uint32_t>(m),
      .n = static_cast<uint32_t>(n),
      .k = static_cast<uint32_t>(k),
  };
  LaunchKernel(grid, T::kThreads, device, T::kSmemBytes)
      .config({.use_pdl = kUsePDL, .cluster_dim = dim3{T::kClusterDim, 1, 1}})(kernel, act_map, params);
}

}  // namespace sglang
