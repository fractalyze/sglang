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

#pragma once

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstdint>
#include <cuda_runtime.h>

// clang-format off
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"
// clang-format on

namespace sglang {

using namespace host;
using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

namespace fp8_channelwise_impl {

// Same epilogue math as sgl-kernel's DeviceGemmFp8RowwiseSm120:
// D = OutType(scale_a[m] * (scale_b[n] * acc)).
template <typename OutType, int kTileM, int kTileN, int kTileK, bool kStreamK>
struct ChannelwiseGemmSm120 {
  using ElementAB = cutlass::float_e4m3_t;
  using TileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;
  using ClusterShape = Shape<_1, _1, _1>;

  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
  using ScaleA = cutlass::epilogue::fusion::
      Sm90ColBroadcast<0, TileShape, float, float, Stride<Int<1>, Int<0>, Int<0>>>;
  using ScaleB = cutlass::epilogue::fusion::
      Sm90RowBroadcast<0, TileShape, float, float, Stride<Int<0>, Int<1>, Int<0>>>;
  using Compute0 = cutlass::epilogue::fusion::
      Sm90Compute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT0 = cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;
  using Compute1 = cutlass::epilogue::fusion::
      Sm90Compute<cutlass::multiplies, OutType, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT = cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVT0>;

  static constexpr int kAlignAB = 128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int kAlignD = 128 / cutlass::sizeof_bits<OutType>::value;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120,
      cutlass::arch::OpClassTensorOp,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float,
      float,
      void,
      cutlass::layout::RowMajor,
      kAlignD,
      OutType,
      cutlass::layout::RowMajor,
      kAlignD,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      EVT>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120,
      cutlass::arch::OpClassTensorOp,
      ElementAB,
      cutlass::layout::RowMajor,
      kAlignAB,
      ElementAB,
      cutlass::layout::ColumnMajor,
      kAlignAB,
      float,
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

  // StreamK keeps CUTLASS's default deterministic reduction, so results are reproducible run to run.
  using TileScheduler = std::conditional_t<kStreamK, cutlass::gemm::StreamKScheduler, void>;
  using GemmKernel = cutlass::gemm::kernel::
      GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, TileScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename OutType, int kTileM, int kTileN, int kTileK, bool kStreamK>
void run_channelwise_gemm_sm120(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b) {
  using Cfg = ChannelwiseGemmSm120<OutType, kTileM, kTileN, kTileK, kStreamK>;
  using GemmKernel = typename Cfg::GemmKernel;

  const int32_t m = static_cast<int32_t>(a.size(0));
  const int32_t k = static_cast<int32_t>(a.size(1));
  const int32_t n = static_cast<int32_t>(b.size(1));

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideD = typename GemmKernel::StrideD;
  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n, 1));

  typename Cfg::EVT0::Arguments evt0{{static_cast<float const*>(scales_b.data_ptr())}, {}, {}};
  typename Cfg::EVT::Arguments evt{{static_cast<float const*>(scales_a.data_ptr())}, evt0, {}};

  auto* ptr_d = static_cast<OutType*>(out.data_ptr());
  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {static_cast<typename Cfg::ElementAB const*>(a.data_ptr()),
       stride_a,
       static_cast<typename Cfg::ElementAB const*>(b.data_ptr()),
       stride_b},
      {evt, nullptr, stride_d, ptr_d, stride_d},
  };
  args.hw_info.device_id = a.device().device_id;
  args.hw_info.sm_count = static_cast<int>(host::runtime::get_sm_count(a.device().device_id));

  typename Cfg::Gemm gemm_op;
  cutlass::Status status = gemm_op.can_implement(args);
  RuntimeCheck(status == cutlass::Status::kSuccess, cutlassGetStatusString(status));
  const size_t workspace_size = Cfg::Gemm::get_workspace_size(args);
  auto workspace_tensor = host::ffi::alloc_workspace_tensor(workspace_size, a.device());
  void* workspace = workspace_size == 0 ? nullptr : workspace_tensor.data_ptr();
  const cudaStream_t stream = LaunchKernel::resolve_device(a.device());
  status = gemm_op.initialize(args, workspace, stream);
  RuntimeCheck(status == cutlass::Status::kSuccess, cutlassGetStatusString(status));
  status = gemm_op.run(stream);
  RuntimeCheck(status == cutlass::Status::kSuccess, cutlassGetStatusString(status));
}

}  // namespace fp8_channelwise_impl

/**
 * \brief out[M, N] = scale_a[M] * scale_b[N] * (a[M, K] @ b[K, N]) on SM120, e4m3 inputs.
 * \tparam kTileM, kTileN, kTileK CTA tile shape.
 * \tparam kStreamK use the StreamK tile scheduler instead of the persistent one.
 * \param out [M, N] bf16 or fp16, row major.
 * \param a [M, K] e4m3, row major.
 * \param b [K, N] e4m3, column major (the transposed [N, K] weight).
 * \param scales_a M fp32 per-token scales.
 * \param scales_b N fp32 per-channel scales.
 */
template <int kTileM, int kTileN, int kTileK, bool kStreamK>
void fp8_channelwise_scaled_mm_sm120(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b) {
  CHECK_HOST(a.dim() == 2 && b.dim() == 2) << "a and b must be 2D";
  CHECK_HOST(a.stride(1) == 1) << "a must be row major";
  CHECK_HOST(b.stride(0) == 1) << "b must be column major";
  CHECK_HOST(a.size(1) == b.size(0)) << "a and b shapes cannot be multiplied";
  CHECK_HOST(host::is_type<fp8_e4m3_t>(a.dtype()) && host::is_type<fp8_e4m3_t>(b.dtype())) << "a and b must be e4m3";
  CHECK_HOST(a.size(1) % 16 == 0) << "K must be a multiple of 16";
  CHECK_HOST(out.size(1) % 8 == 0) << "N must be a multiple of 8";
  CHECK_HOST(scales_a.numel() == a.size(0)) << "scales_a must hold one scale per row of a";
  CHECK_HOST(scales_b.numel() == b.size(1)) << "scales_b must hold one scale per column of b";
  CHECK_HOST(host::is_type<float>(scales_a.dtype()) && host::is_type<float>(scales_b.dtype()))
      << "scales must be fp32";

  if (host::is_type<bf16_t>(out.dtype())) {
    fp8_channelwise_impl::run_channelwise_gemm_sm120<cutlass::bfloat16_t, kTileM, kTileN, kTileK, kStreamK>(
        out, a, b, scales_a, scales_b);
  } else if (host::is_type<fp16_t>(out.dtype())) {
    fp8_channelwise_impl::run_channelwise_gemm_sm120<cutlass::half_t, kTileM, kTileN, kTileK, kStreamK>(
        out, a, b, scales_a, scales_b);
  } else {
    Panic("out dtype must be bf16 or fp16");
  }
}

#endif  // defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

}  // namespace sglang
