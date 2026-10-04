/// \file tma.cuh
/// \brief TMA (SM90+) bulk copies into shared memory that complete on an
/// mbarrier, and the host-side 2D tensor-map encoder.
///
/// A module that includes this header calls the driver API
/// (`cuTensorMapEncodeTiled`), so its `load_jit` links
/// `-L{cuda_stubs_dir()} -lcuda`.

#pragma once

#include <sgl_kernel/utils.h>

#include <sgl_kernel/mbarrier.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstdint>
#include <cuda.h>

namespace sglang {

namespace device::ptx {

/// \brief Warm the cache line holding a tensor-map descriptor, so the first TMA
/// does not pay the descriptor fetch on top of the DRAM latency.
SGL_DEVICE void prefetch_tensormap(const void* tmap) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(tmap) : "memory");
}

/// \brief Copy one 2D tile into shared memory and credit `bar` with its bytes.
///
/// Coordinate convention: the tensor map's globalDim is (inner, outer) -- dim 0
/// is the stride-1 axis -- and the load takes (x = inner, y = outer). Swapping
/// them loads a transposed tile with plausible magnitudes and scrambled pairing.
SGL_DEVICE void
cp_async_bulk_tensor_2d(uint32_t dst_smem, const CUtensorMap* tmap, int32_t x, int32_t y, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];" ::"r"(dst_smem),
      "l"(tmap),
      "r"(x),
      "r"(y),
      "r"(to_shared(bar))
      : "memory");
}

/// \brief Copy `bytes` contiguous bytes into shared memory and credit `bar`.
/// Source, destination and size must all be multiples of 16 bytes.
SGL_DEVICE void cp_async_bulk(uint32_t dst_smem, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst_smem),
      "l"(src),
      "r"(bytes),
      "r"(to_shared(bar))
      : "memory");
}

}  // namespace device::ptx

namespace host {

/// \brief Encode a 2D row-major tensor map.
///
/// \param base             Global base address, 16-byte aligned
/// \param cols             Inner (stride-1) extent, in elements
/// \param rows             Outer extent; TMA zero-fills rows past it
/// \param row_stride_bytes Outer stride, a multiple of 16
/// \param box_cols         Tile width, in elements
/// \param box_rows         Tile height
/// \param swizzle          Shared-memory swizzle of the landed tile
inline CUtensorMap make_tma_map_2d(
    const void* base,
    CUtensorMapDataType dtype,
    uint64_t cols,
    uint64_t rows,
    uint64_t row_stride_bytes,
    uint32_t box_cols,
    uint32_t box_rows,
    CUtensorMapSwizzle swizzle) {
  CUtensorMap map{};
  uint64_t dim[2] = {cols, rows};
  uint64_t stride[1] = {row_stride_bytes};
  uint32_t box[2] = {box_cols, box_rows};
  uint32_t elem_stride[2] = {1, 1};
  const CUresult status = cuTensorMapEncodeTiled(
      &map,
      dtype,
      2,
      const_cast<void*>(base),
      dim,
      stride,
      box,
      elem_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  CHECK_HOST(status == CUDA_SUCCESS) << "cuTensorMapEncodeTiled failed with CUresult " << static_cast<int>(status);
  return map;
}

}  // namespace host

}  // namespace sglang
