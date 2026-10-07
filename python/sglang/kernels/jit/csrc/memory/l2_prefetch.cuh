/// \file l2_prefetch.cuh
/// \brief Stage byte ranges of global memory in L2 ahead of the kernel that
/// reads them, SM90+.
///
/// Each range is one `cp.async.bulk.prefetch.L2` issued by one thread. The
/// prefetch is a hint: it writes nothing, completes asynchronously in the copy
/// engine, and the kernel exits as soon as every range is issued. A caller
/// overlaps it with work that leaves HBM idle (a collective), so the next
/// kernel finds its operands in L2.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace {

constexpr uint32_t kThreads = 128;

/// \tparam kEvictLast Tag the staged lines evict-last, so traffic arriving
///   before their reader evicts other lines first.
template <bool kEvictLast>
__global__ void l2_prefetch_kernel(const int64_t* __restrict__ ranges, int64_t num_ranges) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint64_t policy = 0;
  if constexpr (kEvictLast) {
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
  }
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < num_ranges; i += stride) {
    const uint64_t addr = static_cast<uint64_t>(ranges[2 * i]);
    const uint32_t bytes = static_cast<uint32_t>(ranges[2 * i + 1]);
    if constexpr (kEvictLast) {
      asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" ::"l"(addr), "r"(bytes), "l"(policy)
                   : "memory");
    } else {
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(addr), "r"(bytes) : "memory");
    }
  }
#endif
}

}  // namespace

/// \brief Issue an L2 prefetch for every (address, bytes) row of `ranges`.
/// \tparam kEvictLast See `l2_prefetch_kernel`.
/// \param ranges [R, 2] int64 on the device: a 16-byte-aligned address and a
///   byte count that is a positive multiple of 16 and below 2^32.
/// \param num_ctas Grid size; a few CTAs suffice, since each thread only issues.
template <bool kEvictLast>
void l2_prefetch(tvm::ffi::TensorView ranges, int64_t num_ctas) {
  using namespace host;
  auto R = SymbolicSize{"num_ranges"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({R, 2}).with_dtype<int64_t>().with_device<kDLCUDA>(device).verify(ranges);
  const int64_t num_ranges = R.unwrap();
  CHECK_HOST(num_ctas > 0) << "num_ctas must be positive, got " << num_ctas;
  if (num_ranges == 0) return;
  LaunchKernel(static_cast<uint32_t>(num_ctas), kThreads, device.unwrap())(
      l2_prefetch_kernel<kEvictLast>, static_cast<const int64_t*>(ranges.data_ptr()), num_ranges);
}

}  // namespace sglang
