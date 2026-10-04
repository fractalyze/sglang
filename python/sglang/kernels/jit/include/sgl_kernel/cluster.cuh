/// \file cluster.cuh
/// \brief Thread-block-cluster primitives (SM90+): DSMEM addressing, the split
/// cluster barrier, and `st.async` pushes that credit a peer's mbarrier.
///
/// A split-K reduction across a cluster uses them in this order: every CTA
/// initializes its inbox mbarrier, publishes it with
/// `fence_mbarrier_init_release_cluster`, and calls `cluster_arrive_relaxed`;
/// a CTA calls `cluster_wait_acquire` before its first `st_async_*` into a
/// peer, and each owner waits on its own mbarrier, which also keeps it alive
/// until every peer's push has landed.

#pragma once

#include <sgl_kernel/mbarrier.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstdint>

namespace sglang {

namespace device::ptx {

/// \brief Retarget a local shared-memory offset at CTA `rank` of this cluster.
SGL_DEVICE uint32_t mapa(uint32_t addr, uint32_t rank) {
  uint32_t out;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(out) : "r"(addr), "r"(rank));
  return out;
}

/// \brief This CTA's rank within its cluster.
SGL_DEVICE uint32_t cluster_ctarank() {
  uint32_t r;
  asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

/// \brief Push one value into a peer's inbox and credit its mbarrier byte count.
SGL_DEVICE void st_async_b32(uint32_t dst_dsmem, float value, uint32_t dst_bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(dst_dsmem),
               "f"(value),
               "r"(dst_bar)
               : "memory");
}

/// \brief Two-value form of `st_async_b32`; `dst_dsmem` must be 8-byte aligned.
SGL_DEVICE void st_async_v2_b32(uint32_t dst_dsmem, float x, float y, uint32_t dst_bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];" ::"r"(dst_dsmem),
               "f"(x),
               "f"(y),
               "r"(dst_bar)
               : "memory");
}

/// \brief Publish mbarrier initialization to the whole cluster. Because this
/// fence carries the release, the cluster arrive after it needs no ordering of
/// its own.
SGL_DEVICE void fence_mbarrier_init_release_cluster() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

/// \brief Arrive half of the split cluster barrier: every CTA arrives in the
/// prologue, but only the code that actually touches a peer's memory pays for
/// the wait.
SGL_DEVICE void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}

/// \brief Wait half of the split cluster barrier.
SGL_DEVICE void cluster_wait_acquire() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

}  // namespace device::ptx

}  // namespace sglang
