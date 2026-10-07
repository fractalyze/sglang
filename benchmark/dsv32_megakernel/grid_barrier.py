"""Time one grid barrier of a persistent kernel on the local GPU.

A persistent decode megakernel keeps one grid resident and synchronizes it
where the unfused path launches a kernel, so each launch it removes costs a
grid barrier instead. This measures that barrier for a grid of `--ctas-per-sm`
CTAs on every SM, two ways:

  * cg: `cooperative_groups::this_grid().sync()`;
  * flag: one atomic arrival counter and a generation flag, the barrier a
    hand-written megakernel uses.

Each barrier's cost is the slope between two loop lengths, so launch overhead
cancels. Pass the smaller of the two to `trace_split.py --grid-barrier-us`.
"""

import argparse
import json

import torch
from torch.utils.cpp_extension import load_inline

_CPP = "void run(int64_t mode, int64_t ctas, int64_t threads, int64_t iters, torch::Tensor scratch);"

_CUDA = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <torch/extension.h>

namespace cg = cooperative_groups;

__global__ void cg_barrier_loop(int iters, unsigned* scratch) {
  cg::grid_group grid = cg::this_grid();
  for (int i = 0; i < iters; ++i) grid.sync();
}

__device__ __forceinline__ unsigned load_acquire(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// scratch[0] counts arrivals, scratch[1] is the generation the last arrival bumps.
__global__ void flag_barrier_loop(int iters, unsigned* scratch) {
  unsigned* count = scratch;
  unsigned* generation = scratch + 1;
  for (int i = 0; i < iters; ++i) {
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned seen = load_acquire(generation);
      __threadfence();
      if (atomicAdd(count, 1) == gridDim.x - 1) {
        *count = 0;
        __threadfence();
        atomicAdd(generation, 1);
      } else {
        while (load_acquire(generation) == seen) {
        }
      }
    }
    __syncthreads();
  }
}

void run(int64_t mode, int64_t ctas, int64_t threads, int64_t iters, torch::Tensor scratch) {
  int n = static_cast<int>(iters);
  unsigned* ptr = reinterpret_cast<unsigned*>(scratch.data_ptr<int32_t>());
  void* args[] = {&n, &ptr};
  void* kernel = mode == 0 ? reinterpret_cast<void*>(cg_barrier_loop)
                           : reinterpret_cast<void*>(flag_barrier_loop);
  TORCH_CHECK(
      cudaLaunchCooperativeKernel(kernel, dim3(ctas), dim3(threads), args, 0,
                                  at::cuda::getCurrentCUDAStream()) == cudaSuccess,
      "cooperative launch failed; is the grid larger than one wave?");
}
"""

_MODES = ("cg", "flag")


def _time_ms(
    *, module, mode: int, ctas: int, threads: int, iters: int, scratch
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    module.run(mode, ctas, threads, iters, scratch)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def _barrier_us(*, module, mode: int, ctas: int, threads: int, iters: int, reps: int):
    """Median per-barrier microseconds over `reps`, from the slope of 2n vs n barriers."""
    launch = dict(
        module=module,
        mode=mode,
        ctas=ctas,
        threads=threads,
        scratch=torch.zeros(2, dtype=torch.int32, device="cuda"),
    )
    _time_ms(iters=iters, **launch)  # warm up
    slopes = sorted(
        1e3
        * (_time_ms(iters=2 * iters, **launch) - _time_ms(iters=iters, **launch))
        / iters
        for _ in range(reps)
    )
    return slopes[len(slopes) // 2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ctas-per-sm", type=int, default=1)
    parser.add_argument("--threads", type=int, nargs="+", default=[128, 256, 384])
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()

    module = load_inline(
        name="dsv32_grid_barrier",
        cpp_sources=_CPP,
        cuda_sources=_CUDA,
        functions=["run"],
        extra_cuda_cflags=["-O3"],
    )
    props = torch.cuda.get_device_properties(0)
    ctas = args.ctas_per_sm * props.multi_processor_count
    for threads in args.threads:
        for mode, name in enumerate(_MODES):
            us = _barrier_us(
                module=module,
                mode=mode,
                ctas=ctas,
                threads=threads,
                iters=args.iters,
                reps=args.reps,
            )
            row = {
                "device": props.name,
                "barrier": name,
                "ctas": ctas,
                "threads": threads,
                "us": round(us, 3),
            }
            print(json.dumps(row))


if __name__ == "__main__":
    main()
