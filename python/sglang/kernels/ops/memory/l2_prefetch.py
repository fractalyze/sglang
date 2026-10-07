"""Stage tensors in L2 ahead of the kernel that reads them (SM90+).

`plan_l2_prefetch` turns a list of tensors into the device-side range table,
once and outside graph capture; `l2_prefetch` issues it, cheap enough to sit in
a captured decode step. See csrc/memory/l2_prefetch.cuh.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Bytes per prefetch instruction; arbitrary, small enough to spread a few MB
# over a few hundred threads.
_RANGE_BYTES = 64 * 1024
# The bulk prefetch moves 16-byte units from a 16-byte-aligned address.
_ALIGN = 16
_NUM_CTAS = 8


@cache_once
def _jit_l2_prefetch_module(evict_last: bool) -> Module:
    if torch.cuda.get_device_capability() < (9, 0):
        raise RuntimeError("l2_prefetch requires SM90 or newer")
    args = make_cpp_args(evict_last)
    return load_jit(
        "l2_prefetch",
        *args,
        cuda_files=["memory/l2_prefetch.cuh"],
        cuda_wrappers=[("run", f"l2_prefetch<{args}>")],
    )


def _byte_span(tensor: torch.Tensor) -> tuple[int, int]:
    """(start, bytes) of the memory a view reaches, widened to 16-byte bounds."""
    assert tensor.is_cuda and all(s >= 0 for s in tensor.stride())
    elements = 1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))
    start = tensor.data_ptr() // _ALIGN * _ALIGN
    end = tensor.data_ptr() + elements * tensor.element_size()
    return start, -(-(end - start) // _ALIGN) * _ALIGN


def plan_l2_prefetch(
    tensors: Sequence[torch.Tensor], budget_bytes: int
) -> torch.Tensor:
    """The [R, 2] int64 (address, bytes) table covering `tensors` in order.

    Coverage stops at `budget_bytes`: a later range would evict an earlier one
    from L2 before its reader arrives. The table holds raw addresses, so the
    caller keeps `tensors` alive while it is in use.
    """
    rows = []
    remaining = budget_bytes // _ALIGN * _ALIGN
    for tensor in tensors:
        if tensor.numel() == 0:
            continue
        start, nbytes = _byte_span(tensor)
        nbytes = min(nbytes, remaining)
        remaining -= nbytes
        for offset in range(0, nbytes, _RANGE_BYTES):
            rows.append((start + offset, min(_RANGE_BYTES, nbytes - offset)))
        if remaining == 0:
            break
    return torch.tensor(rows, dtype=torch.int64, device="cuda").reshape(-1, 2)


def l2_prefetch(ranges: torch.Tensor, evict_last: bool = False) -> None:
    """Prefetch every range of a `plan_l2_prefetch` table into L2.

    Returns once the prefetches are issued, not landed; it writes nothing.
    """
    _jit_l2_prefetch_module(evict_last).run(ranges, _NUM_CTAS)
