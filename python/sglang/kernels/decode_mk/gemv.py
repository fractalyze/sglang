# Copyright 2026 Fractalyze Inc. All rights reserved.
"""Batch-1 bf16 GEMV on one persistent CTA per SM.

Weights are packed once at load; a `GemvSequence` then runs any number of
packed GEMVs back to back in a single launch.
"""

from dataclasses import dataclass

import torch

from sglang.kernels.decode_mk import _ext

# Each warp step reads 32 lanes × 16 bytes = 256 bf16 columns.
_CHUNK_ELEMS = 256
# The kernel reads weights and x as 16-byte vectors.
_ALIGN_BYTES = 16


def num_ctas(device: torch.device | None = None) -> int:
    """One persistent CTA per SM."""
    return torch.cuda.get_device_properties(device).multi_processor_count


@dataclass(frozen=True)
class PackedGemv:
    """An n × k bf16 weight laid out as the kernel streams it.

    CTA c owns rows [c * n / C, (c + 1) * n / C) for C CTAs; with the weight
    row-major, each CTA's rows are one contiguous slice.
    """

    weight: torch.Tensor

    @property
    def n(self) -> int:
        return self.weight.shape[0]

    @property
    def k(self) -> int:
        return self.weight.shape[1]


def pack_gemv(weight: torch.Tensor) -> PackedGemv:
    """Packs an `nn.Linear`-style [out, in] bf16 CUDA weight."""
    if weight.dtype != torch.bfloat16 or not weight.is_cuda or weight.dim() != 2:
        raise ValueError("weight must be a 2-D bf16 CUDA tensor")
    if weight.shape[1] % _CHUNK_ELEMS:
        raise ValueError(f"in-features must be a multiple of {_CHUNK_ELEMS}")
    weight = weight.contiguous()
    if weight.data_ptr() % _ALIGN_BYTES:
        raise ValueError(f"weight must be {_ALIGN_BYTES}-byte aligned")
    return PackedGemv(weight)


class GemvSequence:
    """GEMVs y_i = W_i x_i run in order by one launch.

    The descriptors hold raw pointers: every weight, x and y passed here must
    stay alive, and stay in place, for as long as the sequence is run.
    """

    def __init__(
        self, steps: list[tuple[PackedGemv, torch.Tensor, torch.Tensor]]
    ) -> None:
        device = steps[0][0].weight.device
        self._ctas = num_ctas(device)
        rows = []
        for packed, x, y in steps:
            if packed.weight.device != device:
                raise ValueError("every weight must be on one device")
            if x.shape != (packed.k,) or y.shape != (packed.n,):
                raise ValueError(
                    f"x must be [{packed.k}] and y [{packed.n}] for this weight"
                )
            for t in (x, y):
                if t.dtype != torch.bfloat16 or not t.is_contiguous():
                    raise ValueError("x and y must be contiguous bf16")
                if t.device != device:
                    raise ValueError(f"x and y must be on the weights' {device}")
            if x.data_ptr() % _ALIGN_BYTES:
                raise ValueError(f"x must be {_ALIGN_BYTES}-byte aligned")
            rows.append(
                [
                    packed.weight.data_ptr(),
                    x.data_ptr(),
                    y.data_ptr(),
                    packed.n,
                    packed.k,
                ]
            )
        self._descs = torch.tensor(rows, dtype=torch.int64, device=device)
        max_k = max(p.k for p, _, _ in steps)
        max_rows = max(-(-p.n // self._ctas) for p, _, _ in steps)
        self._smem = _ext.load().gemv_smem_bytes(max_k, max_rows)

    def run(self) -> None:
        _ext.load().run_gemv(self._descs, self._ctas, self._smem)


def gemv(packed: PackedGemv, x: torch.Tensor) -> torch.Tensor:
    """Returns W x for a 1-D bf16 x, in bf16 with fp32 accumulation."""
    y = torch.empty(packed.n, dtype=torch.bfloat16, device=x.device)
    GemvSequence([(packed, x, y)]).run()
    return y
