# Copyright 2026 Fractalyze Inc. All rights reserved.
"""int4 weights in groups of 32 as compressed-tensors W4A16 stores them, as
decode-mk's int4 cores read them (s2mk/csrc/int4_*_core.cuh).

A weight row of k values is `packed` int32 [k / 8], value j in bits
4 (j % 8) of word j / 8 as q + 8, with q in [-8, 7], and `scales` bf16
[k / 32]. Symmetric weights, as Qwen3-Omni's checkpoint and vLLM keep them,
are w = q × scale over the group of 32 a scale covers. Asymmetric weights, as
Qwen3.8's checkpoint keeps them, add a zero point z in [-8, 7] per group:
w = (q − z) × scale. The kernels read a row's zero points as `zeros` int32
[k / 256], group j's z + 8 in bits 4 (j % 8) of word j / 8; the checkpoint
packs them down the rows instead (`zeros_from_checkpoint`).
"""

from __future__ import annotations

from collections.abc import Iterator

import torch

GROUP = 32
PER_WORD = 8
# Rows a large weight is quantized or dequantized in at a time, so fp32 never
# holds the whole of it (Qwen3.8's LM head is 248,320 rows).
SLICE_ROWS = 1 << 15


def _pack(values: torch.Tensor) -> torch.Tensor:
    """int32 [n, m / 8] holding int [n, m] values in [0, 15], value j of a row
    in bits 4 (j % 8) of word j / 8."""
    n, m = values.shape
    shifts = torch.arange(0, 32, 4, device=values.device, dtype=torch.int64)
    words = (values.to(torch.int64).view(n, m // PER_WORD, PER_WORD) << shifts).sum(dim=-1)
    # The same 32 bits as a signed int32, as the checkpoint stores them.
    return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32).contiguous()


def _unpack(words: torch.Tensor) -> torch.Tensor:
    """The int32 [n, m] values in [0, 15] that int32 [n, m / 8] `words` hold."""
    shifts = torch.arange(0, 32, 4, device=words.device, dtype=torch.int32)
    return ((words[..., None] >> shifts) & 0xF).flatten(-2)


def quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(packed, scales) for a float [n, k] weight, symmetric: each group's
    scale maps its largest magnitude to 7."""
    n, k = w.shape
    groups = w.float().view(n, k // GROUP, GROUP)
    scales = (groups.abs().amax(dim=-1, keepdim=True) / 7).clamp(min=1e-8)
    scales = scales.bfloat16()
    q = (groups / scales.float()).round().clamp(-8, 7) + 8
    return _pack(q.view(n, k)), scales.view(n, k // GROUP).contiguous()


def quantize_asymmetric(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(packed, scales, zeros) for a float [n, k] weight: each group's scale
    spreads its range, widened to hold 0, over the 16 levels, and its zero
    point puts the group's minimum on the lowest. Works in slices of
    SLICE_ROWS rows."""
    if w.shape[0] > SLICE_ROWS:
        slices = [quantize_asymmetric(rows) for rows in w.split(SLICE_ROWS)]
        return tuple(torch.cat(part) for part in zip(*slices))
    n, k = w.shape
    groups = w.float().view(n, k // GROUP, GROUP)
    lo = groups.amin(dim=-1, keepdim=True).clamp(max=0)
    hi = groups.amax(dim=-1, keepdim=True).clamp(min=0)
    scales = ((hi - lo) / 15).clamp(min=1e-8).bfloat16()
    z = (-8 - (lo / scales.float()).round()).clamp(-8, 7)
    q = ((groups / scales.float()).round() + z).clamp(-8, 7) + 8
    return (_pack(q.view(n, k)), scales.view(n, k // GROUP).contiguous(),
            _pack((z + 8).view(n, k // GROUP)))


def zeros_from_checkpoint(zero_point: torch.Tensor) -> torch.Tensor:
    """The kernels' `zeros` for a compressed-tensors `weight_zero_point`,
    int32 [n / 8, k / 32], which packs row r's zero points into words r / 8
    at bits 4 (r % 8)."""
    rows = _unpack(zero_point.transpose(0, 1).contiguous())  # [k / 32, n]
    return _pack(rows.transpose(0, 1).contiguous())


def dequantize(packed: torch.Tensor, scales: torch.Tensor,
               zeros: torch.Tensor | None = None) -> torch.Tensor:
    """The fp32 [n, k] weight (packed, scales), with `zeros` if asymmetric,
    stand for."""
    n = packed.shape[0]
    q = _unpack(packed).view(n, -1, GROUP)
    z = 8 if zeros is None else _unpack(zeros).view(n, -1, 1)
    return ((q - z).float() * scales.float().view(n, -1, 1)).view(n, -1)


# (packed, scales, zeros): an asymmetric int4 weight.
Int4Weight = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
# A projection as Qwen3.8's kernels take it: int4, or bf16 where the checkpoint
# keeps it so.
Projection = Int4Weight | torch.Tensor


def dense(w: Projection) -> torch.Tensor:
    """The fp32 [n, k] weight a projection stands for."""
    return w.float() if isinstance(w, torch.Tensor) else dequantize(*w)


def dense_slices(w: Projection) -> Iterator[torch.Tensor]:
    """dense(w) in slices of SLICE_ROWS rows, each made only when reached."""
    if isinstance(w, torch.Tensor):
        return (rows.float() for rows in w.split(SLICE_ROWS))
    return (dequantize(*parts) for parts in zip(*(part.split(SLICE_ROWS) for part in w)))


def rows(w: Projection) -> int:
    """The rows of a projection."""
    return (w if isinstance(w, torch.Tensor) else w[0]).shape[0]


def parts(w: Projection) -> list[torch.Tensor]:
    """A projection as the kernels' params builders take it (ops.cpp's
    WeightOf)."""
    return [w] if isinstance(w, torch.Tensor) else list(w)
