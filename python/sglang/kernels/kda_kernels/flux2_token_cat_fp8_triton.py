from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.quantization.fp8_kernel import (
    fp8_dtype,
    fp8_max,
    fp8_min,
)
from sglang.kernels.ops.quantization.fp8_utils import fp8_dtype_to_triton

_BLOCK = 4096
_NUM_WARPS = 8


@triton.jit
def _token_cat_fp8_kernel(
    attention,
    mlp,
    output,
    input_scale,
    attention_hidden: tl.constexpr,
    mlp_hidden: tl.constexpr,
    output_hidden: tl.constexpr,
    BLOCK: tl.constexpr,
    FP8_DTYPE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    columns = block * BLOCK + tl.arange(0, BLOCK)
    output_mask = columns < output_hidden
    attention_mask = output_mask & (columns < attention_hidden)
    mlp_columns = columns - attention_hidden
    mlp_mask = output_mask & (columns >= attention_hidden)

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    attention_values = tl.load(
        attention + row * attention_hidden + columns,
        mask=attention_mask,
        other=0.0,
    ).to(tl.float32)
    mlp_values = tl.load(
        mlp + row * mlp_hidden + mlp_columns,
        mask=mlp_mask,
        other=0.0,
    ).to(tl.float32)
    values = tl.where(columns < attention_hidden, attention_values, mlp_values)
    scale = tl.load(input_scale).to(tl.float32)
    quantized = tl.clamp(values * (1.0 / scale), FP8_MIN, FP8_MAX).to(FP8_DTYPE)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    tl.store(
        output + row * output_hidden + columns,
        quantized.to(tl.uint8, bitcast=True),
        mask=output_mask,
    )


def try_flux2_token_cat_fp8(
    attention: torch.Tensor,
    mlp: torch.Tensor,
    input_scale: torch.Tensor,
) -> torch.Tensor | None:
    """Concatenate FLUX.2 single-block branches directly into static FP8."""
    if torch.compiler.is_compiling():
        return None
    if not (
        isinstance(attention, torch.Tensor)
        and isinstance(mlp, torch.Tensor)
        and attention.is_cuda
        and mlp.is_cuda
        and attention.device == mlp.device
        and attention.dtype == torch.bfloat16
        and mlp.dtype == torch.bfloat16
        and attention.ndim == 3
        and mlp.ndim == 3
        and attention.shape[:-1] == mlp.shape[:-1]
        and attention.is_contiguous()
        and mlp.is_contiguous()
        and attention.numel() > 0
        and mlp.numel() > 0
    ):
        return None
    if (
        torch.cuda.is_current_stream_capturing()
        or torch.cuda.get_device_capability(attention.device)[0] < 10
    ):
        return None
    if not (
        isinstance(input_scale, torch.Tensor)
        and input_scale.is_cuda
        and input_scale.device == attention.device
        and input_scale.dtype == torch.float32
        and input_scale.numel() == 1
        and input_scale.is_contiguous()
    ):
        return None

    attention_hidden = attention.shape[-1]
    mlp_hidden = mlp.shape[-1]
    output_hidden = attention_hidden + mlp_hidden
    rows = attention.numel() // attention_hidden
    output = torch.empty(
        (*attention.shape[:-1], output_hidden),
        dtype=fp8_dtype,
        device=attention.device,
    )
    pdl_kwargs = (
        {"USE_PDL": True, "launch_pdl": True}
        if is_arch_support_pdl()
        else {"USE_PDL": False}
    )
    with torch.cuda.device(attention.device):
        _token_cat_fp8_kernel[(rows, triton.cdiv(output_hidden, _BLOCK))](
            attention,
            mlp,
            output.view(torch.uint8),
            input_scale,
            attention_hidden=attention_hidden,
            mlp_hidden=mlp_hidden,
            output_hidden=output_hidden,
            BLOCK=_BLOCK,
            FP8_DTYPE=fp8_dtype_to_triton(fp8_dtype),
            FP8_MIN=fp8_min,
            FP8_MAX=fp8_max,
            num_warps=_NUM_WARPS,
            num_stages=1,
            **pdl_kwargs,
        )
    return output


# One program holds a whole output row (fp32) to take its absolute maximum.
_PER_TOKEN_ROW_MAX = 16384


@triton.jit
def _token_cat_fp8_per_token_kernel(
    attention,
    mlp,
    output,
    output_scale,
    attention_hidden: tl.constexpr,
    mlp_hidden: tl.constexpr,
    BLOCK: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK)
    output_hidden = attention_hidden + mlp_hidden
    attention_values = tl.load(
        attention + row * attention_hidden + columns,
        mask=columns < attention_hidden,
        other=0.0,
    ).to(tl.float32)
    mlp_values = tl.load(
        mlp + row * mlp_hidden + (columns - attention_hidden),
        mask=(columns >= attention_hidden) & (columns < output_hidden),
        other=0.0,
    ).to(tl.float32)
    values = tl.where(columns < attention_hidden, attention_values, mlp_values)
    # per_token_quant_fp8_warp_kernel's arithmetic (IEEE divisions):
    # scale = amax / 448, then x * (1 / scale), clamped.
    scale = tl.math.div_rn(tl.max(tl.abs(values), 0), FP8_MAX)
    tl.store(output_scale + row, scale)
    inverse = tl.where(scale == 0.0, 0.0, tl.math.div_rn(1.0, scale))
    quantized = tl.clamp(values * inverse, -FP8_MAX, FP8_MAX)
    tl.store(
        output + row * output_hidden + columns,
        quantized.to(output.dtype.element_ty),
        mask=columns < output_hidden,
    )


def try_flux2_token_cat_fp8_per_token(
    attention: torch.Tensor,
    mlp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Concatenate FLUX.2 single-block branches directly into per-token FP8.

    Returns ``(q [M, A + B] e4m3, scale [M, 1] fp32)``, bitwise what
    ``sglang_per_token_quant_fp8(torch.cat([attention, mlp], -1))`` returns
    (its warp kernel), without the bf16 concatenation; None when the inputs do
    not fit (the caller then concatenates).
    """
    if torch.compiler.is_compiling():
        return None
    if not (
        isinstance(attention, torch.Tensor)
        and isinstance(mlp, torch.Tensor)
        and attention.is_cuda
        and attention.device == mlp.device
        and attention.dtype == torch.bfloat16
        and mlp.dtype == torch.bfloat16
        and attention.shape[:-1] == mlp.shape[:-1]
        and attention.is_contiguous()
        and mlp.is_contiguous()
        and attention.numel() > 0
        and mlp.numel() > 0
        and attention.shape[-1] + mlp.shape[-1] <= _PER_TOKEN_ROW_MAX
    ):
        return None
    attention_hidden = attention.shape[-1]
    mlp_hidden = mlp.shape[-1]
    rows = attention.numel() // attention_hidden
    output = torch.empty(
        (rows, attention_hidden + mlp_hidden), dtype=fp8_dtype, device=attention.device
    )
    scale = torch.empty((rows, 1), dtype=torch.float32, device=attention.device)
    with torch.cuda.device(attention.device):
        _token_cat_fp8_per_token_kernel[(rows,)](
            attention,
            mlp,
            output,
            scale,
            attention_hidden=attention_hidden,
            mlp_hidden=mlp_hidden,
            BLOCK=triton.next_power_of_2(attention_hidden + mlp_hidden),
            FP8_MAX=fp8_max,
            num_warps=16,
        )
    return output, scale


__all__ = ["try_flux2_token_cat_fp8", "try_flux2_token_cat_fp8_per_token"]
