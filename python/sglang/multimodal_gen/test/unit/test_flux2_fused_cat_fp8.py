# SPDX-License-Identifier: Apache-2.0
"""The FLUX.2 single-block cat + per-token FP8 quantization kernel is bitwise
the unfused concatenation followed by sglang's per-token quantization."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("rows", [4608, 4096, 512, 3])
def test_fused_cat_quant_matches_cat_then_quant(rows):
    from sglang.kernels.kda_kernels.flux2_token_cat_fp8_triton import (
        try_flux2_token_cat_fp8_per_token,
    )
    from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8

    generator = torch.Generator(device="cuda").manual_seed(rows)
    attention = torch.randn(
        1, rows, 3072, device="cuda", generator=generator
    ).bfloat16()
    mlp = (
        torch.randn(1, rows, 9216, device="cuda", generator=generator) * 4
    ).bfloat16()
    attention[0, 0] = 0
    mlp[0, 0] = 0  # an all-zero row quantizes to zeros with scale 0
    if rows > 2:
        mlp[0, 1, 17] = 3.0e4  # an outlier sets the row scale
        attention[0, 2].mul_(1e-6)
    q, scale = try_flux2_token_cat_fp8_per_token(attention, mlp)
    expected_q, expected_scale = sglang_per_token_quant_fp8(
        torch.cat([attention, mlp], dim=-1).reshape(rows, -1).contiguous()
    )
    assert torch.equal(scale, expected_scale)
    assert torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8))


def test_unsupported_inputs_fall_back():
    from sglang.kernels.kda_kernels.flux2_token_cat_fp8_triton import (
        try_flux2_token_cat_fp8_per_token,
    )

    attention = torch.zeros(1, 8, 3072, device="cuda", dtype=torch.bfloat16)
    assert try_flux2_token_cat_fp8_per_token(attention, attention.float()) is None
    wide = torch.zeros(1, 8, 16384, device="cuda", dtype=torch.bfloat16)
    assert try_flux2_token_cat_fp8_per_token(attention, wide) is None
