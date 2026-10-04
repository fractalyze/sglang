"""Tests for the Gemma4 fused norm pairs (decode glue groups B, C, D).

Each fused kernel is compared with the unfused SGLang modules it replaces:
- B ``gemma_rmsnorm_add_rmsnorm``: RMSNorm, then FusedAddRMSNorm with the residual.
- C ``gemma_dual_output_rmsnorm``: two RMSNorms of one input (router norm, pre-FF norm 2).
- D ``gemma_dual_rmsnorm_residual_scalar_next_norm``: the dual-norm epilogue plus the
  next layer's input RMSNorm.

The fused kernels sum squares in a different order than the FlashInfer norm kernels, so
outputs may differ by one bf16 ulp; the tests bound that and the fraction of such elements.

Requires a CUDA GPU; skips otherwise.
"""

from __future__ import annotations

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton fused norms need a CUDA GPU"
)

_HIDDEN = 2816
_EPS = 1e-6


def _norm(seed: int):
    from sglang.srt.layers.layernorm import RMSNorm

    m = RMSNorm(_HIDDEN, eps=_EPS).to("cuda", torch.bfloat16)
    g = torch.Generator(device="cuda").manual_seed(seed)
    m.weight.data = (
        1.0 + 0.3 * torch.randn(_HIDDEN, device="cuda", generator=g)
    ).bfloat16()
    return m


def _rows(M: int, seed: int, scale: float = 4.0) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (scale * torch.randn(M, _HIDDEN, device="cuda", generator=g)).bfloat16()


def _assert_close_bf16(got: torch.Tensor, want: torch.Tensor, max_frac: float = 0.01):
    ulps = (got.view(torch.int16).int() - want.view(torch.int16).int()).abs()
    assert ulps.max().item() <= 1, ulps.max().item()
    assert (ulps > 0).float().mean().item() <= max_frac


def _bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    # bf16 keeps 8 significant bits, so the spacing at |x| in [2^e, 2^(e+1)) is 2^(e-7).
    return torch.exp2(torch.floor(torch.log2(x.float().abs().clamp_min(2.0**-100))) - 7)


def _assert_within_ulp_of(got, want, magnitude, max_frac: float = 0.001):
    d = (got.float() - want.float()).abs()
    assert (d <= _bf16_ulp(magnitude)).all(), d.max().item()
    assert (d > 0).float().mean().item() <= max_frac


@pytest.mark.parametrize("M", [1, 8, 32, 513])
def test_rmsnorm_add_rmsnorm_matches_unfused(M):
    from sglang.kernels.ops.layernorm.gemma4_fused_ops import gemma_rmsnorm_add_rmsnorm

    post, pre = _norm(1), _norm(2)
    x, residual = _rows(M, 3), _rows(M, 4)

    ref_h = post(x.clone())
    h_magnitude = (
        ref_h.float().abs()
    )  # FusedAddRMSNorm overwrites ref_h with its output.
    ref_out, ref_res = pre(ref_h, residual.clone())

    out, res = gemma_rmsnorm_add_rmsnorm(
        x.clone(), post.weight.data, residual.clone(), pre.weight.data, _EPS, _EPS
    )
    # The first norm may land one ulp away (different sum order). That ulp is the larger addend's,
    # so the rounded residual can move by one ulp of max(|norm|, |residual|), which is two ulps of a
    # smaller residual; the second norm scales the residual, so its output moves by up to two ulps.
    magnitude = torch.maximum(h_magnitude, ref_res.float().abs())
    _assert_within_ulp_of(res, ref_res, magnitude)
    _assert_within_ulp_of(out, ref_out, 2.0 * ref_out.float().abs())


@pytest.mark.parametrize("M", [1, 8, 32, 513])
def test_dual_output_rmsnorm_matches_two_norms(M):
    from sglang.kernels.ops.layernorm.gemma4_fused_ops import gemma_dual_output_rmsnorm

    a, b = _norm(5), _norm(6)
    x = _rows(M, 7)
    out_a, out_b = gemma_dual_output_rmsnorm(
        x, a.weight.data, b.weight.data, _EPS, _EPS
    )
    _assert_close_bf16(out_a, a(x.clone()))
    _assert_close_bf16(out_b, b(x.clone()))


@pytest.mark.parametrize("M", [1, 8, 32, 513])
def test_dual_norm_epilogue_adds_next_input_norm(M):
    from sglang.kernels.ops.layernorm.gemma4_fused_ops import (
        gemma_dual_rmsnorm_residual_scalar,
        gemma_dual_rmsnorm_residual_scalar_next_norm,
    )

    n1, n2, n3, nxt = _norm(8), _norm(9), _norm(10), _norm(11)
    x1, x2, residual = _rows(M, 12), _rows(M, 13), _rows(M, 14)
    scalar = torch.tensor([0.7], device="cuda", dtype=torch.bfloat16)
    args = (x1, n1.weight.data, x2, n2.weight.data, n3.weight.data, residual, scalar)

    ref_out = gemma_dual_rmsnorm_residual_scalar(*args, _EPS, _EPS, _EPS)
    out, next_out = gemma_dual_rmsnorm_residual_scalar_next_norm(
        *args, nxt.weight.data, _EPS, _EPS, _EPS, _EPS
    )
    # The first output is the existing kernel's, unchanged.
    assert torch.equal(out, ref_out)
    _assert_close_bf16(next_out, nxt(ref_out.clone()))
