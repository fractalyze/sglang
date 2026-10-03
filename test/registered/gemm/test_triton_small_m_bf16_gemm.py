"""
Tests the Triton small-M BF16 GEMM against cuBLAS (F.linear) and an fp32
reference, and that SGLANG_OPT_USE_TRITON_SMALL_M_BF16_GEMM toggles the
UnquantizedLinearMethod route.
"""

import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")

_TUNED_NK = [(2816, 4096), (2816, 8192), (4224, 2816), (2816, 2112)]


def _bf16_ulp(t: torch.Tensor) -> torch.Tensor:
    """Spacing of bf16 values at |t| (8 significand bits)."""
    mag = t.abs().clamp_min(torch.finfo(torch.bfloat16).tiny)
    return torch.exp2(torch.floor(torch.log2(mag)) - 7)


class _Linear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)


@unittest.skipIf(not torch.cuda.is_available(), "requires a CUDA GPU")
class TestTritonSmallMBf16Gemm(unittest.TestCase):
    def test_matches_cublas_within_bf16_rounding(self):
        from sglang.kernels.ops.gemm.triton_small_m_bf16_gemm import (
            triton_small_m_bf16_gemm,
        )

        torch.manual_seed(0)
        for n, k in _TUNED_NK:
            w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02
            for m in (1, 3, 8, 16, 17, 32):
                with self.subTest(n=n, k=k, m=m):
                    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
                    out = triton_small_m_bf16_gemm(x, w).float()
                    cub = F.linear(x, w).float()
                    ref = x.float() @ w.float().t()
                    # Each kernel returns the bf16 rounding of an fp32 dot product;
                    # summation order may differ by gamma = K * 2^-24 * sum|x||w|,
                    # and the fp32 reference carries the same order error.
                    gamma = k * 2.0**-24 * (x.float().abs() @ w.float().abs().t())
                    for name, y in (("triton", out), ("cublas", cub)):
                        excess = (y - ref).abs() - (0.5 * _bf16_ulp(ref) + 2 * gamma)
                        self.assertLessEqual(excess.max().item(), 0.0, name)

    def test_env_switch_toggles_route(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.quantization import unquant

        torch.manual_seed(0)
        layer = _Linear(
            torch.randn(2816, 4096, dtype=torch.bfloat16, device="cuda") * 0.02
        )
        x = torch.randn(8, 4096, dtype=torch.bfloat16, device="cuda")
        prefill = torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda")
        other_shape = _Linear(
            torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda")
        )
        for enabled in (False, True):
            with (
                self.subTest(enabled=enabled),
                envs.SGLANG_OPT_USE_TRITON_SMALL_M_BF16_GEMM.override(enabled),
                mock.patch.object(
                    unquant,
                    "triton_small_m_bf16_gemm",
                    wraps=unquant.triton_small_m_bf16_gemm,
                ) as spy,
            ):
                method = unquant.UnquantizedLinearMethod()
                out = method.apply(layer, x)
                self.assertEqual(spy.call_count, 1 if enabled else 0)
                self.assertEqual(out.shape, (8, 2816))
                method.apply(layer, prefill)
                method.apply(other_shape, x)
                self.assertEqual(spy.call_count, 1 if enabled else 0)


if __name__ == "__main__":
    unittest.main()
