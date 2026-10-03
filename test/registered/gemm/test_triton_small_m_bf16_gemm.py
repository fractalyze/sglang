"""
Tests the Triton small-M BF16 GEMM against cuBLAS (F.linear) and an fp32
reference, and that SGLANG_OPT_USE_TRITON_SMALL_M_BF16_GEMM toggles the
UnquantizedLinearMethod route. Also tests the FP8
weight-only variant behind SGLANG_OPT_USE_TRITON_SMALL_M_FP8_WEIGHT_GEMM.
"""

import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")

_TUNED_NK = [(2816, 4096), (2816, 8192), (4224, 2816), (2816, 2112)]
# (N, K) -> largest routed M: o_proj stops at decode widths, qkv_proj covers MTP verify.
_FP8_WEIGHT_MAX_M = {(2816, 4096): 32, (2816, 8192): 32, (8192, 2816): 48, (10240, 2816): 48}
_FP8_WEIGHT_NK = list(_FP8_WEIGHT_MAX_M)


def _bf16_ulp(t: torch.Tensor) -> torch.Tensor:
    """Spacing of bf16 values at |t| (8 significand bits)."""
    mag = t.abs().clamp_min(torch.finfo(torch.bfloat16).tiny)
    return torch.exp2(torch.floor(torch.log2(mag)) - 7)


def _assert_within_bf16_rounding(test, y, x, w_fp32, roundings):
    """Each bf16 rounding of the fp32 result costs half an ulp; order error is gamma."""
    ref = x.float() @ w_fp32.t()
    gamma = x.shape[1] * 2.0**-24 * (x.float().abs() @ w_fp32.abs().t())
    excess = (y.float() - ref).abs() - (roundings * 0.5 * _bf16_ulp(ref) + 2 * gamma)
    test.assertLessEqual(excess.max().item(), 0.0)


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

    def test_fp8_weight_matches_dequantized_reference(self):
        from sglang.kernels.ops.gemm.triton_small_m_bf16_gemm import (
            quantize_fp8_weight_per_channel,
            triton_small_m_fp8_weight_gemm,
        )

        torch.manual_seed(0)
        for n, k in _FP8_WEIGHT_NK:
            w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02
            w8, scale = quantize_fp8_weight_per_channel(w)
            # The scale is the row absmax over E4M3's largest finite value.
            self.assertEqual(w8.float().abs().amax(dim=1).min().item(), 448.0)
            w_deq = w8.float() * scale[:, None]
            for m in sorted({1, 6, 8, 17, 32, _FP8_WEIGHT_MAX_M[(n, k)]}):
                with self.subTest(n=n, k=k, m=m):
                    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
                    y = triton_small_m_fp8_weight_gemm(x, w8, scale)
                    _assert_within_bf16_rounding(self, y, x, w_deq, roundings=1)

    def test_fp8_weight_switch_converts_and_routes(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.quantization import unquant

        torch.manual_seed(0)
        n, k = _FP8_WEIGHT_NK[0]
        w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02
        decode = torch.randn(8, k, dtype=torch.bfloat16, device="cuda")
        prefill = torch.randn(1024, k, dtype=torch.bfloat16, device="cuda")
        for enabled in (False, True):
            with (
                self.subTest(enabled=enabled),
                envs.SGLANG_OPT_USE_TRITON_SMALL_M_FP8_WEIGHT_GEMM.override(enabled),
                mock.patch.object(
                    unquant,
                    "triton_small_m_fp8_weight_gemm",
                    wraps=unquant.triton_small_m_fp8_weight_gemm,
                ) as spy,
            ):
                method = unquant.UnquantizedLinearMethod()
                layer = _Linear(w.clone())
                other_shape = _Linear(w[:1024].clone())
                method.process_weights_after_loading(layer)
                method.process_weights_after_loading(other_shape)
                self.assertEqual(other_shape.weight.dtype, torch.bfloat16)
                if not enabled:
                    self.assertEqual(layer.weight.dtype, torch.bfloat16)
                    continue
                self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
                w_deq = layer.weight.float() * layer.weight_scale[:, None]
                y = method.apply(layer, decode)
                self.assertEqual(spy.call_count, 1)
                _assert_within_bf16_rounding(self, y, decode, w_deq, roundings=1)
                # Prefill rounds the unscaled GEMM output, then the scaled one.
                y = method.apply(layer, prefill)
                self.assertEqual(spy.call_count, 1)
                _assert_within_bf16_rounding(self, y, prefill, w_deq, roundings=3)

    def test_fp8_weight_route_stops_at_each_shapes_max_m(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.quantization import unquant

        torch.manual_seed(0)
        method_cls = unquant.UnquantizedLinearMethod
        for (n, k), max_m in _FP8_WEIGHT_MAX_M.items():
            with (
                self.subTest(n=n, k=k),
                envs.SGLANG_OPT_USE_TRITON_SMALL_M_FP8_WEIGHT_GEMM.override(True),
                mock.patch.object(
                    unquant,
                    "triton_small_m_fp8_weight_gemm",
                    wraps=unquant.triton_small_m_fp8_weight_gemm,
                ) as spy,
            ):
                method = method_cls()
                layer = _Linear(
                    torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02
                )
                method.process_weights_after_loading(layer)
                self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
                w_deq = layer.weight.float() * layer.weight_scale[:, None]
                x = torch.randn(max_m, k, dtype=torch.bfloat16, device="cuda")
                y = method.apply(layer, x)
                self.assertEqual(spy.call_count, 1)
                _assert_within_bf16_rounding(self, y, x, w_deq, roundings=1)
                x = torch.randn(max_m + 1, k, dtype=torch.bfloat16, device="cuda")
                y = method.apply(layer, x)
                self.assertEqual(spy.call_count, 1)
                _assert_within_bf16_rounding(self, y, x, w_deq, roundings=3)


if __name__ == "__main__":
    unittest.main()
