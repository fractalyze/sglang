"""Tuned `triton_scaled_mm` tiles for Gemma-4-26B-A4B's dense FP8 linears on the RTX 5090.

On SM120 at decode batch sizes the CUTLASS per-token x per-channel FP8 GEMM launches 22-64 CTAs and streams the
weights at 0.2-0.6 TB/s. The six config files give each decode M (1-48) a Triton tile and leave M >= 64 on
CUTLASS (null entries). These tests pin that lookup, and on an SM120 GPU that every tuned tile computes the same
product as CUTLASS up to accumulation order.
"""

import unittest
from unittest import mock

import torch

from sglang.kernels.ops.quantization import fp8_kernel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

DEVICE = "NVIDIA GeForce RTX 5090"
# (N, K): qkv sliding / full, o sliding / full, dense MLP gate_up / down.
SHAPES = (
    (8192, 2816),
    (10240, 2816),
    (2816, 4096),
    (2816, 8192),
    (4224, 2816),
    (2816, 2112),
)
DECODE_MS = (1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48)
PREFILL_MS = (64, 100, 512, 2048, 8192)
TILE_KEYS = {"BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_warps", "num_stages"}


def _on_rtx5090():
    fp8_kernel.get_w8a8_channelwise_fp8_configs.cache_clear()
    return mock.patch.object(fp8_kernel, "get_device_name", return_value=DEVICE)


class TestRTX5090ChannelwiseConfigs(CustomTestCase):
    def tearDown(self):
        fp8_kernel.get_w8a8_channelwise_fp8_configs.cache_clear()

    def test_decode_ms_get_a_tile_and_prefill_keeps_cutlass(self):
        with _on_rtx5090():
            for n, k in SHAPES:
                for m in DECODE_MS:
                    cfg = fp8_kernel.get_w8a8_channelwise_fp8_config(N=n, K=k, M=m)
                    self.assertIsNotNone(cfg, (n, k, m))
                    self.assertEqual(set(cfg), TILE_KEYS)
                for m in PREFILL_MS:
                    self.assertIsNone(
                        fp8_kernel.get_w8a8_channelwise_fp8_config(N=n, K=k, M=m),
                        (n, k, m),
                    )

    def test_other_devices_are_untouched(self):
        fp8_kernel.get_w8a8_channelwise_fp8_configs.cache_clear()
        with mock.patch.object(
            fp8_kernel, "get_device_name", return_value="NVIDIA GeForce RTX 4090"
        ):
            self.assertIsNone(
                fp8_kernel.get_w8a8_channelwise_fp8_config(N=8192, K=2816, M=8)
            )

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
        "needs an SM120 GPU",
    )
    def test_tuned_tiles_match_cutlass(self):
        from sglang.kernels.ops.gemm import fp8_scaled_mm as cutlass_scaled_mm
        from sglang.srt.layers.quantization.fp8_utils import sglang_per_token_quant_fp8

        torch.manual_seed(0)
        with _on_rtx5090():
            for n, k in SHAPES:
                w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
                scale_w = w.float().abs().amax(dim=1, keepdim=True) / 448.0
                w_fp8 = (
                    (w.float() / scale_w).to(torch.float8_e4m3fn).t()
                )  # served layout: [K, N] view
                for m in (1, 8, 28, 48):
                    x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
                    qx, sx = sglang_per_token_quant_fp8(x)
                    ref = cutlass_scaled_mm(
                        qx, w_fp8, sx, scale_w, out_dtype=torch.bfloat16
                    )
                    cfg = fp8_kernel.get_w8a8_channelwise_fp8_config(N=n, K=k, M=m)
                    out = fp8_kernel.triton_scaled_mm(
                        qx,
                        w_fp8,
                        sx,
                        scale_w,
                        torch.bfloat16,
                        None,
                        block_size_m=cfg["BLOCK_SIZE_M"],
                        block_size_n=cfg["BLOCK_SIZE_N"],
                        block_size_k=cfg["BLOCK_SIZE_K"],
                        use_heuristic=False,
                        num_warps=cfg["num_warps"],
                        num_stages=cfg["num_stages"],
                    )
                    err = (out.float() - ref.float()).norm() / ref.float().norm()
                    self.assertLess(err.item(), 1e-2, (n, k, m))


if __name__ == "__main__":
    unittest.main()
