"""Tile selection for Triton extend attention, SGLANG_OPT_TRITON_EXTEND_SM120_FP8_KV_TILES.

With the switch on, sm120 extend attention over an FP8 KV cache takes the re-tuned
tiles in ``_SM120_FP8_KV_EXTEND_TILES`` for its head dim. Any other GPU, a BF16
KV cache, an untuned head dim, Lq != Lv, or the switch off keeps the default tiles.
"""

import unittest
from unittest import mock

import torch

from sglang.kernels.ops.attention import extend_attention as ea
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _select(cap=(12, 0), on=True, dtype=torch.float8_e4m3fn, lq=512, lv=512):
    k_buffer = torch.empty(1, 1, 1, dtype=dtype)
    # CUDA_CAPABILITY exists only on a CUDA host; create it for the patch elsewhere.
    with (
        mock.patch.object(ea, "_is_cuda", True),
        mock.patch.object(ea, "CUDA_CAPABILITY", cap, create=True),
        envs.SGLANG_OPT_TRITON_EXTEND_SM120_FP8_KV_TILES.override(on),
    ):
        return ea._sm120_fp8_kv_extend_tiles(lq, lv, k_buffer)


class TestSm120Fp8KvExtendTiles(CustomTestCase):
    def test_applies_on_sm120_fp8_kv_when_enabled(self):
        self.assertEqual(_select(), ea._SM120_FP8_KV_EXTEND_TILES[512])
        self.assertEqual(_select(lq=256, lv=256), ea._SM120_FP8_KV_EXTEND_TILES[256])

    def test_defaults_kept_otherwise(self):
        self.assertIsNone(_select(on=False))
        self.assertIsNone(_select(cap=(10, 0)))
        self.assertIsNone(_select(dtype=torch.bfloat16))
        self.assertIsNone(_select(lq=128, lv=128))
        self.assertIsNone(_select(lq=576, lv=512))


if __name__ == "__main__":
    unittest.main()
