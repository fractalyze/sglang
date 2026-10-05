"""Selection of the opt-in sm120 FP8-KV extend tiles (no GPU needed; needs the SGLang tree)."""

from unittest import mock

import torch
from absl.testing import absltest

from sglang.kernels.ops.attention import extend_attention as ea
from sglang.srt.environ import envs


def _buf(dtype):
    return torch.empty(1, 1, 1, dtype=dtype)


class Sm120TilesTest(absltest.TestCase):
    def _select(self, cap=(12, 0), on=True, dtype=torch.float8_e4m3fn, lq=512, lv=512):
        with mock.patch.object(ea, "_is_cuda", True), mock.patch.object(ea, "CUDA_CAPABILITY", cap), \
                envs.SGLANG_OPT_TRITON_EXTEND_SM120_FP8_KV_TILES.override(on):
            return ea._sm120_fp8_kv_extend_tiles(lq, lv, _buf(dtype))

    def test_applies_on_sm120_fp8_kv_when_enabled(self):
        self.assertEqual(self._select(), ea._SM120_FP8_KV_EXTEND_TILES[512])
        self.assertEqual(self._select(lq=256, lv=256), ea._SM120_FP8_KV_EXTEND_TILES[256])

    def test_defaults_kept_otherwise(self):
        self.assertIsNone(self._select(on=False))
        self.assertIsNone(self._select(cap=(10, 0)))
        self.assertIsNone(self._select(dtype=torch.bfloat16))
        self.assertIsNone(self._select(lq=128, lv=128))
        self.assertIsNone(self._select(lq=576, lv=512))


if __name__ == "__main__":
    absltest.main()
