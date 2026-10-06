import os
import sys

from absl.testing import absltest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k2_channelwise_configs as cc  # noqa: E402


def _row(n, k, m, bn):
    return {"n": n, "k": k, "m": m, "w8a8_triton_best_config": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": bn,
                                                                "BLOCK_SIZE_K": 256, "num_warps": 4,
                                                                "num_stages": 4}}


class ConfigsTest(absltest.TestCase):
    def test_one_file_per_shape_with_decode_tiles_and_null_prefill(self):
        bench = {"device": "NVIDIA GeForce RTX 5090",
                 "rows": [_row(8192, 2816, 8, 64), _row(8192, 2816, 28, 32), _row(2816, 2112, 8, 32)]}
        out = cc.configs_from_bench(bench)
        name = "N=8192,K=2816,device_name=NVIDIA_GeForce_RTX_5090,dtype=fp8_w8a8_channelwise.json"
        self.assertLen(out, 2)
        table = out[name]
        self.assertEqual(table["8"]["BLOCK_SIZE_N"], 64)
        self.assertEqual(table["28"]["BLOCK_SIZE_N"], 32)
        for m in cc.PREFILL_MS:
            self.assertIsNone(table[str(m)])
        # Keys ascend by M so the file reads as a grid.
        self.assertEqual([int(m) for m in table], sorted(int(m) for m in table))


if __name__ == "__main__":
    absltest.main()
