"""The committed C1 fused_moe config is the one the gate refs use (sha256 in gate/refs.json) and covers every
token count C1 tuned: a lost entry would send that size to its nearest neighbour's tiles."""

import hashlib
import json
import os

from absl.testing import absltest

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "moe-configs", "c1", "configs", "triton_3_7_1",
                      "E=128,N=704,device_name=NVIDIA_GeForce_RTX_5090,dtype=fp8_w8a8,per_channel_quant=True.json")
SHA256 = "b3bcce12ad59fa5793621ddce8840b3989bb9cb245fda646bd79c0c868ed9576"
TUNED_M = [1, 2, 4, 8, 12, 16, 24, 32, 256, 512, 768, 1024, 1536, 2048, 4096]


class MoeConfigC1Test(absltest.TestCase):
    def test_file_is_the_deployed_one(self):
        with open(CONFIG, "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), SHA256)

    def test_covers_every_tuned_token_count(self):
        with open(CONFIG) as f:
            self.assertEqual(sorted(int(m) for m in json.load(f)), TUNED_M)

    def test_refs_cite_it(self):
        with open(os.path.join(HERE, "..", "gate", "refs.json")) as f:
            refs = json.load(f)
        for name in ("c1-moe-tuned", "final-mem-c1-c2a"):
            self.assertIn(SHA256[:8], refs[name]["description"], name)


if __name__ == "__main__":
    absltest.main()
