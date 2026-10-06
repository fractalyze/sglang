import os
import sys

import torch
from absl.testing import absltest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k3_routing_capture as rc  # noqa: E402


def _counts(layers, tokens_by_expert):
    c = torch.zeros(layers, 128, dtype=torch.int32)
    for e, n in tokens_by_expert.items():
        c[:, e] = n
    return c


class RoutingTest(absltest.TestCase):
    def test_decode_passes_and_summary(self):
        # A 2-token decode pass: 16 slots on 10 experts; a 2,048-token extend pass is dropped.
        dec = _counts(3, {**{e: 2 for e in range(6)}, **{e: 1 for e in range(6, 10)}})
        ext = _counts(3, {e: 128 for e in range(128)})
        passes = rc.decode_passes([{"global_physical_count": dec}, {"global_physical_count": ext}])
        self.assertLen(passes, 1)
        self.assertEqual(passes[0]["tokens"], 2)
        s = rc.summarize(passes)[2]
        self.assertEqual(s["n_passes"], 1)
        self.assertAlmostEqual(s["distinct_experts_per_layer"], 10.0)
        self.assertAlmostEqual(s["max_tokens_per_expert"], 2.0)
        self.assertAlmostEqual(s["uniform_expectation"], 128 * (1 - (1 - 8 / 128) ** 2))


if __name__ == "__main__":
    absltest.main()
