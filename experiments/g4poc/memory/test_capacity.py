import os
import sys

from absl.testing import absltest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory import capacity  # noqa: E402

R03_LINES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "baseline", "runs",
                         "r03-m093-swa03-noprefillgraph", "server-key-lines.txt")


class CapacityTest(absltest.TestCase):
    def test_pool_facts_read_r03_log(self):
        with open(R03_LINES) as f:
            facts = capacity.pool_facts(f.read())
        self.assertEqual(facts["full_tokens"], 89571)
        self.assertEqual(facts["swa_tokens"], 26871)
        self.assertAlmostEqual(facts["weights_gib"], 25.12)
        self.assertAlmostEqual(facts["swa_k_gib"], 1.28)
        self.assertEqual(facts["max_running_requests"], 2799)
        self.assertAlmostEqual(facts["available_gpu_mem_gib"], 1.72)

    def test_clean_needs_full_batch_and_no_retraction(self):
        self.assertTrue(capacity.is_clean(16, {"peak_running": 16, "retract_lines": 0}))
        self.assertFalse(capacity.is_clean(17, {"peak_running": 16, "retract_lines": 0}))
        self.assertFalse(capacity.is_clean(17, {"peak_running": 17, "retract_lines": 1}))


if __name__ == "__main__":
    absltest.main()
