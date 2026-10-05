import os
import sys

from absl.testing import absltest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory import capacity, exactness  # noqa: E402

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


class ExactnessTest(absltest.TestCase):
    @staticmethod
    def _run(*outs):
        return {"outputs": {"1": [{"seed": i, "text": t, "output_ids": ids} for i, (t, ids) in enumerate(outs)]}}

    def test_compare_reports_first_mismatch_and_length_change(self):
        control = self._run(("abc", [1, 2, 3]), ("xy", [7, 8]), ("q", [5]))
        candidate = self._run(("abc", [1, 2, 3]), ("xz", [7, 9]), ("q!", [5, 6]))
        rows = exactness.compare(control, candidate)["1"]
        self.assertEqual(rows["compared_on"], "output_ids")
        self.assertEqual(rows["exact"], 1)
        self.assertEqual([r["first_mismatch"] for r in rows["rows"]], [None, 1, 1])

    def test_compare_falls_back_to_text_without_ids(self):
        control = self._run(("abc", None))
        candidate = self._run(("abd", None))
        rows = exactness.compare(control, candidate)["1"]
        self.assertEqual(rows["compared_on"], "text")
        self.assertEqual(rows["rows"][0]["first_mismatch"], 2)


if __name__ == "__main__":
    absltest.main()
