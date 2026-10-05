import os
import sys

from absl.testing import absltest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c4_pick  # noqa: E402


def _rows(g12, g20):
    return [{"concurrency": 12, "e2e_p90_gain": g12}, {"concurrency": 20, "e2e_p90_gain": g20}]


class PickTest(absltest.TestCase):
    def test_best_gain_at_12_that_holds_at_20(self):
        self.assertEqual(c4_pick.pick({"a": _rows(1.03, 1.0), "b": _rows(1.05, 0.98), "c": _rows(1.02, 1.01)}), "a")

    def test_none_when_nothing_clears_one_percent(self):
        self.assertEqual(c4_pick.pick({"a": _rows(1.005, 1.02)}), "none")

    def test_missing_point_is_skipped(self):
        self.assertEqual(c4_pick.pick({"a": [{"concurrency": 12, "e2e_p90_gain": 1.2}]}), "none")


if __name__ == "__main__":
    absltest.main()
