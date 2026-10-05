import os
import sys

from absl.testing import absltest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c4_pick  # noqa: E402


def _rows(g24, g32):
    return [{"concurrency": 24, "e2e_p90_gain": g24}, {"concurrency": 32, "e2e_p90_gain": g32}]


class PickTest(absltest.TestCase):
    def test_best_mean_gain_without_a_losing_point(self):
        self.assertEqual(c4_pick.pick({"a": _rows(1.03, 1.02), "b": _rows(1.10, 0.98), "c": _rows(1.01, 1.02)}), "a")

    def test_none_when_the_mean_stays_under_one_percent(self):
        self.assertEqual(c4_pick.pick({"a": _rows(1.005, 1.012)}), "none")

    def test_missing_point_is_skipped(self):
        self.assertEqual(c4_pick.pick({"a": [{"concurrency": 24, "e2e_p90_gain": 1.2}]}), "none")


if __name__ == "__main__":
    absltest.main()
