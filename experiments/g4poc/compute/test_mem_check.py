import os
import sys

from absl.testing import absltest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mem_check  # noqa: E402


def _lines(levels, dt=0.1):
    return [f"2026/10/05 20:00:{i * dt:06.3f}, {m}" for i, m in enumerate(levels)]


class MemCheckTest(absltest.TestCase):
    def test_short_spike_is_not_the_plateau(self):
        levels = [30000] * 30 + [31700] * 3 + [31000] * 30  # 0.3 s spike over a 3 s plateau
        res = mem_check.check(mem_check.parse(_lines(levels)), capacity_mib=32111)
        self.assertEqual(res["peak_mib"], 31700)
        self.assertEqual(res["plateau_mib"], 31000)
        self.assertTrue(res["plateau_ok"])
        self.assertFalse(res["peak_ok"])

    def test_held_level_over_limit_fails(self):
        res = mem_check.check(mem_check.parse(_lines([31650] * 20)), capacity_mib=32111)
        self.assertFalse(res["plateau_ok"])

    def test_skips_unparsable_lines(self):
        self.assertLen(mem_check.parse(["garbage", "2026/10/05 20:00:00.000, [N/A]"] + _lines([1])), 1)


if __name__ == "__main__":
    absltest.main()
