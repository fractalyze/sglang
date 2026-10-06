import os
import sys
import tempfile

from absl.testing import absltest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k2_smi_windows as sw  # noqa: E402

LINES = [
    "2026/10/06 12:00:00.000, 2800, 14001, 300.0, 50, 90, 31000",
    "2026/10/06 12:00:01.000, 2700, 14001, 500.0, 60, 95, 31000",
    "2026/10/06 12:00:02.000, 210, 405, 30.0, 40, 0, 31000",  # idle: not counted
    "2026/10/06 12:00:03.000, [N/A], 14001, 300.0, 50, 90, 31000",  # unreadable: skipped
    "2026/10/06 12:00:09.000, 2600, 14001, 400.0, 55, 99, 31000",  # after the run
]


class SmiWindowsTest(absltest.TestCase):
    def test_busy_samples_inside_the_span(self):
        samples = sw.parse(LINES)
        self.assertLen(samples, 4)
        t0 = samples[0][0]
        s = sw.summarize(samples, t0, t0 + 5)
        self.assertEqual(s["n_busy"], 2)
        self.assertAlmostEqual(s["clocks_sm_mean"], 2750)
        self.assertAlmostEqual(s["power_w_mean"], 400)

    def test_run_span_from_the_directory(self):
        with tempfile.TemporaryDirectory() as d:
            run = os.path.join(d, "sweep-x-20261006-120000-build-server-3-abc123")
            os.makedirs(run)
            path = os.path.join(run, "sweep.json")
            open(path, "w").close()
            os.utime(path, (1.0e9, 1.0e9))
            t0, t1 = sw.run_span(run)
            self.assertEqual(t1, 1.0e9)
            self.assertGreater(t0, 1.0e9)


if __name__ == "__main__":
    absltest.main()
