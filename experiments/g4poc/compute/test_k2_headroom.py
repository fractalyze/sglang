import os
import sys

from absl.testing import absltest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k2_headroom as hr  # noqa: E402


class HeadroomTest(absltest.TestCase):
    def test_decode_only_gain_is_diluted_by_the_decode_share(self):
        # 1 ms of a 10 ms step, decode 80% of wall: 8% of wall.
        self.assertAlmostEqual(hr.wall_gain(10.0, 0.8, 1.0), 0.08)

    def test_prefill_gain_counts_on_the_prefill_part(self):
        self.assertAlmostEqual(hr.wall_gain(10.0, 0.8, 0.0, prefill_gain=0.05), 0.01)

    def test_after_shrinks_the_step_and_the_decode_share(self):
        step, share = hr.after(10.0, 0.8, 2.0)
        self.assertAlmostEqual(step, 8.0)
        # Decode 0.8 -> 0.64 of the old wall, prefill stays 0.2: share 0.64 / 0.84.
        self.assertAlmostEqual(share, 0.64 / 0.84)

    def test_stack_applies_levers_in_order(self):
        point = {"name": "P", "step_ms": 10.0, "decode_share": {"lo": 0.8, "hi": 0.8}}
        levers = [{"name": "a", "save_ms": {"P": {"lo": 2.0, "hi": 2.0}}},
                  {"name": "b", "save_ms": {"P": {"lo": 1.0, "hi": 1.0}}}]
        a, b = hr.stack(point, levers, "lo")
        self.assertAlmostEqual(a["gain"], 0.16)
        self.assertAlmostEqual(b["step_ms"], 8.0)
        self.assertAlmostEqual(b["gain"], (0.64 / 0.84) * 1.0 / 8.0)
        # The stacked gains compose to the total wall saved: (2 + 1) ms of decode at 0.8 / 10 ms.
        self.assertAlmostEqual(1 - (1 - a["gain"]) * (1 - b["gain"]), 0.8 * 3.0 / 10.0)

    def test_predicted_metrics(self):
        base = {"tok_s": 900.0, "p90_s": 9.0, "usd_per_mtok": 0.2, "sessions": 70.0}
        p = hr.predicted(base, 0.1, closed_loop=True)
        self.assertAlmostEqual(p["tok_s"], 1000.0)
        self.assertAlmostEqual(p["p90_s"], 8.1)
        self.assertAlmostEqual(p["usd_per_mtok"], 0.18)
        self.assertAlmostEqual(hr.predicted(base, 0.1, closed_loop=False)["sessions"], 70.0 / 0.9)


if __name__ == "__main__":
    absltest.main()
