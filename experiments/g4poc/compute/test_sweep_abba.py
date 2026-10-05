import os
import sys

from absl.testing import absltest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_abba  # noqa: E402


def _sweep(ref, p90s, tput=100.0):
    return {"ref": ref, "points": [
        {"load": {"name": f"inflight-C{c}", "concurrency": c},
         "summary": {"e2e_p90_s": p90, "output_tok_s_per_gpu": tput, "prefix_cache_hit_rate": 0.5}}
        for c, p90 in p90s.items()]}


class CompareTest(absltest.TestCase):
    def test_gain_is_geomean_over_both_pairs(self):
        a1, a2 = _sweep("base", {8: 4.0, 12: 6.0}), _sweep("base", {8: 4.0, 12: 6.0})
        b1, b2 = _sweep("c", {8: 2.0, 12: 6.0}, 200.0), _sweep("c", {8: 8.0, 12: 3.0}, 50.0)
        rows = {r["concurrency"]: r for r in sweep_abba.compare(a1, b1, b2, a2)}
        self.assertAlmostEqual(rows[8]["e2e_p90_gain"], 1.0)  # 2x faster, then 2x slower
        self.assertAlmostEqual(rows[12]["e2e_p90_gain"], 2 ** 0.5)
        self.assertAlmostEqual(rows[12]["output_tput_gain"], 1.0)
        self.assertAlmostEqual(rows[8]["control_drift_e2e_p90"], 1.0)

    def test_control_drift_compares_first_and_last_control(self):
        a1, a2 = _sweep("base", {12: 6.0}, 100.0), _sweep("base", {12: 5.0}, 110.0)
        b = _sweep("c", {12: 5.5})
        row = sweep_abba.compare(a1, b, b, a2)[0]
        self.assertAlmostEqual(row["control_drift_e2e_p90"], 1.2)
        self.assertAlmostEqual(row["control_drift_tput"], 1.1)

    def test_only_points_every_sweep_reached(self):
        full, short = _sweep("base", {8: 4.0, 16: 9.0}), _sweep("c", {8: 4.0})
        self.assertEqual([r["concurrency"] for r in sweep_abba.compare(full, short, short, full)], [8])


if __name__ == "__main__":
    absltest.main()
