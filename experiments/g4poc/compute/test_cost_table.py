import os
import sys

from absl.testing import absltest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_table  # noqa: E402


def _pt(c, p90, out, failed=0):
    return {"load": {"name": f"inflight-C{c}", "concurrency": c},
            "summary": {"e2e_p90_s": p90, "output_tok_s_per_gpu": out, "total_tok_s_per_gpu": out * 30,
                        "prefix_cache_hit_rate": 0.7, "n_failed": failed, "n_ok": 100}}


SWEEP = {"points": [_pt(8, 5.0, 470), _pt(12, 6.3, 570), _pt(16, 11.9, 430), _pt(20, 14.5, 380)]}


class CostTableTest(absltest.TestCase):
    def test_cheapest_is_not_the_most_in_flight(self):
        rows = {r["slo_s"]: r for r in cost_table.rows_for(SWEEP, [6, 10, 15])}
        self.assertEqual(rows[6]["max_inflight"], 8)
        self.assertEqual((rows[15]["max_inflight"], rows[15]["cheapest"]["inflight"]), (20, 12))
        self.assertAlmostEqual(rows[10]["cheapest"]["usd_per_mtok_output"]["0.70"], 0.70 / (570 * 3600) * 1e6)

    def test_no_point_meets_a_tight_slo(self):
        self.assertIsNone(cost_table.rows_for(SWEEP, [3])[0]["cheapest"])

    def test_failed_points_never_qualify(self):
        rows = cost_table.rows_for({"points": [_pt(8, 5.0, 470), _pt(12, 6.0, 570, failed=1)]}, [10])
        self.assertEqual(rows[0]["cheapest"]["inflight"], 8)


if __name__ == "__main__":
    absltest.main()
