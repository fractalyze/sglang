import os
import sys

from absl.testing import absltest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fleet_model as fm  # noqa: E402


def _sweep(points, pool=60000):
    return {"server_info": {"max_total_num_tokens": pool}, "points": [
        {"summary": {"inflight_mean": a, "requests_per_s": r, "e2e_mean_s": e, "e2e_p90_s": p90,
                     "mean_output_tokens": 200.0, "mean_prompt_tokens": 5800.0, "prefix_cache_hit_rate": hit,
                     "n_failed": 0}} for a, r, e, p90, hit in points]}


# Cached: 8 and 12 in flight below the cliff, 16 past it (hit collapses). 60,000 / 6,000 = 10 sessions per GPU.
CACHED = _sweep([(8, 2.0, 4.0, 5.0, 0.74), (12, 2.4, 5.0, 7.0, 0.70), (16, 1.6, 10.0, 14.0, 0.2)])
# No cache: every turn re-prefills; 8 in flight meets a 10 s SLO, 12 does not.
NOCACHE = _sweep([(4, 1.0, 4.0, 5.0, 0.0), (8, 1.25, 6.4, 9.0, 0.0), (12, 1.3, 9.2, 12.0, 0.0)])


class FleetModelTest(absltest.TestCase):
    def test_zero_think_time_is_the_inflight_point(self):
        res = fm.model(CACHED, NOCACHE, sessions=100, slo_s=10.0, think=[0])
        a, b = res["rows"][0]["sticky_cached"], res["rows"][0]["drop_idle"]
        self.assertAlmostEqual(res["cached_sessions_per_gpu_cap"], 10.0)
        self.assertAlmostEqual(a["sessions_per_gpu"], 10.0)  # 2.4 x 5 = 12 in flight, capped by the pool at 10
        self.assertTrue(a["memory_bound"])
        self.assertAlmostEqual(b["sessions_per_gpu"], 8.0)  # 1.25 x 6.4
        self.assertEqual(b["gpus"], 13)

    def test_past_the_cliff_points_never_serve_the_cached_policy(self):
        res = fm.model(CACHED, NOCACHE, sessions=100, slo_s=20.0, think=[0])
        self.assertEqual(res["rows"][0]["sticky_cached"]["point_inflight"], 12)

    def test_crossover_where_dropping_matches_the_memory_cap(self):
        # n_b(T) = max(1.0 (T + 4), 1.25 (T + 6.4)) reaches the cap of 10 at T = 10 / 1.25 - 6.4 = 1.6 s.
        res = fm.model(CACHED, NOCACHE, sessions=100, slo_s=10.0, think=[0, 30])
        self.assertAlmostEqual(res["crossover_think_s"], 1.75)  # first 0.25 s step at or past 1.6
        late = res["rows"][1]
        self.assertLess(late["drop_idle"]["gpus"], late["sticky_cached"]["gpus"])

    def test_cost_uses_turn_rate_at_think_time(self):
        row = fm.policy_row({"inflight": 8, "turns_per_s": 2.0, "e2e_mean_s": 4.0, "e2e_p90_s": 5.0,
                             "output_tokens": 200.0, "prompt_tokens": 5800.0, "hit": 0.7}, 6.0, 20)
        # 2 x (6 + 4) = 20 sessions per GPU -> 1 GPU; 20 sessions / 10 s x 200 tokens = 400 out tok/s.
        self.assertEqual(row["gpus"], 1)
        self.assertAlmostEqual(row["usd_per_mtok_output"]["1.00"], 1.0 / (400 * 3600) * 1e6)


if __name__ == "__main__":
    absltest.main()
