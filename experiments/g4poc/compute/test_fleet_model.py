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
# HiCache: 24 in flight at 3 turns/s, 8 s mean E2E, within a 10 s p90.
HICACHE = _sweep([(24, 3.0, 8.0, 9.5, 0.74)])


def _pols(**kw):
    return {p["name"]: p for p in fm.policies(CACHED, NOCACHE, HICACHE, host_gb=(12,), **kw)}


class FleetModelTest(absltest.TestCase):
    def test_policies_and_caps(self):
        pols = _pols()
        self.assertAlmostEqual(pols["a_sticky_device"]["cap"], 60000 / fm.TOKENS_PER_STORED_SESSION)
        self.assertIsNone(pols["b_drop_idle"]["cap"])
        self.assertEqual([p["inflight"] for p in pols["b_lru_oversubscribed"]["points"]], [16])
        # The host pool mirrors the device (write_through): the larger of the two.
        self.assertAlmostEqual(pols["c_sticky_host_12gb"]["cap"], 12 / fm.HOST_GB_PER_SESSION)  # ~52
        self.assertAlmostEqual(fm.storage_cap(60000, 1), 60000 / fm.TOKENS_PER_STORED_SESSION)

    def test_zero_think_time_is_the_inflight_point(self):
        row = fm.model(list(_pols().values()), sessions=100, slo_s=10.0, think=[0])["rows"][0]
        a, b = row["a_sticky_device"], row["b_drop_idle"]
        cap = 60000 / fm.TOKENS_PER_STORED_SESSION  # ~9.2 stored sessions
        self.assertAlmostEqual(a["sessions_per_gpu"], cap)  # 2.4 x 5 = 12 in flight, capped by the pool
        self.assertTrue(a["capped"])
        self.assertAlmostEqual(b["sessions_per_gpu"], 8.0)  # 1.25 x 6.4
        self.assertEqual(b["gpus"], 13)
        self.assertIsNone(row["b_lru_oversubscribed"])  # its only point misses the SLO
        self.assertAlmostEqual(row["c_sticky_host_12gb"]["sessions_per_gpu"], 24.0)  # compute-bound at T = 0

    def test_crossover_where_dropping_matches_the_memory_cap(self):
        # n_b(T) = 1.25 (T + 6.4) reaches the cap of 60,000 / 6,500 = 9.23 at T = 9.23 / 1.25 - 6.4 = 0.98 s.
        m = fm.model(list(_pols().values()), sessions=100, slo_s=10.0, think=[0, 30])
        self.assertAlmostEqual(m["crossover_vs_drop_idle_s"]["a_sticky_device"], 1.0)  # first 0.25 s step past 0.98
        late = m["rows"][1]
        self.assertLess(late["b_drop_idle"]["gpus"], late["a_sticky_device"]["gpus"])

    def test_cost_uses_turn_rate_at_think_time(self):
        row = fm.policy_row({"inflight": 8, "turns_per_s": 2.0, "e2e_mean_s": 4.0, "e2e_p90_s": 5.0,
                             "output_tokens": 200.0, "prompt_tokens": 5800.0, "hit": 0.7}, 6.0, 20)
        # 2 x (6 + 4) = 20 sessions per GPU -> 1 GPU; 20 sessions / 10 s x 200 tokens = 400 out tok/s.
        self.assertEqual(row["gpus"], 1)
        self.assertAlmostEqual(row["usd_per_mtok_output"]["1.00"], 1.0 / (400 * 3600) * 1e6)

    def test_points_from_triples_follow_littles_law(self):
        like = fm.points(CACHED)
        pt = fm.points_from_triples("24:8.0:800", like)[0]
        self.assertAlmostEqual(pt["turns_per_s"], 4.0)  # 800 tok/s / 200 tokens per turn
        self.assertAlmostEqual(pt["e2e_mean_s"], 6.0)  # 24 in flight / 4 turns/s
        pols = {p["name"]: p for p in fm.policies(CACHED, NOCACHE, None, (12,), hicache_points="24:8.0:800")}
        self.assertIn("c_sticky_host_12gb", pols)

    def test_host_sizing_lifts_storage_to_the_compute_bound(self):
        m = fm.model(list(_pols().values()), sessions=100, slo_s=10.0, think=[30])
        z = m["host_sizing"][0]
        self.assertAlmostEqual(z["compute_bound_sessions_per_gpu"], 3.0 * (30 + 8.0))  # 114 sessions
        self.assertAlmostEqual(z["host_gb_needed"], 114 * fm.HOST_GB_PER_SESSION)
        self.assertAlmostEqual(fm.storage_cap(60000, z["host_gb_needed"]), 114.0)

    def test_validation_uses_the_closed_population_think_time(self):
        v = fm.validate(["t30-120:120:30:5:532:24.3:0.002"], list(_pols().values()), 10.0)[0]
        self.assertAlmostEqual(v["think_eff_s"], 24.0)  # turn 0 of each 5-turn session has no think time
        self.assertTrue(v["past_storage_bound"])  # 120 > 12 GB / 0.23 GB = 52

    def test_measured_capacity_conservative_and_interpolated(self):
        specs = ["final-hc:30:5.15:48:334:7.07:0.286", "final-hc:30:5.15:72:434:11.41:0.076",
                 "final-hc:30:5.15:96:536:16.55:0.013"]
        row = fm.measured_capacity(specs, slo_s=10.0, sessions=2200)[0]
        self.assertEqual(row["conservative"]["sessions_per_gpu"], 48)
        f = (10.0 - 7.07) / (11.41 - 7.07)
        self.assertAlmostEqual(row["interpolated"]["sessions_per_gpu"], 48 + f * 24)
        self.assertAlmostEqual(row["interpolated"]["out_tok_s_per_gpu"], 334 + f * 100)
        self.assertEqual(row["conservative"]["gpus_for_sessions"], 46)  # ceil(2200 / 48)
        self.assertAlmostEqual(row["conservative"]["usd_per_mtok_output"]["0.70"], 0.70 / (334 * 3600) * 1e6)
        self.assertAlmostEqual(row["think_eff_s"], 30 * (1 - 1 / 5.15))

    def test_failed_or_empty_points_are_dropped(self):
        sweep = _sweep([(8, 2.0, 4.0, 5.0, 0.7)])
        sweep["points"].append({"summary": {"requests_per_s": 0.0, "n_failed": 5}})
        self.assertLen(fm.points(sweep), 1)


if __name__ == "__main__":
    absltest.main()
