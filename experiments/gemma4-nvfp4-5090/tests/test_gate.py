import math
import os
import sys

from absl.testing import absltest, parameterized

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gate import fidelity, quality, runner, sol, stats  # noqa: E402


def _leg(prefill, decode, w1_decode=2.55, w32_wall=4.0):
    w8 = [{"streams": [{"ttft_s": prefill / 8, "e2e_s": prefill / 8 + decode / 8, "output_tokens": 128}] * 8}]
    w1 = [{"streams": [{"ttft_s": 0.1, "e2e_s": 0.1 + w1_decode, "output_tokens": 256}]}]
    w32 = [{"wall_s": w32_wall, "streams": [{"ttft_s": 1, "e2e_s": 4, "output_tokens": 128}] * 32}]
    return stats.leg_sums({"workloads": {"W8": w8, "W1": w1, "W32": w32}})


class StatsTest(absltest.TestCase):
    def test_identical_legs_give_unit_gains(self):
        s = stats.summarize_pairs([_leg(1, 2)] * 4, [_leg(1, 2)] * 4)
        for m in stats.GATED_METRICS:
            self.assertAlmostEqual(s["overall"][m], 1.0)

    def test_composite_weights_decode_three_quarters(self):
        s = stats.summarize_pairs([_leg(2, 2)] * 4, [_leg(1, 1)] * 4)
        self.assertAlmostEqual(s["overall"]["w8_prefill_gain"], 2.0)
        self.assertAlmostEqual(s["overall"]["w8_composite"], 2.0)
        s = stats.summarize_pairs([_leg(1, 2)] * 4, [_leg(1, 1)] * 4)
        self.assertAlmostEqual(s["overall"]["w8_composite"], 2 ** 0.75)

    def test_ratio_of_sums_not_mean_of_ratios(self):
        s = stats.summarize_pairs([_leg(1, 1), _leg(1, 3)], [_leg(1, 1), _leg(1, 1)])
        self.assertAlmostEqual(s["overall"]["w8_decode_gain"], 4 / 2)

    def test_bar_has_one_percent_floor(self):
        self.assertEqual(stats.promotion_bar(0.001), 0.01)
        self.assertAlmostEqual(stats.promotion_bar(0.005), 0.015)

    def test_verdict_requires_clearing_bar(self):
        noise = {m: 0.002 for m in stats.GATED_METRICS}
        win = stats.summarize_pairs([_leg(1, 1.05)] * 4, [_leg(1, 1.0)] * 4)
        self.assertTrue(stats.timing_verdict(win, noise)["promote"])
        tiny = stats.summarize_pairs([_leg(1, 1.005)] * 4, [_leg(1, 1.0)] * 4)
        self.assertFalse(stats.timing_verdict(tiny, noise)["promote"])

    def test_w1_regression_blocks_promotion(self):
        noise = {m: 0.002 for m in stats.GATED_METRICS}
        s = stats.summarize_pairs([_leg(1, 1.05)] * 4, [_leg(1, 1.0, w1_decode=3.0)] * 4)
        v = stats.timing_verdict(s, noise)
        self.assertFalse(v["checks"]["w1_tpot_no_regression"])
        self.assertFalse(v["promote"])


class FidelityTest(parameterized.TestCase):
    @parameterized.parameters(
        ([1, 2, 3], [1, 2, 3], -1),
        ([1, 2, 3], [1, 9, 3], 1),
        ([1, 2], [1, 2, 3], 2),
    )
    def test_first_divergence(self, a, b, want):
        self.assertEqual(fidelity.first_divergence(a, b), want)

    def test_match_rate_counts_length_mismatch(self):
        self.assertAlmostEqual(fidelity.token_match_rate([1, 2], [1, 2, 3, 4]), 0.5)

    def test_kl_zero_for_identical_and_positive_otherwise(self):
        p = [(-0.1, 5), (-2.5, 7), (-4.0, 9)]
        self.assertAlmostEqual(fidelity.topk_kl(p, p), 0.0, places=12)
        q = [(-0.3, 5), (-1.5, 7), (-4.0, 11)]
        self.assertGreater(fidelity.topk_kl(p, q), 0.0)

    def test_compare_only_scores_kl_up_to_divergence(self):
        top = [[(-0.1, 1), (-3.0, 2)]] * 3
        ref = [{"id": "a", "category": "c", "prompt_tokens": 4, "output_ids": [1, 1, 1], "top_logprobs": top}]
        cand = [{"id": "a", "category": "c", "prompt_tokens": 4, "output_ids": [1, 2, 2], "top_logprobs": top}]
        cmp = fidelity.compare(ref, cand)
        self.assertEqual(cmp["per_prompt"][0]["first_divergence"], 1)
        self.assertEqual(cmp["n_kl_positions"], 2)

    def test_thresholds_have_floors(self):
        t = fidelity.thresholds_from_calibration({"kl_mean": 0.0, "kl_p99": 0.0, "min_token_match_rate": 1.0,
                                                  "n_diverged": 0})
        self.assertGreater(t["kl_mean_max"], 0)
        self.assertGreater(t["kl_p99_max"], 0)


class QualityTest(parameterized.TestCase):
    @parameterized.parameters(
        ("so 3+4 = 7\n#### 7", 7.0),
        ("#### 1,234", 1234.0),
        ("answer is 42.", 42.0),
        ("no number", None),
    )
    def test_gsm8k_answer(self, text, want):
        self.assertEqual(quality.gsm8k_answer(text), want)

    def test_tool_call_scoring(self):
        exp = {"name": "set_timer", "arguments": {"minutes": 5, "label": "tea"}}
        self.assertTrue(quality.tool_call_correct(
            quality.parse_tool_call('```json\n{"name": "set_timer", "arguments": {"minutes": "5", "label": "Tea"}}\n```'),
            exp))
        self.assertFalse(quality.tool_call_correct({"name": "set_timer", "arguments": {"minutes": 6, "label": "tea"}},
                                                   exp))

    def test_tool_items_are_deterministic(self):
        self.assertEqual(quality.tool_items(), quality.tool_items())


class RunnerTest(absltest.TestCase):
    def test_abba_alternates(self):
        self.assertEqual(runner.abba_order(3), [["control", "candidate"], ["candidate", "control"],
                                                ["control", "candidate"]])

    def test_undeclared_server_arg_diff(self):
        a = {"server_info": {"attention_backend": "triton", "random_seed": 1}}
        b = {"server_info": {"attention_backend": "fa3", "random_seed": 2}}
        self.assertEqual(runner.server_arg_diff(a, b, [])["undeclared"], ["attention_backend"])
        self.assertEqual(runner.server_arg_diff(a, b, ["--attention-backend", "fa3"])["undeclared"], [])


class SolTest(absltest.TestCase):
    def setUp(self):
        cfg = {"num_key_value_heads": 8, "head_dim": 256, "num_global_key_value_heads": 2, "global_head_dim": 512,
               "attention_k_eq_v": True, "sliding_window": 1024,
               "layer_types": ["sliding_attention"] * 5 + ["full_attention"], "num_experts": 128}
        self._orig = sol._text_config
        sol._text_config = lambda: cfg

    def tearDown(self):
        sol._text_config = self._orig

    def test_kv_k_eq_v_counts_one_copy(self):
        kv = sol.kv_bytes_per_token(1.0)
        self.assertEqual(kv["sliding"], 8 * 256 * 2)
        self.assertEqual(kv["full"], 2 * 512)

    def test_sliding_kv_read_caps_at_window(self):
        weights = {"bytes_by_component": {"attention_weights": 0, "dense_mlp": 0, "router": 0, "norms_misc": 0,
                                          "lm_head_tied_embed": 0}, "bytes_per_expert_by_layer": {}}
        short = sol.decode_step_bytes([1024], {}, 1.0, weights)["kv_read"]
        long = sol.decode_step_bytes([4096], {}, 1.0, weights)["kv_read"]
        self.assertEqual(long - short, 3072 * 2 * 512)

    def test_distinct_experts_skips_other_batch_sizes(self):
        class Count:
            def __init__(self, rows):
                self.rows = rows
                self.shape = (len(rows),)

            def __getitem__(self, i):
                return _Row(self.rows[i])

        recs = [{"input_ids": [0] * 2, "extend_seq_lens": None, "global_physical_count": Count([[1, 0, 3]])},
                {"input_ids": [0] * 3, "extend_seq_lens": None, "global_physical_count": Count([[1, 1, 1]])}]
        self.assertEqual(sol.distinct_experts_from_records(recs, batch=2), {0: 2.0})


class _Row:
    def __init__(self, vals):
        self.vals = vals

    def __gt__(self, x):
        return _Row([v > x for v in self.vals])

    def sum(self):
        return sum(self.vals)


if __name__ == "__main__":
    absltest.main()
