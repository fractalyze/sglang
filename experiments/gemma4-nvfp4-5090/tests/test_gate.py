import math
import os
import sys

from absl.testing import absltest, parameterized

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gate import config, fidelity, hostwatch, quality, runner, server, sol, stats  # noqa: E402


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
        t = fidelity.thresholds_from_calibration(
            {"kl_mean": 0.0, "kl_p99": 0.0, "min_token_match_rate": 1.0, "mean_token_match_rate": 1.0,
             "n_diverged": 0},
            {"kl_mean": 0.0, "kl_p99": 0.0, "min_top1_agreement": 1.0, "mean_top1_agreement": 1.0})
        for k in ("decode_kl_mean_max", "decode_kl_p99_max", "forced_kl_mean_max", "forced_kl_p99_max"):
            self.assertGreater(t[k], 0)

    def test_forced_agreement_counts_reference_tokens(self):
        rows_ok = [[(-0.1, 5), (-3.0, 6)], [(-0.2, 7), (-2.0, 8)]]
        rows_bad = [[(-0.1, 5), (-3.0, 6)], [(-2.0, 7), (-0.2, 8)]]
        ref = [{"id": "a", "category": "c", "output_ids": [5, 7]}]
        cmp = fidelity.compare_forced(ref, [{"id": "a", "top_logprobs": rows_ok}],
                                      [{"id": "a", "top_logprobs": rows_bad}])
        self.assertAlmostEqual(cmp["per_prompt"][0]["top1_agreement"], 0.5)
        thr = {"top1_agreement_min": 0.9, "forced_kl_mean_max": 9, "forced_kl_p99_max": 9,
               "decode_kl_mean_max": 9, "decode_kl_p99_max": 9}
        free = {"kl_mean": 0.0, "kl_p99": 0.0}
        self.assertFalse(fidelity.verdict(free, cmp, thr)["pass"])


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
        a = {"server_info": {"attention_backend": "triton", "random_seed": 1, "startup_time": 1.0}}
        b = {"server_info": {"attention_backend": "fa3", "random_seed": 2, "startup_time": 2.0}}
        self.assertEqual(runner.server_arg_diff(a, b, [])["undeclared"], ["attention_backend"])
        self.assertEqual(runner.server_arg_diff(a, b, ["--attention-backend", "fa3"])["undeclared"], [])

    def test_pool_size_flag_declares_its_derived_keys(self):
        a = {"server_info": {"mem_fraction_static": 0.718, "max_total_num_tokens": 37081, "max_req_input_len": 37075,
                             "launch_command": "--port 1"}}
        b = {"server_info": {"mem_fraction_static": 0.76, "max_total_num_tokens": 51892, "max_req_input_len": 51886,
                             "launch_command": "--port 1 --mem-fraction-static 0.76"}}
        diff = runner.server_arg_diff(a, b, ["--mem-fraction-static", "0.76"])
        self.assertEqual(diff, {"differing_keys": ["mem_fraction_static"], "undeclared": []})

    def test_chunk_size_flag_declares_prefill_graph_shape(self):
        # T1 on bs2: --chunked-prefill-size resizes the prefill cuda-graph config the server reports.
        graphs = lambda n: {"decode": {"max_bs": 32}, "prefill": {"backend": "disabled", "max_bs": n, "bs": [4, n]}}
        a = {"server_info": {"chunked_prefill_size": 4096, "cuda_graph_config": graphs(4096),
                             "launch_command": "--port 1"}}
        b = {"server_info": {"chunked_prefill_size": 8192, "cuda_graph_config": graphs(8192),
                             "launch_command": "--port 1 --chunked-prefill-size 8192"}}
        diff = runner.server_arg_diff(a, b, ["--chunked-prefill-size", "8192"])
        self.assertEqual(diff["undeclared"], [])
        self.assertIn("cuda_graph_config.prefill.max_bs", diff["differing_keys"])
        self.assertEqual(runner.server_arg_diff(a, b, [])["undeclared"], [
            "chunked_prefill_size", "cuda_graph_config.prefill.bs", "cuda_graph_config.prefill.max_bs",
            "launch_command:--chunked-prefill-size"])

    def test_derived_path_does_not_cover_siblings(self):
        a = {"server_info": {"cuda_graph_config": {"decode": {"backend": "full"}, "prefill": {"max_bs": 1}}}}
        b = {"server_info": {"cuda_graph_config": {"decode": {"backend": "piecewise"}, "prefill": {"max_bs": 2}}}}
        diff = runner.server_arg_diff(a, b, ["--chunked-prefill-size", "2"])
        self.assertEqual(diff["undeclared"], ["cuda_graph_config.decode.backend"])

    def test_undeclared_launch_flag_is_flagged(self):
        a = {"server_info": {"launch_command": "--port 1"}}
        b = {"server_info": {"launch_command": "--port 1 --enable-foo"}}
        self.assertEqual(runner.server_arg_diff(a, b, [])["undeclared"], ["launch_command:--enable-foo"])
        self.assertEqual(runner.server_arg_diff(a, b, ["--enable-foo"])["undeclared"], [])


class HostWatchTest(absltest.TestCase):
    def _with(self, mem, swap, foreign):
        orig = (hostwatch.meminfo_gb, hostwatch.foreign_gpu_gb, hostwatch.load1)
        hostwatch.meminfo_gb = lambda: {"mem_available_gb": mem, "swap_used_gb": swap}
        hostwatch.foreign_gpu_gb = lambda own_root_pid=None: foreign
        hostwatch.load1 = lambda: 1.0
        try:
            return hostwatch.preflight()
        finally:
            hostwatch.meminfo_gb, hostwatch.foreign_gpu_gb, hostwatch.load1 = orig

    def test_preflight_passes_on_quiet_host(self):
        self._with(50, 0, 0)

    def test_preflight_refuses_low_ram_swap_or_foreign_gpu(self):
        for args in ((config.MIN_HOST_AVAILABLE_GB - 1, 0, 0), (50, config.MAX_SWAP_USED_GB + 1, 0),
                     (50, 0, config.MAX_FOREIGN_GPU_GB + 1)):
            with self.assertRaises(hostwatch.HostUnsafe):
                self._with(*args)

    def test_phase_follows_server_log(self):
        import tempfile

        d = tempfile.mkdtemp()
        log = os.path.join(d, "s.log")
        with open(log, "w") as f:
            f.write("Load weight begin\n")
        dog = hostwatch.Watchdog(os.getpid(), os.path.join(d, "h.csv"), log)
        dog._advance_phase_from_log()
        self.assertEqual(dog.phase, "weight_load")
        with open(log, "a") as f:
            f.write("Running FlashInfer autotune with cache\n")
        dog._advance_phase_from_log()
        self.assertEqual(dog.phase, "autotune")


class FlushTest(absltest.TestCase):
    def test_flush_retries_while_busy_then_fails_loudly(self):
        class R:
            def __init__(self, code):
                self.status_code, self.text = code, ""

        codes = [400, 400, 200]
        orig = server.requests.post
        server.requests.post = lambda *a, **k: R(codes.pop(0))
        try:
            srv = server.Server.__new__(server.Server)
            srv.url = "http://x"
            self.assertEqual(srv.flush_cache(timeout_s=5), 2)
            server.requests.post = lambda *a, **k: R(400)
            with self.assertRaises(RuntimeError):
                srv.flush_cache(timeout_s=0.2)
        finally:
            server.requests.post = orig


class SolTest(absltest.TestCase):
    def setUp(self):
        cfg = {"num_key_value_heads": 8, "head_dim": 256, "num_global_key_value_heads": 2, "global_head_dim": 512,
               "attention_k_eq_v": True, "sliding_window": 1024,
               "layer_types": ["sliding_attention"] * 5 + ["full_attention"], "num_experts": 128}
        self._orig = sol._text_config
        sol._text_config = lambda: cfg

    def tearDown(self):
        sol._text_config = self._orig

    def test_kv_k_eq_v_still_counts_k_and_v(self):
        kv = sol.kv_bytes_per_token(1.0)
        self.assertEqual(kv["sliding"], 8 * 256 * 2)
        self.assertEqual(kv["full"], 2 * 512 * 2)

    def test_sliding_kv_read_caps_at_window(self):
        weights = {"bytes_by_component": {"attention_weights": 0, "dense_mlp": 0, "router": 0, "norms_misc": 0,
                                          "lm_head_tied_embed": 0}, "bytes_per_expert_by_layer": {}}
        short = sol.decode_step_bytes([1024], {}, 1.0, weights)["kv_read"]
        long = sol.decode_step_bytes([4096], {}, 1.0, weights)["kv_read"]
        self.assertEqual(long - short, 3072 * 2 * 512 * 2)

    def test_distinct_experts_unions_streams_per_step(self):
        import numpy as np

        a = np.array([[[0, 1]], [[2, 3]]])  # 2 decode steps, 1 layer, top-2
        b = np.array([[[1, 5]], [[2, 3]]])
        # step 0: {0,1,5} -> 3; step 1: {2,3} -> 2.
        self.assertEqual(sol.distinct_experts_from_routes([a, b]), {0: 2.5})


if __name__ == "__main__":
    absltest.main()
