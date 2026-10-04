import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

from absl.testing import absltest, parameterized

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gate import config, fidelity, hostwatch, quality, runner, server, sol, specrule, stats, treehash  # noqa: E402


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

    def test_ci_is_reported_but_never_a_check(self):
        noise = {m: 0.002 for m in stats.GATED_METRICS}
        legs = [_leg(1, 1.05, w1_decode=d) for d in (2.3, 2.9, 2.3, 2.9)]
        s = stats.summarize_pairs(legs, [_leg(1, 1.0)] * 4)
        v = stats.timing_verdict(s, noise)
        self.assertFalse(v["ci95_narrower_than_bar"]["w1_tpot_gain"])
        self.assertTrue(v["ci95_narrower_than_bar"]["w8_decode_gain"])
        self.assertNotIn("w1_tpot_ci", " ".join(v["checks"]))
        self.assertTrue(v["promote"])

    def test_pair_ci95_uses_student_t(self):
        logs = [math.log(1.1), math.log(1.3)]
        ci = stats.pair_ci95(logs)
        half = 12.706 * (math.log(1.3) - math.log(1.1)) / math.sqrt(2) / math.sqrt(2)
        self.assertAlmostEqual(ci["half_width"], math.expm1(half))
        self.assertAlmostEqual(ci["low"] * ci["high"], 1.1 * 1.3)
        self.assertEqual(stats.pair_ci95([0.0, 0.0, 0.0])["half_width"], 0.0)


class DecideOnTest(absltest.TestCase):
    noise = {m: 0.002 for m in stats.GATED_METRICS}

    def test_default_rule_keeps_old_check_names(self):
        s = stats.summarize_pairs([_leg(1, 1.05)] * 4, [_leg(1, 1.0)] * 4)
        v = stats.timing_verdict(s, self.noise)
        self.assertEqual(list(v["checks"]), ["w8_composite_clears_bar", "w1_tpot_no_regression", "w8_no_regression",
                                             "enough_pairs"])
        self.assertEqual(v["decided_on"], "w8_composite")

    def test_w1_win_with_neutral_w8_promotes_only_when_deciding_on_w1(self):
        # T2 shape: W1 TPOT -2%, W8 and W32 unchanged.
        s = stats.summarize_pairs([_leg(1, 1.0, w1_decode=2.55)] * 4, [_leg(1, 1.0, w1_decode=2.50)] * 4)
        self.assertFalse(stats.timing_verdict(s, self.noise)["promote"])
        v = stats.timing_verdict(s, self.noise, "w1_tpot_gain")
        self.assertTrue(v["promote"])
        self.assertEqual(set(v["checks"]), {"w1_tpot_clears_bar", "w8_no_regression", "w32_tput_no_regression",
                                            "w1_tpot_no_regression", "enough_pairs"})

    def test_non_default_rule_guards_w32(self):
        s = stats.summarize_pairs([_leg(1, 1.0, w1_decode=2.55)] * 4,
                                  [_leg(1, 1.0, w1_decode=2.50, w32_wall=4.2)] * 4)
        v = stats.timing_verdict(s, self.noise, "w1_tpot_gain")
        self.assertFalse(v["checks"]["w32_tput_no_regression"])
        self.assertFalse(v["promote"])

    def test_unknown_metric_is_refused(self):
        s = stats.summarize_pairs([_leg(1, 1.0)] * 4, [_leg(1, 1.0)] * 4)
        with self.assertRaises(ValueError):
            stats.timing_verdict(s, self.noise, "w8_prefill_gain")


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

    def test_paired_delta_counts_discordant_items(self):
        control = [True] * 90 + [False] * 10
        candidate = [True] * 88 + [False] * 2 + [True] * 1 + [False] * 9
        d = quality.paired_delta(control, candidate)
        self.assertEqual((d["control_only_correct"], d["candidate_only_correct"]), (2, 1))
        self.assertAlmostEqual(d["control_pt"], 90.0)
        self.assertAlmostEqual(d["candidate_pt"], 89.0)
        self.assertAlmostEqual(d["delta_pt"], -1.0)
        lo, hi = d["ci95_pt"]
        self.assertLess(lo, -1.0)
        self.assertGreater(hi, 0.0)

    def test_paired_delta_interval_keeps_width_without_discordance(self):
        d = quality.paired_delta([True] * 50, [True] * 50)
        self.assertEqual(d["delta_pt"], 0.0)
        self.assertLess(d["ci95_pt"][0], 0.0)
        self.assertGreater(d["ci95_pt"][1], 0.0)
        self.assertEqual(d["mcnemar_exact_p"], 1.0)

    def test_paired_delta_rejects_unpaired_arms(self):
        with self.assertRaises(ValueError):
            quality.paired_delta([True], [True, False])

    def test_mcnemar_exact_p(self):
        # Binomial(10, 1/2): P(X <= 1) = 11/1024, doubled.
        self.assertAlmostEqual(quality._mcnemar_exact_p(1, 9), 22 / 1024)
        self.assertEqual(quality._mcnemar_exact_p(5, 5), 1.0)

    def test_compare_applies_ci_floor_and_tool_no_drop(self):
        def res(gsm, tool):
            return {"gsm8k": {"correct": gsm}, "tool_json": {"correct": tool}}

        same = [True] * 1300 + [False] * 19
        # Large n, no change: the CI lower bound sits well inside -1 pt.
        self.assertTrue(quality.compare(res(same, [True] * 40), res(same, [True] * 40))["pass"])
        # Thirty net losses on 1,319 items push the CI lower bound below -1 pt.
        worse = [False] * 30 + same[30:]
        out = quality.compare(res(same, [True] * 40), res(worse, [True] * 40))
        self.assertFalse(out["checks"]["gsm8k_ci_low"])
        # One tool-JSON miss fails regardless of GSM8K.
        out = quality.compare(res(same, [True] * 40), res(same, [False] + [True] * 39))
        self.assertFalse(out["checks"]["tool_json_no_drop"])
        self.assertTrue(out["checks"]["gsm8k_ci_low"])

    def test_verdict_refuses_mismatched_sample_sizes(self):
        base = {"gsm8k": {"n": 200, "accuracy_pt": 97.0}, "tool_json": {"n": 40, "accuracy_pt": 100.0}}
        cand = {"gsm8k": {"n": 1319, "accuracy_pt": 96.0}, "tool_json": {"n": 40, "accuracy_pt": 100.0}}
        self.assertIsNone(quality.verdict(cand, base)["pass"])


class RunnerTest(absltest.TestCase):
    def test_gated_workloads_time_the_same_prompts_in_every_pair(self):
        for wl in (config.W1, config.W8):
            seeds = [runner.timing_seed(wl, f"nonce/pair{k}", 3) for k in range(4)]
            self.assertEqual(len(set(seeds)), 1, wl.name)
            self.assertNotEqual(runner.timing_seed(wl, "p", 0), runner.timing_seed(wl, "p", 1))
        self.assertNotEqual(runner.timing_seed(config.W1, "p", 0), runner.timing_seed(config.W8, "p", 0))
        w32 = [runner.timing_seed(config.W32, f"nonce/pair{k}", 0) for k in range(4)]
        self.assertEqual(len(set(w32)), 4)

    def test_cache_is_flushed_between_fidelity_passes(self):
        calls = []

        class Srv:
            url = "http://x"

            def flush_cache(self):
                calls.append("flush")

        orig = fidelity.run, fidelity.run_forced, fidelity.load_json
        fidelity.run = lambda url: calls.append("run") or "out"
        fidelity.run_forced = lambda url, ref: calls.append("forced") or "forced"
        fidelity.load_json = lambda path: {}
        try:
            self.assertEqual(runner.fidelity_passes(Srv()), ("out", "forced"))
        finally:
            fidelity.run, fidelity.run_forced, fidelity.load_json = orig
        self.assertEqual(calls, ["run", "flush", "forced"])

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


class AgreementTest(parameterized.TestCase):
    def _meta(self, candidate="t2", numerics_unchanged=False):
        return {"control": {"name": "base"}, "candidate": {"name": candidate, "numerics_unchanged": numerics_unchanged},
                "config": {"workloads": [{"name": "W8", "decode": 128}]}}

    def test_threshold_from_identical_aa_uses_margin(self):
        self.assertAlmostEqual(runner.agreement_threshold([1.0] * 6), 1.0 - config.AGREEMENT_MIN_MARGIN)

    def test_threshold_widens_with_aa_spread(self):
        means = [0.9, 1.0, 0.9, 1.0]
        sd = (sum((x - 0.95) ** 2 for x in means) / 3) ** 0.5
        self.assertAlmostEqual(runner.agreement_threshold(means), 0.9 - config.NOISE_SIGMAS * sd)

    @parameterized.parameters(
        # (candidate, numerics_unchanged, mean, ok): T2 on bs2 agreed 0.16 with a pass on teacher-forced fidelity.
        ("t2", False, 0.16, True),
        ("t2", True, 0.16, False),
        ("t2", True, 0.99, True),
        ("base", False, 0.16, False),
    )
    def test_hard_only_for_numerics_unchanged_or_aa(self, candidate, numerics_unchanged, mean, ok):
        noise = {"timed_output_agreement": {"from_exp": "AA-x", "threshold": 0.98}}
        check = runner.agreement_check(self._meta(candidate, numerics_unchanged), [{"mean": mean}], noise)
        self.assertEqual(check["ok"], ok)
        self.assertEqual(check["above_threshold"], mean >= 0.98)

    def test_uncalibrated_falls_back_to_floor(self):
        check = runner.agreement_check(self._meta(numerics_unchanged=True), [{"mean": 0.6}], {})
        self.assertEqual((check["threshold"], check["calibrated_from"], check["ok"]),
                         (config.TIMED_OUTPUT_AGREEMENT_MIN, None, True))

    def test_short_timed_stream_fails_integrity(self):
        leg = {"workloads": {"W8": [{"streams": [{"output_tokens": 128}, {"output_tokens": 128}]}]}}
        self.assertTrue(runner.timed_streams_full_length(leg, self._meta()))
        leg["workloads"]["W8"][0]["streams"][1]["output_tokens"] = 7
        self.assertFalse(runner.timed_streams_full_length(leg, self._meta()))


class RefTest(absltest.TestCase):
    def test_numerics_unchanged_is_not_inherited(self):
        refs = {"base": {"commit": "c", "server_args": ["--a"], "numerics_unchanged": True},
                "child": {"extends": "base", "server_args": ["--b"]}}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(refs, f)
        old, server.REFS_PATH = server.REFS_PATH, f.name
        try:
            child = server.load_ref("child")
        finally:
            server.REFS_PATH = old
            os.unlink(f.name)
        self.assertEqual(child["server_args"], ["--a", "--b"])
        self.assertFalse(child["numerics_unchanged"])


class HarnessCommitTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.repo = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.repo)
        self.exp = os.path.join(self.repo, "experiments")
        os.makedirs(os.path.join(self.exp, "gate", "__pycache__"))
        with open(os.path.join(self.exp, "gate", "runner.py"), "w") as f:
            f.write("x = 1\n")
        with open(os.path.join(self.exp, "README"), "w") as f:
            f.write("readme\n")

    def _git(self, *args):
        return subprocess.run(["git", "-C", self.repo, "-c", "user.name=t", "-c", "user.email=t@t", *args],
                              check=True, capture_output=True, text=True).stdout.strip()

    def _stamp(self, commit, tree):
        with open(os.path.join(self.exp, treehash.STAMP), "w") as f:
            json.dump({"commit": commit, "tree_sha256": tree}, f)

    def test_tree_hash_ignores_caches_and_stamp(self):
        before = treehash.tree_sha256(self.exp)
        with open(os.path.join(self.exp, "gate", "__pycache__", "runner.cpython-312.pyc"), "w") as f:
            f.write("bytecode")
        self._stamp("abc", before)
        self.assertEqual(treehash.tree_sha256(self.exp), before)
        with open(os.path.join(self.exp, "gate", "runner.py"), "a") as f:
            f.write("y = 2\n")
        self.assertNotEqual(treehash.tree_sha256(self.exp), before)

    def test_clean_tracked_tree_reports_git_head(self):
        self._git("init", "-q")
        self._git("add", "experiments/gate/runner.py", "experiments/README")
        self._git("commit", "-qm", "c")
        h = server.harness_commit(self.exp)
        self.assertEqual((h["commit"], h["source"], h["dirty"]), (self._git("rev-parse", "HEAD"), "git", False))

    def test_rsynced_tree_reports_stamp_commit_not_src_head(self):
        # bs2: experiments/ rsynced untracked onto a checkout of the SGLang base commit.
        self._git("init", "-q")
        self._git("commit", "-q", "--allow-empty", "-m", "sglang base")
        self._stamp("deadbeef", treehash.tree_sha256(self.exp))
        h = server.harness_commit(self.exp)
        self.assertEqual((h["commit"], h["source"]), ("deadbeef", "deploy_stamp"))
        self.assertEqual(h["src_head"], self._git("rev-parse", "HEAD"))

    def test_edited_after_deploy_has_no_commit(self):
        self._stamp("deadbeef", treehash.tree_sha256(self.exp))
        with open(os.path.join(self.exp, "gate", "runner.py"), "a") as f:
            f.write("y = 2\n")
        h = server.harness_commit(self.exp)
        self.assertIsNone(h["commit"])
        self.assertEqual(runner._ledger_harness(h), f"tree-sha256:{treehash.tree_sha256(self.exp)}")


_SPEC = ["--speculative-algorithm", "NEXTN"]


def _tau_rows(taus, n_tokens=256):
    return [{"id": f"p{i}", "completion_tokens": n_tokens, "verify_ct": round(n_tokens / t)} for i, t in enumerate(taus)]


class SpecRuleTest(absltest.TestCase):
    _NOISE = {m: 0.002 for m in stats.GATED_METRICS}
    _A = {"control": 3, "candidate": 3}

    def test_rule_applies_to_numerics_changes_under_speculation(self):
        plain, spec = {"server_args": []}, {"server_args": _SPEC}
        self.assertEqual(stats.timing_rule(plain, plain), stats.PAIRED_RULE)
        self.assertEqual(stats.timing_rule(spec, {**spec, "numerics_unchanged": True}), stats.PAIRED_RULE)
        self.assertEqual(stats.timing_rule(spec, spec), stats.SPEC_RULE)
        self.assertEqual(stats.timing_rule(plain, spec), stats.SPEC_RULE)

    def test_consistent_tau_shift_is_significant_and_one_prompt_is_not(self):
        base = [3.0, 3.5, 2.8, 4.0, 3.2, 3.6]
        shifted = stats.tau_ratio(_tau_rows(base), _tau_rows([t * 1.1 for t in base]))
        self.assertTrue(shifted["significant"])
        self.assertAlmostEqual(shifted["ratio"], 1.1, delta=0.01)
        # T3d's shape: one category swings a lot, the rest stay put.
        lottery = stats.tau_ratio(_tau_rows(base), _tau_rows([base[0] * 2.4] + base[1:]))
        self.assertGreater(lottery["ratio"], 1.1)
        self.assertFalse(lottery["significant"])

    def test_tau_ratio_refuses_unpaired_prompts(self):
        with self.assertRaises(ValueError):
            stats.tau_ratio(_tau_rows([3, 3]), _tau_rows([3, 3, 3]))

    def test_counted_tau_needs_significance_and_quality_for_a_gain(self):
        gain = {"significant": True, "ratio": 1.1}
        loss = {"significant": True, "ratio": 0.9}
        self.assertEqual(stats.counted_tau_ratio({"significant": False, "ratio": 1.3}, True)["ratio"], 1.0)
        self.assertEqual(stats.counted_tau_ratio(gain, None)["ratio"], 1.0)
        self.assertEqual(stats.counted_tau_ratio(gain, True)["ratio"], 1.1)
        self.assertEqual(stats.counted_tau_ratio(loss, None)["ratio"], 0.9)

    def _replay(self, decode_ratio, jitter=0.0):
        """Replay summary: candidate decode time = control / decode_ratio, per-pair jitter on W1."""
        legs_c = [_leg(1, 1.0 * decode_ratio, w1_decode=2.55 * decode_ratio * (1 + j), w32_wall=4.0 * decode_ratio)
                  for j in (jitter, -jitter, jitter, -jitter)]
        return stats.summarize_pairs(legs_c, [_leg(1, 1.0)] * 4)

    def test_round_time_must_clear_the_bar_whatever_tau_did(self):
        # T3d: no round-time change, tau up 10% on every prompt but quality unknown.
        tau = stats.tau_ratio(_tau_rows([3, 3.5, 4]), _tau_rows([3.3, 3.85, 4.4]))
        v = stats.spec_verdict(self._replay(1.0), tau, None, self._NOISE, self._A, "w1_tpot_gain")
        self.assertFalse(v["checks"]["w1_tpot_round_time_clears_bar"])
        self.assertFalse(v["promote"])
        # Even a significant tau gain with passing quality cannot carry a flat round time.
        v = stats.spec_verdict(self._replay(1.0), tau, True, self._NOISE, self._A, "w1_tpot_gain")
        self.assertGreater(v["decomposed"]["w1_tpot_gain"], 1.05)
        self.assertFalse(v["promote"])

    def test_round_time_win_promotes_with_insignificant_tau(self):
        tau = stats.tau_ratio(_tau_rows([3, 3.5, 4]), _tau_rows([2.0, 3.5, 4]))
        v = stats.spec_verdict(self._replay(1.04, jitter=0.002), tau, None, self._NOISE, self._A, "w1_tpot_gain")
        self.assertFalse(tau["significant"])
        self.assertAlmostEqual(v["decomposed"]["w1_tpot_gain"], v["round_time"]["w1_tpot_gain"])
        self.assertTrue(v["promote"])

    def test_significant_tau_loss_can_block_a_round_time_win(self):
        tau = stats.tau_ratio(_tau_rows([3, 3.5, 4, 3.2]), _tau_rows([2.7, 3.15, 3.6, 2.88]))
        v = stats.spec_verdict(self._replay(1.04, jitter=0.002), tau, True, self._NOISE, self._A, "w1_tpot_gain")
        self.assertTrue(v["checks"]["w1_tpot_round_time_clears_bar"])
        self.assertFalse(v["checks"]["w1_tpot_decomposed_clears_bar"])
        self.assertFalse(v["promote"])

    def test_noisy_round_time_does_not_promote(self):
        tau = stats.tau_ratio(_tau_rows([3, 3.5]), _tau_rows([3, 3.5]))
        v = stats.spec_verdict(self._replay(1.02, jitter=0.05), tau, None, self._NOISE, self._A, "w1_tpot_gain")
        self.assertFalse(v["checks"]["w1_tpot_round_time_ci_above_1"])
        self.assertFalse(v["promote"])

    def test_mode_change_scales_replay_by_tokens_per_round(self):
        # Plain decode vs MTP replayed at 3 tokens per round, hidden tau 3.3: per-token gain = replay x 3.3 / 3.
        rows_c = [{"id": f"p{i}", "completion_tokens": 256, "verify_ct": 256} for i in range(4)]
        tau = stats.tau_ratio(rows_c, _tau_rows([3.2, 3.4, 3.3, 3.3]))
        a = {"control": 1, "candidate": 3}
        replay = self._replay(1.5)
        v = stats.spec_verdict(replay, tau, True, self._NOISE, a, "w8_composite")
        self.assertTrue(v["mode_change"])
        self.assertAlmostEqual(v["decomposed"]["w8_decode_gain"], replay["overall"]["w8_decode_gain"] * tau["candidate"] / 3)
        self.assertTrue(v["promote"])
        self.assertFalse(stats.spec_verdict(replay, tau, None, self._NOISE, a, "w8_composite")["promote"])

    def test_speculative_arms_must_replay_one_accept_length(self):
        tau = stats.tau_ratio(_tau_rows([3]), _tau_rows([3]))
        with self.assertRaises(ValueError):
            stats.spec_verdict(self._replay(1.0), tau, None, self._NOISE, {"control": 3, "candidate": 4})


class ReplayTest(absltest.TestCase):
    def test_replay_key_matches_the_servers_prompt_key(self):
        # The server's table (sglang.srt.speculative.spec_replay) files prompts under this hash.
        with open(specrule.OVERLAY) as f:
            patch = f.read()
        self.assertIn('+    return hashlib.sha256(",".join(map(str, input_ids)).encode()).hexdigest()', patch)
        self.assertEqual(specrule.replay_key([2, 10, 7]), __import__("hashlib").sha256(b"2,10,7").hexdigest())

    def test_replay_text_mismatch_is_flagged_per_stream(self):
        by_wl = {"W1": [[[2, 5]], [[2, 6]]]}
        replay = {"continuations": [{"input_ids": [2, 5], "output_ids": [9, 9]},
                                    {"input_ids": [2, 6], "output_ids": [8, 8]}]}
        leg = {"workloads": {"W1": [{"streams": [{"output_ids": [9, 9]}]}, {"streams": [{"output_ids": [8, 7]}]}]}}
        res = specrule.replay_text_exact(leg, replay, by_wl)
        self.assertEqual((res["ok"], res["mismatched"]), (False, ["W1/1/0"]))

    def test_replay_legs_time_every_workload_on_fixed_prompts(self):
        mode = specrule.replay_mode("x")
        for wl in config.WORKLOADS:
            self.assertEqual(runner.timing_seed(wl, "pair0", 0, mode), runner.timing_seed(wl, "pair1", 0, mode))
        drawn = [wl for wl in config.WORKLOADS if not wl.fixed_prompt_seed]
        self.assertTrue(drawn)
        for wl in drawn:
            self.assertNotEqual(runner.timing_seed(wl, "pair0", 0), runner.timing_seed(wl, "pair1", 0))


class OverlayTreeTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.src = os.path.join(self.root, "src")
        os.makedirs(self.src)
        self._git("init", "-q")
        with open(os.path.join(self.src, "a.py"), "w") as f:
            f.write("x = 1\n")
        self._git("add", "a.py")
        self._git("commit", "-qm", "c")
        self.sha = self._git("rev-parse", "HEAD")
        with open(os.path.join(self.src, "a.py"), "w") as f:
            f.write("x = 2\n")
        with open(os.path.join(self.src, "b.py"), "w") as f:
            f.write("y = 1\n")
        self._git("add", "-N", "b.py")
        self.patch = os.path.join(self.root, "hook.patch")
        with open(self.patch, "w") as f:
            f.write(self._git("diff") + "\n")
        self._git("checkout", "--", "a.py")
        os.remove(os.path.join(self.src, "b.py"))
        self._git("reset", "-q")
        old = (config.SRC_REPO, config.TREES_DIR)
        config.SRC_REPO, config.TREES_DIR = self.src, os.path.join(self.root, "trees")
        self.addCleanup(lambda: (setattr(config, "SRC_REPO", old[0]), setattr(config, "TREES_DIR", old[1])))

    def _git(self, *args, cwd=None):
        return subprocess.run(["git", "-C", cwd or self.src, "-c", "user.name=t", "-c", "user.email=t@t", *args],
                              check=True, capture_output=True, text=True).stdout.strip()

    def _tree_for(self, overlay=None):
        # tree_for's defaults bind config.SRC_REPO at import; route its git calls to the temp repo.
        old = server._git.__kwdefaults__
        server._git.__kwdefaults__ = {"cwd": self.src}
        try:
            return server.tree_for(self.sha, overlay)
        finally:
            server._git.__kwdefaults__ = old

    def test_overlay_tree_is_commit_plus_patch_and_separate_from_plain(self):
        plain, hooked = self._tree_for(), self._tree_for(self.patch)
        self.assertNotEqual(plain, hooked)
        self.assertEqual(open(os.path.join(plain, "a.py")).read(), "x = 1\n")
        self.assertEqual(open(os.path.join(hooked, "a.py")).read(), "x = 2\n")
        self.assertTrue(os.path.exists(os.path.join(hooked, "b.py")))
        self.assertEqual(self._tree_for(self.patch), hooked)

    def test_edited_overlay_tree_is_refused(self):
        hooked = self._tree_for(self.patch)
        with open(os.path.join(hooked, "a.py"), "w") as f:
            f.write("x = 3\n")
        with self.assertRaises(RuntimeError):
            self._tree_for(self.patch)
        self._git("add", "a.py", cwd=hooked)
        with self.assertRaises(RuntimeError):
            self._tree_for(self.patch)


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

    def test_fp8_o_proj_counts_one_byte_per_weight_plus_row_scales(self):
        tensors = {
            "model.layers.0.self_attn.o_proj.weight": {"dtype": "BF16", "shape": [2816, 4096], "bytes": 2816 * 4096 * 2},
            "model.layers.0.self_attn.qkv_proj.weight": {"dtype": "BF16", "shape": [8, 4], "bytes": 64},
        }
        self.assertIs(sol.served_tensors(tensors, fp8_o_proj=False), tensors)
        served = sol.served_tensors(tensors, fp8_o_proj=True)
        self.assertEqual(served["model.layers.0.self_attn.o_proj.weight"]["bytes"], 2816 * 4096 + 4 * 2816)
        self.assertEqual(served["model.layers.0.self_attn.qkv_proj.weight"], tensors["model.layers.0.self_attn.qkv_proj.weight"])

    def test_fp8_o_proj_follows_the_ref_env(self):
        from gate import solrun

        self.assertFalse(solrun.fp8_o_proj_served("base3"))
        self.assertTrue(solrun.fp8_o_proj_served("base4"))
        self.assertTrue(solrun.fp8_o_proj_served("base5-spec"))


class GpuParseTest(absltest.TestCase):
    """bs3 2026-10-03: a GPU needing a reset lists '[N/A]' compute rows and N/A counters."""

    def setUp(self):
        from gate import gpu

        self._gpu, self._orig = gpu, gpu._smi

    def tearDown(self):
        self._gpu._smi = self._orig

    def test_unresolvable_pid_is_a_foreign_process(self):
        self._gpu._smi = lambda args: "[N/A], [N/A], [N/A]\n"
        self.assertEqual(self._gpu.foreign_processes(None), [{"pid": None, "name": "[N/A]", "used_mib": "[N/A]"}])
        self.assertEqual(hostwatch.foreign_gpu_gb(), 0.0)

    def test_unreadable_counters_raise_gpu_unhealthy(self):
        self._gpu._smi = lambda args: "2026/10/03 14:57:12.000, 49, 0, 0, [N/A], [N/A], 15, [N/A]\n"
        with self.assertRaises(self._gpu.GpuUnhealthy):
            self._gpu.gpu_state()


if __name__ == "__main__":
    absltest.main()
