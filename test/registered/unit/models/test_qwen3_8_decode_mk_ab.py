# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""benchmark/qwen38_decode_mk/ab.py's arms and timing: every arm's command
parses as a server, ours pass the SGLANG_DECODE_MK_* switches' checks under
their own switches and serve this checkout, and the stock arms serve the
installed SGLang with every switch off. No GPU, no server."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import argparse
import importlib.util
import os
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

from sglang.srt.arg_groups.decode_mk_hook import decode_mk_errors
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase

_REPO = Path(__file__).resolve().parents[4]


def _load_ab():
    spec = importlib.util.spec_from_file_location(
        "qwen38_decode_mk_ab", _REPO / "benchmark/qwen38_decode_mk/ab.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Its dataclasses look their module up by name.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ab = _load_ab()


def _parsed(arm) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    command = ab.launch_command(arm, "/snapshot")
    self_module = command.index("sglang.launch_server")
    return parser.parse_args(command[self_module + 1 :])


def _switches(env: dict) -> ExitStack:
    stack = ExitStack()
    for name in ("DECODE", "PREFILL", "MTP"):
        var = getattr(envs, f"SGLANG_DECODE_MK_{name}")
        stack.enter_context(var.override(env.get(f"SGLANG_DECODE_MK_{name}") == "1"))
    return stack


_OURS = [name for name in ab.ARMS if name.startswith("ours")]
_STOCK = [name for name in ab.ARMS if not name.startswith("ours")]


class TestArms(CustomTestCase):
    def test_every_arm_parses(self):
        for name, arm in ab.ARMS.items():
            with self.subTest(arm=name):
                args = _parsed(arm)
                self.assertEqual(args.model_path, "/snapshot")
                self.assertEqual(args.port, ab.PORT)

    def test_ours_pass_the_switches_checks(self):
        for name in _OURS:
            arm = ab.ARMS[name]
            with self.subTest(arm=name), _switches(arm.env):
                self.assertEqual(decode_mk_errors(_parsed(arm)), [])

    def test_ours_draft_as_named(self):
        for k in (1, 2, 3):
            args = _parsed(ab.ARMS[f"ours-mtp{k}"])
            self.assertEqual(args.speculative_algorithm, "DECODE_MK_MTP")
            self.assertEqual(args.speculative_num_steps, k)
        self.assertIsNone(_parsed(ab.ARMS["ours"]).speculative_algorithm)

    def test_every_arm_serves_one_request(self):
        for name, arm in ab.ARMS.items():
            self.assertEqual(_parsed(arm).max_running_requests, 1, name)


class TestLaunchEnv(CustomTestCase):
    def test_ours_serve_this_checkout_under_their_switches(self):
        for name in _OURS:
            env = ab.launch_env(ab.ARMS[name])
            self.assertEqual(env["PYTHONPATH"], str(_REPO / "python"))
            self.assertEqual(env["SGLANG_DECODE_MK_DECODE"], "1")
            self.assertEqual(env["SGLANG_DECODE_MK_PREFILL"], "1")
            self.assertEqual(
                env.get("SGLANG_DECODE_MK_MTP"), "1" if "mtp" in name else None
            )

    def test_stock_arms_drop_inherited_switches_and_path(self):
        inherited = {"SGLANG_DECODE_MK_DECODE": "1", "PYTHONPATH": "/elsewhere"}
        with mock.patch.dict(os.environ, inherited):
            for name in _STOCK:
                env = ab.launch_env(ab.ARMS[name])
                self.assertNotIn("PYTHONPATH", env)
                self.assertFalse(
                    any(k.startswith("SGLANG_DECODE_MK_") for k in env), name
                )


class TestTiming(CustomTestCase):
    def test_decode_rate_counts_tokens_after_the_first(self):
        self.assertAlmostEqual(ab.decode_rate(11, 1.0, 1.5), 20.0)
        self.assertIsNone(ab.decode_rate(1, 1.0, 1.5))
        self.assertIsNone(ab.decode_rate(11, 1.0, 1.0))

    def test_summarize_skips_missing_rates(self):
        rows = [
            {"ttft": 0.3, "decode_tok_s": 100.0},
            {"ttft": 0.1, "decode_tok_s": None},
            {"ttft": 0.2, "decode_tok_s": 120.0},
        ]
        summary = ab.summarize(rows)
        self.assertEqual(summary["ttft"], {"median": 0.2, "min": 0.1, "max": 0.3})
        self.assertEqual(
            summary["decode_tok_s"], {"median": 110.0, "min": 100.0, "max": 120.0}
        )


if __name__ == "__main__":
    unittest.main()
