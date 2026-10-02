# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""The SGLANG_DECODE_MK_* switches: off by default and inert when off, the
server they refuse, the model they build, and the vendored decode-mk tree.
No GPU, no weights; test/registered/kernels/ops/decode_mk/ runs the kernels
and test/registered/e2e/models/test_qwen3_8_decode_mk.py serves on them."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

import hashlib
import importlib.util
import os
import re
import subprocess
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.arg_groups.decode_mk_hook import decode_mk_errors
from sglang.srt.environ import envs
from sglang.test.test_utils import CustomTestCase

_REPO = Path(__file__).resolve().parents[4]
_VENDORED = _REPO / "python/sglang/kernels/decode_mk"


def _switches(decode=False, prefill=False, mtp=False) -> ExitStack:
    stack = ExitStack()
    stack.enter_context(envs.SGLANG_DECODE_MK_DECODE.override(decode))
    stack.enter_context(envs.SGLANG_DECODE_MK_PREFILL.override(prefill))
    stack.enter_context(envs.SGLANG_DECODE_MK_MTP.override(mtp))
    return stack


def _args(**overrides) -> SimpleNamespace:
    """A server the switches serve; `overrides` change it."""
    fields = dict(
        max_running_requests=1,
        disable_radix_cache=True,
        disable_cuda_graph=True,
        context_length=16384,
        tp_size=1,
        pp_size=1,
        dp_size=1,
        disaggregation_mode="null",
        speculative_algorithm=None,
        disable_overlap_schedule=False,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


_STOCK = _args(
    max_running_requests=None,
    disable_radix_cache=False,
    disable_cuda_graph=False,
    context_length=None,
)


class TestSwitches(CustomTestCase):
    def test_off_by_default(self):
        self.assertFalse(envs.SGLANG_DECODE_MK_DECODE.get())
        self.assertFalse(envs.SGLANG_DECODE_MK_PREFILL.get())
        self.assertFalse(envs.SGLANG_DECODE_MK_MTP.get())

    def test_off_refuses_nothing(self):
        with _switches():
            self.assertEqual(decode_mk_errors(_STOCK), [])

    def test_decode_refuses_a_stock_server(self):
        with _switches(decode=True):
            errors = "\n".join(decode_mk_errors(_STOCK))
        for flag in (
            "--max-running-requests 1",
            "--disable-radix-cache",
            "--disable-cuda-graph",
            "--context-length",
        ):
            self.assertIn(flag, errors)

    def test_decode_serves_one_request(self):
        with _switches(decode=True, prefill=True):
            self.assertEqual(decode_mk_errors(_args()), [])

    def test_decode_refuses_batches(self):
        for running in (2, 48):
            with _switches(decode=True):
                errors = decode_mk_errors(_args(max_running_requests=running))
            self.assertEqual(len(errors), 1)
            self.assertIn("--max-running-requests 1", errors[0])

    def test_decode_refuses_parallelism(self):
        for parallel in ("tp_size", "pp_size", "dp_size"):
            with _switches(decode=True):
                errors = decode_mk_errors(_args(**{parallel: 2}))
            self.assertEqual(len(errors), 1)

    def test_prefill_and_mtp_need_decode(self):
        for kind in ("prefill", "mtp"):
            with _switches(**{kind: True}):
                errors = decode_mk_errors(_args())
            self.assertIn("SGLANG_DECODE_MK_DECODE", errors[0])

    def test_mtp_needs_its_algorithm_without_overlap(self):
        with _switches(decode=True, mtp=True):
            errors = "\n".join(decode_mk_errors(_args()))
            self.assertIn("--speculative-algorithm DECODE_MK_MTP", errors)
            self.assertIn("--disable-overlap-schedule", errors)
            served = _args(
                speculative_algorithm="DECODE_MK_MTP", disable_overlap_schedule=True
            )
            self.assertEqual(decode_mk_errors(served), [])

    def test_decode_refuses_other_speculation(self):
        with _switches(decode=True):
            errors = decode_mk_errors(_args(speculative_algorithm="EAGLE"))
        self.assertIn("SGLANG_DECODE_MK_MTP", errors[0])

    def test_mtp_algorithm_needs_its_switch(self):
        with _switches():
            errors = decode_mk_errors(_args(speculative_algorithm="DECODE_MK_MTP"))
        self.assertIn("SGLANG_DECODE_MK_MTP=1", errors[0])


class TestMtpAlgorithm(CustomTestCase):
    def setUp(self):
        from sglang.srt.speculative.decode_mk_mtp import DecodeMkMtpAlgo

        self.algo = DecodeMkMtpAlgo("DECODE_MK_MTP", factory=lambda args: None)

    def test_defaults_to_one_draft(self):
        args = SimpleNamespace(
            speculative_num_steps=None,
            speculative_eagle_topk=None,
            speculative_num_draft_tokens=None,
        )
        self.algo.handle_server_args(args)
        self.assertEqual(args.speculative_num_steps, 1)
        self.assertEqual(args.speculative_eagle_topk, 1)
        self.assertEqual(args.speculative_num_draft_tokens, 2)

    def test_verifies_up_to_four_tokens(self):
        args = SimpleNamespace(speculative_num_steps=3)
        self.algo.handle_server_args(args)
        self.assertEqual(args.speculative_num_draft_tokens, 4)
        with self.assertRaisesRegex(ValueError, "1 to 3"):
            self.algo.handle_server_args(SimpleNamespace(speculative_num_steps=4))

    def test_answers_the_enum_interface(self):
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        interface = {
            name
            for name in vars(SpeculativeAlgorithm)
            if name.startswith(("is_", "supports_", "carries_", "has_"))
        }
        for name in interface:
            self.assertTrue(callable(getattr(self.algo, name)), name)
        self.assertFalse(self.algo.is_eagle())
        self.assertTrue(self.algo.is_speculative())


class TestMtpRegistration(CustomTestCase):
    """DECODE_MK_MTP registers as spec_info is imported, so each case runs
    in a fresh interpreter."""

    def _resolves(self, switch: str) -> bool:
        code = (
            "from sglang.srt.speculative.spec_info import SpeculativeAlgorithm\n"
            "try:\n"
            "    SpeculativeAlgorithm.from_string('DECODE_MK_MTP')\n"
            "except ValueError:\n"
            "    raise SystemExit(1)\n"
        )
        env = {**os.environ, "SGLANG_DECODE_MK_MTP": switch}
        return subprocess.run([sys.executable, "-c", code], env=env).returncode == 0

    def test_absent_when_off(self):
        self.assertFalse(self._resolves("0"))

    def test_registered_when_on(self):
        self.assertTrue(self._resolves("1"))


class TestArchitecture(CustomTestCase):
    def _resolve(self, architectures):
        from sglang.srt.configs.model_config import ModelImpl
        from sglang.srt.model_loader.utils import get_model_architecture

        config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=architectures),
            quantization=None,
            model_impl=ModelImpl.AUTO,
        )
        return get_model_architecture(config)[0].__name__

    def test_off_builds_the_stock_model(self):
        with _switches():
            self.assertEqual(
                self._resolve(["Qwen3_5ForConditionalGeneration"]),
                "Qwen3_5ForConditionalGeneration",
            )

    def test_decode_builds_the_kernels_model(self):
        with _switches(decode=True):
            self.assertEqual(
                self._resolve(["Qwen3_5ForConditionalGeneration"]),
                "Qwen3_5DecodeMkForConditionalGeneration",
            )

    def test_decode_refuses_other_models(self):
        with _switches(decode=True):
            with self.assertRaisesRegex(ValueError, "Qwen3.8-27B"):
                self._resolve(["Qwen3ForCausalLM"])


def _sync_script():
    spec = importlib.util.spec_from_file_location(
        "sync_decode_mk", _REPO / "scripts/sync_decode_mk.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestVendoredTree(CustomTestCase):
    def test_files_match_their_manifest(self):
        lines = (_VENDORED / "VENDORED").read_text().splitlines()
        self.assertRegex(lines[0], r"^decode-mk [0-9a-f]{40}$")
        listed = {}
        for line in lines[1:]:
            digest, name = line.split("  ")
            listed[name] = digest
        self.assertEqual(sorted(listed), sorted(_sync_script().vendored_files()))
        for name, digest in listed.items():
            actual = hashlib.sha256((_VENDORED / name).read_bytes()).hexdigest()
            self.assertEqual(
                actual, digest, f"{name} differs from decode-mk; resync it instead"
            )

    def test_imports_point_into_sglang(self):
        for path in _VENDORED.glob("*.py"):
            self.assertIsNone(
                re.search(r"^(from|import) s2mk\b", path.read_text(), re.MULTILINE),
                path.name,
            )

    def test_slice_cuts_named_functions_and_bindings(self):
        sync = _sync_script()
        source = (
            '#include "slow_ar.h"\n'
            "\n"
            "// Runs it.\n"
            "void RunSlowAr(int a,\n"
            "               int b) {\n"
            "  Body();\n"
            "}\n"
            "\n"
            "void RunGdn(int a) {\n"
            "  Body();\n"
            "}\n"
            "\n"
            "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"
            '  m.def("run_slow_ar", &s2mk::RunSlowAr, "Runs it.",\n'
            '        py::arg("a"));\n'
            '  m.def("run_gdn", &s2mk::RunGdn, "Runs gdn.");\n'
            "}\n"
        )
        self.assertEqual(
            sync.slice_ops(source),
            '#include "slow_ar.h"\n'
            "\n"
            "void RunGdn(int a) {\n"
            "  Body();\n"
            "}\n"
            "\n"
            "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"
            '  m.def("run_gdn", &s2mk::RunGdn, "Runs gdn.");\n'
            "}\n",
        )

    def test_imports_are_rewritten(self):
        sync = _sync_script()
        self.assertEqual(
            sync.rewrite_imports("from s2mk import _ext\nfrom s2mk.gdn import Gdn\n"),
            "from sglang.kernels.decode_mk import _ext\n"
            "from sglang.kernels.decode_mk.gdn import Gdn\n",
        )


if __name__ == "__main__":
    unittest.main()
