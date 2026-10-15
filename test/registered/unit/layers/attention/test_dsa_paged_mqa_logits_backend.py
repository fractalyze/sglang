"""Contract tests for how `--dsa-paged-mqa-logits-backend` resolves per platform.

`auto` picks the CuTe DSL kernel on SM100: DeepGEMM chunks a decode batch larger
than the SM count and concatenates full-context-width logits under CUDA graph.
"""

import unittest
from unittest.mock import patch

from sglang.srt.layers.attention.dsa.paged_mqa_logits_backend import (
    DSAPagedMQALogitsBackend,
)
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_IS_HIP = "sglang.srt.layers.attention.dsa.paged_mqa_logits_backend.is_hip"


class TestDSAPagedMQALogitsBackendResolve(unittest.TestCase):
    def _resolve(self, value, *, sm100=False, hip=False):
        with patch(_IS_HIP, return_value=hip), override_platform(is_sm100=sm100):
            return DSAPagedMQALogitsBackend.resolve(value)

    def test_auto_picks_cutedsl_on_sm100(self):
        self.assertEqual(
            self._resolve("auto", sm100=True), DSAPagedMQALogitsBackend.CUTEDSL
        )

    def test_auto_picks_deepgemm_off_sm100(self):
        self.assertEqual(
            self._resolve("auto", sm100=False), DSAPagedMQALogitsBackend.DEEPGEMM
        )

    def test_auto_picks_aiter_on_rocm(self):
        self.assertEqual(
            self._resolve("auto", hip=True), DSAPagedMQALogitsBackend.AITER
        )

    def test_explicit_deepgemm_is_kept_on_sm100(self):
        self.assertEqual(
            self._resolve("deepgemm", sm100=True), DSAPagedMQALogitsBackend.DEEPGEMM
        )


if __name__ == "__main__":
    unittest.main()
