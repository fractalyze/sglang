"""Defaults of the opt-in serving switches from the Gemma-4 RTX 5090 work.

Each switch below selects a code path that the tree does not take when the
switch is off, so with none of them set the tree serves on upstream's path.
The one deliberate exception is upstream's tuned channelwise FP8 GEMM route,
which is on by default and for which this work adds RTX 5090 config files.
"""

import unittest

from sglang.srt.environ import Gemma4FusedGlue, envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# Switch -> its value when the environment does not set it.
_OFF_BY_DEFAULT = {
    "SGLANG_OPT_GEMMA4_FUSED_GLUE": Gemma4FusedGlue.OFF,
    "SGLANG_OPT_GEMMA4_FP8_VOCAB_TABLE": False,
    "SGLANG_OPT_TRITON_EXTEND_SM120_FP8_KV_TILES": False,
    "SGLANG_OPT_HICACHE_PIN_LOAD_BACK_WINDOW": False,
    "SGLANG_OPT_HICACHE_FENCE_WRITE_THROUGH": False,
    "SGLANG_OPT_SWA_DECODE_NO_HOST_SYNC": False,
    "SGLANG_OPT_SWA_PREFILL_WINDOW_MARGIN": 0,
}


class TestOptInSwitchDefaults(CustomTestCase):
    def test_every_switch_is_off_unless_set(self):
        for name, off in _OFF_BY_DEFAULT.items():
            with self.subTest(name=name):
                field = getattr(envs, name)
                self.assertEqual(field.default, off)
                if not field.is_set():
                    self.assertEqual(field.get(), off)

    def test_tuned_fp8_gemm_route_is_on_unless_killed(self):
        # The RTX 5090 dense FP8 tiles apply through this route by default;
        # SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE=0 is their kill switch.
        self.assertIs(envs.SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE.default, True)
        with envs.SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE.override(False):
            self.assertIs(envs.SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE.get(), False)


if __name__ == "__main__":
    unittest.main()
