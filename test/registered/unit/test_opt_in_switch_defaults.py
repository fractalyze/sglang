"""Defaults of the opt-in serving switches from the Gemma-4 RTX 5090 work.

Each switch below selects a code path that the tree does not take when the
switch is off, so with none of them set the tree serves on upstream's path.
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
}


class TestOptInSwitchDefaults(CustomTestCase):
    def test_every_switch_is_off_unless_set(self):
        for name, off in _OFF_BY_DEFAULT.items():
            with self.subTest(name=name):
                field = getattr(envs, name)
                self.assertEqual(field.default, off)
                if not field.is_set():
                    self.assertEqual(field.get(), off)


if __name__ == "__main__":
    unittest.main()
