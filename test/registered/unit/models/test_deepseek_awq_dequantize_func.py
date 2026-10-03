"""On CUDA, `awq_dequantize_func` must resolve the AWQ dequant kernel without
an ImportError; otherwise AWQ DeepSeek checkpoints fail weight post-processing.
"""

import unittest
from unittest import mock

from sglang.srt.models.deepseek_common import utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAwqDequantizeFunc(CustomTestCase):
    def test_cuda_branch_resolves_kernel(self):
        with mock.patch.object(utils, "_is_cuda", True):
            dequantize = utils.awq_dequantize_func()
        self.assertTrue(callable(dequantize))


if __name__ == "__main__":
    unittest.main()
