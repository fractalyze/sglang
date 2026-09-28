# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image 2.1 VAE decoder fast paths against their eager references."""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage21 import (
    QwenImage21CausalConv3d,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)


@pytest.mark.parametrize("channels,size,kernel", [(96, 256, 3), (192, 128, 3), (384, 64, 1)])
def test_causal_conv_implicit_padding_matches_padded_copy(channels, size, kernel):
    torch.manual_seed(0)
    conv = QwenImage21CausalConv3d(channels, channels, kernel, padding=kernel // 2)
    conv = conv.cuda().bfloat16()
    x = torch.randn(1, channels, 1, size, size, device="cuda", dtype=torch.bfloat16)
    padded = F.pad(x.squeeze(2), list(conv._padding))
    reference = F.conv2d(padded, conv.weight, conv.bias, conv.stride).unsqueeze(2)
    # Same math; cuDNN may pick another algorithm for the unpadded shape.
    torch.testing.assert_close(conv(x), reference, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
