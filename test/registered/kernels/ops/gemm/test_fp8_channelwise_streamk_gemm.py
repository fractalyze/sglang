import sys

import pytest
import torch

from sglang.kernels.ops.gemm.fp8_channelwise_gemm import (
    MIN_STREAMK_M,
    Sm120ChannelwiseTile,
    fp8_channelwise_streamk_scaled_mm_sm120,
    select_sm120_streamk_tile,
)
from sglang.srt.utils import is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=240,
    stage="base-b",
    runner_config="1-gpu-small",
)

_TILES = (
    Sm120ChannelwiseTile(128, 128, 128),
    Sm120ChannelwiseTile(256, 128, 64),
    Sm120ChannelwiseTile(128, 256, 64),
)


def _operands(M, N, K, device="cuda"):
    fp8 = torch.float8_e4m3fn
    a = (torch.randn(M, K, device=device) * 2).clamp(-448, 448).to(fp8)
    b = (torch.randn(N, K, device=device) * 2).clamp(-448, 448).to(fp8).t()
    scale_a = torch.rand(M, 1, device=device) * 0.01 + 0.001
    scale_b = torch.rand(1, N, device=device) * 0.01 + 0.001
    return a, b, scale_a, scale_b


@pytest.mark.parametrize(
    "m, n, k, expected",
    [
        (4096, 4096, 4096, Sm120ChannelwiseTile(128, 128, 128)),
        (4096, 12288, 4096, Sm120ChannelwiseTile(256, 128, 64)),
        (4096, 4096, 12288, Sm120ChannelwiseTile(128, 256, 64)),
        # 32 x 85 = 2,720 tiles are exactly 16 full waves on 170 SMs: no tail to split.
        (4096, 10880, 4096, None),
        (MIN_STREAMK_M - 1, 4096, 4096, None),
    ],
)
def test_select_sm120_streamk_tile(m, n, k, expected):
    assert select_sm120_streamk_tile(m=m, n=n, k=k, num_sms=170) == expected


@pytest.mark.skipif(not is_sm120_supported(), reason="requires SM120")
@pytest.mark.parametrize("tile", _TILES)
@pytest.mark.parametrize("m", [1024, 1500, 4104])
@pytest.mark.parametrize("n, k", [(4096, 4096), (12288, 4096), (4096, 12288)])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_matches_reference_and_is_deterministic(tile, m, n, k, out_dtype):
    torch.manual_seed(0)
    a, b, scale_a, scale_b = _operands(m, n, k)
    expected = ((a.float() @ b.float()) * scale_a * scale_b).to(out_dtype)
    out = fp8_channelwise_streamk_scaled_mm_sm120(
        a, b, scale_a, scale_b, out_dtype=out_dtype, tile=tile
    )
    again = fp8_channelwise_streamk_scaled_mm_sm120(
        a, b, scale_a, scale_b, out_dtype=out_dtype, tile=tile
    )
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=1e-2)
    assert torch.equal(out, again)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
