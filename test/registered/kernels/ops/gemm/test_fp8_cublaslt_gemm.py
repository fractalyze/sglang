import sys

import pytest
import torch

from sglang.kernels.ops.gemm.fp8_cublaslt_gemm import (
    MIN_CUBLASLT_M,
    _jit_module,
    _workspace,
    fp8_per_channel_scaled_mm_cublaslt,
    maybe_fp8_per_channel_scaled_mm_cublaslt,
)
from sglang.srt.utils import is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60,
    stage="base-b",
    runner_config="1-gpu-small",
)

pytestmark = pytest.mark.skipif(not is_sm120_supported(), reason="requires SM120")


def _operands(m, n, k):
    fp8 = torch.float8_e4m3fn
    a = (torch.randn(m, k, device="cuda") * 2).clamp(-448, 448).to(fp8)
    w = (torch.randn(n, k, device="cuda") * 2).clamp(-448, 448).to(fp8)
    scale_a = torch.rand(m, 1, device="cuda") * 0.01 + 0.001
    scale_b = torch.rand(1, n, device="cuda") * 0.01 + 0.001
    return a, w, scale_a, scale_b


def _reference(a, w, scale_a, scale_b):
    return (a.float() @ w.float().t()) * scale_a * scale_b


@pytest.mark.parametrize("m", [1024, 1500, 4104])
@pytest.mark.parametrize("n, k", [(4096, 4096), (12288, 4096), (4096, 12288)])
def test_matches_reference_and_is_deterministic(m, n, k):
    torch.manual_seed(0)
    a, w, scale_a, scale_b = _operands(m, n, k)
    expected = _reference(a, w, scale_a, scale_b)
    out = fp8_per_channel_scaled_mm_cublaslt(a, w, scale_a, scale_b)
    again = fp8_per_channel_scaled_mm_cublaslt(a, w, scale_a, scale_b)
    # Two bf16 roundings (unscaled product, then scaled): about 2 bf16 ulps.
    torch.testing.assert_close(out.float(), expected, rtol=1.6e-2, atol=1e-2)
    assert torch.equal(out, again)


def test_every_heuristic_algo_is_correct():
    torch.manual_seed(0)
    m, n, k = 1024, 4096, 4096
    a, w, _, _ = _operands(m, n, k)
    expected = a.float() @ w.float().t()
    module = _jit_module()
    workspace = _workspace(0)
    num_algos = module.num_algos(a, w, workspace)
    assert num_algos >= 1
    for idx in range(num_algos):
        out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        module.gemm(out, a, w, workspace, idx)
        torch.testing.assert_close(out.float(), expected, rtol=8e-3, atol=1.0)


def test_fp32_accumulation_does_not_overflow():
    # 32 products of 448 * 448 overflow an fp16 accumulator in the first MMA step.
    m = n = k = MIN_CUBLASLT_M
    a = torch.ones(m, k, device="cuda")
    a[:, :32] = 448
    a = a.to(torch.float8_e4m3fn)
    w = torch.full((n, k), 448.0, device="cuda").to(torch.float8_e4m3fn)
    ones_m, ones_n = torch.ones(m, 1, device="cuda"), torch.ones(1, n, device="cuda")
    out = fp8_per_channel_scaled_mm_cublaslt(a, w, ones_m, ones_n)
    expected = a.float() @ w.float().t()
    torch.testing.assert_close(out.float(), expected, rtol=8e-3, atol=0.0)


@pytest.mark.parametrize(
    "case",
    ["small_m", "fp16_out", "per_tensor_scale", "non_contiguous_a"],
)
def test_declines_unsupported_calls(case):
    torch.manual_seed(0)
    m = 256 if case == "small_m" else MIN_CUBLASLT_M
    a, w, scale_a, scale_b = _operands(m, 4096, 4096)
    out_dtype = torch.float16 if case == "fp16_out" else torch.bfloat16
    if case == "per_tensor_scale":
        scale_a = scale_a[:1, 0]
    if case == "non_contiguous_a":
        a = a.t().contiguous().t()
    assert (
        maybe_fp8_per_channel_scaled_mm_cublaslt(a, w.t(), scale_a, scale_b, out_dtype)
        is None
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
