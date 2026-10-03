import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.jit.utils import KERNEL_PATH, load_jit
from sglang.kernels.ops.moe.w4a16_moe_sm90 import (
    TILE_K,
    TILE_N,
    dequantize_reference,
    pack_codes,
    repack_awq_moe_weights,
)
from sglang.kernels.ops.moe.w4fp8_moe_sm90 import (
    K_PERMUTE16,
    TOKEN_BLOCK,
    quantize_activations,
    w4fp8_moe_sm90_gemm,
)
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_GROUP = 128


def _is_sm90() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


def _has_fp8_tensor_cores() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


def _random_awq(
    num_experts: int, k: int, n: int, device: str, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random AWQ-format (qweight, scales, qzeros) with realistic magnitudes."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    int32 = dict(dtype=torch.int32, generator=gen)
    qweight = torch.randint(-(2**31), 2**31 - 1, (num_experts, k, n // 8), **int32)
    qzeros = torch.randint(
        -(2**31), 2**31 - 1, (num_experts, k // _GROUP, n // 8), **int32
    )
    scales = torch.rand((num_experts, k // _GROUP, n), generator=gen) * 0.02 + 0.002
    return qweight.to(device), scales.to(torch.bfloat16).to(device), qzeros.to(device)


def _permuted_columns(k: int, device) -> torch.Tensor:
    """Column index of the source activation stored at each permuted column."""
    base = torch.arange(0, k, 16, device=device).repeat_interleave(16)
    return base + torch.tensor(K_PERMUTE16, device=device).repeat(k // 16)


def _dequantize_activations(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """quantize_activations output -> float [R, K] in the original column order."""
    k = q.shape[1]
    values = q.float() * scales.repeat_interleave(_GROUP, dim=1)
    out = torch.empty_like(values)
    out[:, _permuted_columns(k, q.device)] = values
    return out


def test_k_permute16_maps_fp8_fragment_columns_onto_bf16_ones():
    # Thread t of a warp holds bf16 m16k16 A columns {2t, 2t + 1, 2t + 8, 2t + 9}
    # and e4m3 m16k32 A columns {4t, ..., 4t + 3} of each 16-column half.
    for t in range(4):
        fp8_columns = range(4 * t, 4 * t + 4)
        bf16_columns = [2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9]
        assert [K_PERMUTE16[f] for f in fp8_columns] == bf16_columns
    assert sorted(K_PERMUTE16) == list(range(16))


@pytest.mark.skipif(not _has_fp8_tensor_cores(), reason="needs an sm89+ GPU")
def test_quantize_activations_matches_reference():
    gen = torch.Generator(device="cpu").manual_seed(2)
    a = (torch.randn(37, 512, generator=gen) * 3).to(torch.bfloat16).cuda()
    a[5, 128:256] = 0  # an all-zero group keeps a finite scale
    q, scales = quantize_activations(a)

    assert q.dtype == torch.float8_e4m3fn and scales.shape == (37, 4)
    groups = a.float().view(37, 4, _GROUP)
    expected_scales = groups.abs().amax(-1).clamp(min=1e-10) / 448.0
    torch.testing.assert_close(scales, expected_scales)
    expected_q = (groups / expected_scales.unsqueeze(-1)).view(37, 512)
    expected_q = expected_q[:, _permuted_columns(512, a.device)]
    torch.testing.assert_close(q.float(), expected_q.to(torch.float8_e4m3fn).float())
    torch.testing.assert_close(
        _dequantize_activations(q, scales), a.float(), rtol=7e-2, atol=1e-2
    )


@pytest.mark.skipif(not _has_fp8_tensor_cores(), reason="needs an sm89+ GPU")
def test_fragment_layout_through_fp8_mma():
    """w4a16 slice words, read as e4m3 fragments, multiply permuted activations exactly."""
    module = load_jit(
        "test_w4fp8_moe_fragment",
        cuda_files=[str(pathlib.Path(__file__).with_name("w4fp8_moe_fragment.cuh"))],
        cuda_wrappers=[("fragment_mma", "fragment_mma")],
        extra_include_paths=[str(KERNEL_PATH / "csrc")],
    )
    gen = torch.Generator(device="cpu").manual_seed(1)
    codes = torch.randint(0, 16, (1, TILE_N, TILE_K), dtype=torch.uint8, generator=gen)
    zeros = torch.randint(0, 16, (TILE_N,), dtype=torch.uint8, generator=gen)
    x = torch.randn(8, TILE_K, generator=gen).to(torch.float8_e4m3fn)
    x_permuted = x[:, _permuted_columns(TILE_K, x.device)].contiguous()
    y = torch.empty(TILE_N, 8, dtype=torch.float32, device="cuda")
    module.fragment_mma(
        pack_codes(codes).view(-1).cuda(), zeros.cuda(), x_permuted.cuda(), y
    )
    # Ground truth from the unpacked codes and unpermuted x, so neither the
    # packing nor the permutation can cancel out.
    q_minus_z = codes[0].float() - zeros.float()[:, None]
    expected = q_minus_z @ x.float().t()
    torch.testing.assert_close(y.cpu(), expected, rtol=1e-5, atol=1e-4)


def _routing(num_tokens: int, top_k: int, num_experts: int, seed: int):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    scores = torch.rand(num_tokens, num_experts, generator=gen)
    topk_weights, topk_ids = scores.topk(top_k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
    return topk_ids.to(torch.int32).cuda(), topk_weights.float().cuda()


def _routed_reference(a, dense, topk_ids, topk_weights, a_row_divisor):
    """out[r] = dense[topk_ids.flat[r]] @ a[r / divisor], optionally weighted."""
    flat_ids = topk_ids.view(-1).long()
    rows = a[torch.arange(flat_ids.numel(), device=a.device) // a_row_divisor].float()
    out = torch.einsum("rk,rnk->rn", rows, dense[flat_ids].float())
    if topk_weights is not None:
        out = out * topk_weights.view(-1, 1)
    return out


# (K, N): DeepSeek-V3 gate-up and down projections at TP8, plus a small shape.
_SHAPES = [(7168, 512), (256, 7168), (256, 256)]
# Tokens per expert across the prefill band, plus partial and padding-heavy blocks.
_TOKENS_PER_EXPERT = [8, 100, 128, 256]


@pytest.mark.skipif(not _is_sm90(), reason="needs an SM90 (Hopper) GPU")
@pytest.mark.parametrize("k,n", _SHAPES)
@pytest.mark.parametrize("tokens_per_expert", _TOKENS_PER_EXPERT)
@pytest.mark.parametrize("weighted", [False, True])
def test_routed_gemm_matches_reference(k, n, tokens_per_expert, weighted):
    num_experts, top_k = 16, 4
    num_tokens = tokens_per_expert * num_experts // top_k
    weights = repack_awq_moe_weights(*_random_awq(num_experts, k, n, "cuda"), _GROUP)
    topk_ids, topk_weights = _routing(num_tokens, top_k, num_experts, seed=k + n)
    a_row_divisor = top_k if n != 7168 else 1
    a_rows = num_tokens if a_row_divisor == top_k else num_tokens * top_k
    a = (torch.randn(a_rows, k, device="cuda") * 0.5).to(torch.bfloat16)
    a_q, a_scales = quantize_activations(a)

    sorted_ids, expert_ids, num_post_padded = moe_align_block_size(
        topk_ids, TOKEN_BLOCK, num_experts
    )
    out = torch.full((num_tokens * top_k, n), float("nan"), device="cuda").to(
        torch.bfloat16
    )
    w4fp8_moe_sm90_gemm(
        a=a_q,
        a_scales=a_scales,
        out=out,
        weights=weights,
        sorted_token_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_post_padded,
        topk_weights=topk_weights.view(-1) if weighted else None,
        a_row_divisor=a_row_divisor,
    )
    # FP32 scales keep (q - z) * s unrounded, as the kernel applies s to an FP32 partial.
    dense = dequantize_reference(weights._replace(scales=weights.scales.float()))
    weighting = topk_weights if weighted else None
    # Against the exact product of the quantised operands: only accumulation order differs.
    expected = _routed_reference(
        _dequantize_activations(a_q, a_scales),
        dense,
        topk_ids,
        weighting,
        a_row_divisor,
    )
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)
    # Against bf16 activations: bounds the e4m3 activation rounding.
    bf16_expected = _routed_reference(a, dense, topk_ids, weighting, a_row_divisor)
    rel_err = (out.float() - bf16_expected).norm() / bf16_expected.norm()
    assert rel_err < 5e-2, rel_err


def _awq_layer(num_experts, hidden, intermediate, device):
    layer = torch.nn.Module()
    for name, (k, n, seed) in (
        ("w13", (hidden, 2 * intermediate, 10)),
        ("w2", (intermediate, hidden, 11)),
    ):
        tensors = _random_awq(num_experts, k, n, device, seed=seed)
        for suffix, t in zip(("qweight", "scales", "qzeros"), tensors):
            layer.register_parameter(
                f"{name}_{suffix}", torch.nn.Parameter(t, requires_grad=False)
            )
    layer.intermediate_size_per_partition = intermediate
    return layer


def _run_awq_moe(backend, layer, hidden_states, topk_ids, topk_weights):
    """Runs the AWQ MoE layer as served: scheme repack, runner, fused func."""
    from sglang.srt.hardware_backend.gpu.quantization.awq_kernels import AWQMoEKernel
    from sglang.srt.layers.moe.moe_runner import MoeRunner, MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    quant_config = SimpleNamespace(pack_factor=8, weight_bits=4, group_size=_GROUP)
    kernel = AWQMoEKernel(quant_config)
    kernel.runner = MoeRunner(
        backend, MoeRunnerConfig(activation="silu", is_gated=True)
    )
    kernel.process_weights_after_loading(layer)
    # Marlin only checks the logits' token count; routing comes from topk_ids.
    router_logits = topk_weights.new_zeros(
        topk_ids.shape[0], layer.w13_qweight.shape[0]
    )
    topk_output = StandardTopKOutput(
        topk_weights=topk_weights, topk_ids=topk_ids, router_logits=router_logits
    )
    dispatch_output = StandardDispatchOutput(
        hidden_states=hidden_states, hidden_states_scale=None, topk_output=topk_output
    )
    return kernel.apply(layer, dispatch_output).hidden_states


@pytest.mark.skipif(not _is_sm90(), reason="needs an SM90 (Hopper) GPU")
def test_w4a16_sm90_backend_runs_large_m_on_fp8_close_to_marlin():
    from sglang.srt.layers.moe.moe_runner.w4a16_sm90 import (
        W4FP8_MIN_TOKENS_PER_EXPERT,
    )
    from sglang.srt.layers.moe.utils import MoeRunnerBackend

    num_experts, top_k, hidden, intermediate = 32, 8, 1024, 256
    num_tokens = 2 * W4FP8_MIN_TOKENS_PER_EXPERT * num_experts // top_k
    topk_ids, topk_weights = _routing(num_tokens, top_k, num_experts, seed=3)
    hidden_states = (torch.randn(num_tokens, hidden, device="cuda") * 0.5).to(
        torch.bfloat16
    )

    ours = _run_awq_moe(
        MoeRunnerBackend.W4A16_SM90,
        _awq_layer(num_experts, hidden, intermediate, "cuda"),
        hidden_states,
        topk_ids,
        topk_weights,
    )
    marlin = _run_awq_moe(
        MoeRunnerBackend.MARLIN,
        _awq_layer(num_experts, hidden, intermediate, "cuda"),
        hidden_states,
        topk_ids,
        topk_weights,
    )
    rel_err = (ours.float() - marlin.float()).norm() / marlin.float().norm()
    assert rel_err < 5e-2, rel_err


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
