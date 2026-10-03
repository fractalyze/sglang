import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.jit.utils import KERNEL_PATH, load_jit
from sglang.kernels.ops.moe.w4a16_moe_sm90 import (
    TILE_K,
    TILE_N,
    W4A16MoeWeights,
    dequantize_reference,
    pack_codes,
    repack_awq_moe_weights,
    select_token_block,
    unpack_awq_codes,
    w4a16_moe_sm90_gemm,
)
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_GROUP = 128


def _is_sm90() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


def _has_tensor_cores() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


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


def _awq_dense(qweight, scales, qzeros) -> torch.Tensor:
    """AWQ checkpoint -> dense [E, N, K], (q - z) * s in bfloat16."""
    codes = unpack_awq_codes(qweight).to(torch.bfloat16)
    zeros = unpack_awq_codes(qzeros).to(torch.bfloat16).repeat_interleave(_GROUP, 1)
    return ((codes - zeros) * scales.repeat_interleave(_GROUP, 1)).transpose(1, 2)


def test_repack_round_trip():
    qweight, scales, qzeros = _random_awq(3, 256, 384, "cpu")
    weights = repack_awq_moe_weights(qweight, scales, qzeros, group_size=_GROUP)
    assert weights.qweight.shape == (
        3,
        384 // TILE_N,
        256 // TILE_K,
        TILE_N * TILE_K // 8,
    )
    assert torch.equal(
        dequantize_reference(weights), _awq_dense(qweight, scales, qzeros)
    )


def test_awq_unpack_order():
    # AutoAWQ stores columns 0, 2, 4, 6, 1, 3, 5, 7 in nibbles 0..7.
    word = sum(col << (4 * slot) for slot, col in enumerate([0, 2, 4, 6, 1, 3, 5, 7]))
    packed = torch.tensor([[word]], dtype=torch.int32)
    assert unpack_awq_codes(packed).tolist() == [list(range(8))]


def test_stage_word_layout_is_the_documented_fragment_map():
    """Pins where every nibble of a stage lands; w4fp8_moe_sm90 reads these words too.

    Word ((wg * 2 + half) * 128 + tid) * 4 + slice holds, in nibble slot
    (0, 4, 1, 5, 2, 6, 3, 7)[v], weight row 64 wg + 16 warp + lane / 4 + 8 v[1]
    at column 16 (4 half + slice) + 2 (lane % 4) + v[0] + 8 v[2].
    """
    rows, cols = torch.meshgrid(
        torch.arange(TILE_N), torch.arange(TILE_K), indexing="ij"
    )
    # A code is 4 bits, so indices up to 127 go through pack_codes one hex digit at a time.
    digits = [(rows >> 4 * d) & 0xF for d in range(2)]
    digits += [(cols >> 4 * d) & 0xF for d in range(2)]
    slot_of_value = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7])
    unpacked = []
    for codes in digits:
        words = pack_codes(codes.to(torch.uint8).unsqueeze(0)).view(-1, 1)
        unpacked.append((words >> (4 * slot_of_value)) & 0xF)  # [words, 8]
    packed_row = unpacked[0] + 16 * unpacked[1]
    packed_col = unpacked[2] + 16 * unpacked[3]

    wg, half, tid, step, v = torch.meshgrid(
        *(torch.arange(n) for n in (2, 2, 128, 4, 8)), indexing="ij"
    )
    warp, lane = tid // 32, tid % 32
    row = 64 * wg + 16 * warp + lane // 4 + 8 * ((v >> 1) & 1)
    col = 16 * (4 * half + step) + 2 * (lane % 4) + (v & 1) + 8 * ((v >> 2) & 1)
    assert torch.equal(packed_row, row.reshape(-1, 8).to(packed_row.dtype))
    assert torch.equal(packed_col, col.reshape(-1, 8).to(packed_col.dtype))


def test_repack_rejects_other_group_sizes():
    qweight, scales, qzeros = _random_awq(1, 256, 128, "cpu")
    with pytest.raises(ValueError, match="group size"):
        repack_awq_moe_weights(qweight, scales, qzeros, group_size=64)


@pytest.mark.skipif(not _has_tensor_cores(), reason="needs an sm80+ GPU")
def test_fragment_layout_through_mma():
    """The packed A fragments multiply correctly under the shared m16n8k16 layout."""
    module = load_jit(
        "test_w4a16_moe_fragment",
        cuda_files=[str(pathlib.Path(__file__).with_name("w4a16_moe_fragment.cuh"))],
        cuda_wrappers=[("fragment_mma", "fragment_mma")],
        extra_include_paths=[str(KERNEL_PATH / "csrc")],
    )
    gen = torch.Generator(device="cpu").manual_seed(1)
    codes = torch.randint(0, 16, (1, TILE_N, TILE_K), dtype=torch.uint8, generator=gen)
    scales = (torch.rand(TILE_N, generator=gen) * 0.1 + 0.01).to(torch.bfloat16)
    zeros = torch.randint(0, 16, (TILE_N,), dtype=torch.uint8, generator=gen)
    weights = W4A16MoeWeights(
        qweight=pack_codes(codes).view(1, 1, 1, -1),
        scales=scales.view(1, 1, 1, -1),
        zeros=zeros.view(1, 1, 1, -1),
    )
    x = torch.randn(8, TILE_K, generator=gen).to(torch.bfloat16)
    y = torch.empty(TILE_N, 8, dtype=torch.float32, device="cuda")
    module.fragment_mma(
        weights.qweight.view(-1).cuda(),
        scales.cuda(),
        zeros.cuda(),
        x.cuda(),
        y,
    )
    # Ground truth from the unpacked codes, so a packing error cannot cancel out.
    dense = (codes[0].to(torch.bfloat16) - zeros.to(torch.bfloat16)[:, None]) * scales[
        :, None
    ]
    expected = dense.float() @ x.float().t()
    torch.testing.assert_close(y.cpu(), expected, rtol=1e-3, atol=1e-3)


def _routed_reference(a, dense, topk_ids, topk_weights, a_row_divisor):
    """out[r] = dense[topk_ids.flat[r]] @ a[r / divisor], optionally weighted."""
    flat_ids = topk_ids.view(-1).long()
    rows = a[torch.arange(flat_ids.numel(), device=a.device) // a_row_divisor].float()
    out = torch.einsum("rk,rnk->rn", rows, dense[flat_ids].float())
    if topk_weights is not None:
        out = out * topk_weights.view(-1, 1)
    return out


def _routing(num_tokens: int, top_k: int, num_experts: int, seed: int):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    scores = torch.rand(num_tokens, num_experts, generator=gen)
    topk_weights, topk_ids = scores.topk(top_k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
    return topk_ids.to(torch.int32).cuda(), topk_weights.float().cuda()


# (K, N): DeepSeek-V3 gate-up and down projections at TP8, plus a small shape.
_SHAPES = [(7168, 512), (256, 7168), (256, 256)]
# Tokens per expert from decode (1, 4) through the target band (8-32) to prefill.
_TOKENS_PER_EXPERT = [1, 4, 8, 16, 32, 64, 128]


@pytest.mark.skipif(not _is_sm90(), reason="needs an SM90 (Hopper) GPU")
@pytest.mark.parametrize("k,n", _SHAPES)
@pytest.mark.parametrize("tokens_per_expert", _TOKENS_PER_EXPERT)
@pytest.mark.parametrize("weighted", [False, True])
def test_routed_gemm_matches_reference(k, n, tokens_per_expert, weighted):
    num_experts, top_k = 16, 4
    num_tokens = max(1, tokens_per_expert * num_experts // top_k)
    weights = repack_awq_moe_weights(*_random_awq(num_experts, k, n, "cuda"), _GROUP)
    dense = dequantize_reference(weights)
    topk_ids, topk_weights = _routing(num_tokens, top_k, num_experts, seed=k + n)
    a_row_divisor = top_k if n != 7168 else 1
    a_rows = num_tokens if a_row_divisor == top_k else num_tokens * top_k
    a = (torch.randn(a_rows, k, device="cuda") * 0.5).to(torch.bfloat16)

    token_block = select_token_block(
        num_tokens=num_tokens, top_k=top_k, num_experts=num_experts
    )
    sorted_ids, expert_ids, num_post_padded = moe_align_block_size(
        topk_ids, token_block, num_experts
    )
    out = torch.full((num_tokens * top_k, n), float("nan"), device="cuda").to(
        torch.bfloat16
    )
    w4a16_moe_sm90_gemm(
        a=a,
        out=out,
        weights=weights,
        sorted_token_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_post_padded,
        topk_weights=topk_weights.view(-1) if weighted else None,
        token_block=token_block,
        a_row_divisor=a_row_divisor,
    )
    expected = _routed_reference(
        a, dense, topk_ids, topk_weights if weighted else None, a_row_divisor
    )
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)


def _awq_layer(num_experts, hidden, intermediate, device):
    w13 = _random_awq(num_experts, hidden, 2 * intermediate, device, seed=10)
    w2 = _random_awq(num_experts, intermediate, hidden, device, seed=11)
    layer = torch.nn.Module()
    for name, (qweight, scales, qzeros) in (("w13", w13), ("w2", w2)):
        for suffix, t in (("qweight", qweight), ("scales", scales), ("qzeros", qzeros)):
            layer.register_parameter(
                f"{name}_{suffix}", torch.nn.Parameter(t.clone(), requires_grad=False)
            )
    layer.intermediate_size_per_partition = intermediate
    dense13, dense2 = _awq_dense(*w13), _awq_dense(*w2)
    return layer, dense13, dense2


def _moe_reference(hidden_states, dense13, dense2, topk_ids, topk_weights):
    num_tokens, top_k = topk_ids.shape
    gate_up = _routed_reference(hidden_states, dense13, topk_ids, None, top_k)
    gate, up = gate_up.chunk(2, dim=-1)
    act = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    down = _routed_reference(act, dense2, topk_ids, topk_weights, 1)
    return down.view(num_tokens, top_k, -1).sum(1)


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
@pytest.mark.parametrize("tokens_per_expert", [1, 8, 16, 32, 128])
def test_awq_moe_matches_reference_and_marlin(tokens_per_expert):
    from sglang.srt.layers.moe.utils import MoeRunnerBackend

    num_experts, top_k, hidden, intermediate = 32, 8, 1024, 256
    num_tokens = max(1, tokens_per_expert * num_experts // top_k)
    topk_ids, topk_weights = _routing(num_tokens, top_k, num_experts, seed=3)
    hidden_states = (torch.randn(num_tokens, hidden, device="cuda") * 0.5).to(
        torch.bfloat16
    )

    layer, dense13, dense2 = _awq_layer(num_experts, hidden, intermediate, "cuda")
    ours = _run_awq_moe(
        MoeRunnerBackend.W4A16_SM90, layer, hidden_states, topk_ids, topk_weights
    )
    expected = _moe_reference(hidden_states, dense13, dense2, topk_ids, topk_weights)
    torch.testing.assert_close(ours.float(), expected, rtol=3e-2, atol=3e-2)

    marlin_layer, _, _ = _awq_layer(num_experts, hidden, intermediate, "cuda")
    marlin = _run_awq_moe(
        MoeRunnerBackend.MARLIN, marlin_layer, hidden_states, topk_ids, topk_weights
    )
    torch.testing.assert_close(ours.float(), marlin.float(), rtol=3e-2, atol=3e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
