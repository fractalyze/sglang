from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.moe.w4a16_moe_sm90 import (
    TILE_K,
    TILE_N,
    W4A16MoeWeights,
    select_token_block,
    w4a16_moe_sm90_gemm,
)
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


class W4A16Sm90MoeQuantInfo(MoeQuantInfo):
    """Gate-up and down projections in the w4a16_moe_sm90 weight layout."""

    def __init__(self, *, w13: W4A16MoeWeights, w2: W4A16MoeWeights):
        self.w13 = w13
        self.w2 = w2


# Tokens per forward the down GEMM's fused top-k sum covers; larger batches
# (prefill) take the separate moe_sum_reduce. Arbitrary: covers decode at any
# served concurrency; the workspace holds max_tokens * hidden / TILE_N int32.
_TOPK_SUM_MAX_TOKENS = 16384


@functools.cache
def _topk_sum_arrivals(device: torch.device, hidden_size: int) -> torch.Tensor:
    """Zeroed once; every fused launch leaves it zero, so graph replays share it."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "w4a16_sm90 must run eagerly once before CUDA graph capture, "
            "so its top-k sum workspace is not allocated from a graph pool"
        )
    return torch.zeros(
        _TOPK_SUM_MAX_TOKENS * (hidden_size // TILE_N), dtype=torch.int32, device=device
    )


def _standard_topk(topk_output):
    from sglang.srt.layers.moe.moe_runner.marlin import _fused_unpack_packed_topk
    from sglang.srt.layers.moe.topk import PackedTopKOutput

    if isinstance(topk_output, PackedTopKOutput):
        topk_ids, topk_weights = _fused_unpack_packed_topk(topk_output.packed_topk_ids)
        return topk_ids, topk_weights
    return topk_output.topk_ids, topk_output.topk_weights


@register_fused_func("none", "w4a16_sm90")
def fused_experts_none_to_w4a16_sm90(
    dispatch_output: StandardDispatchOutput,
    quant_info: W4A16Sm90MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sgl_kernel import moe_sum_reduce

    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    if not (runner_config.is_gated and runner_config.activation == "silu"):
        raise ValueError("w4a16_sm90 supports gated SiLU experts only")
    if runner_config.gemm1_alpha is not None or runner_config.swiglu_limit is not None:
        raise ValueError("w4a16_sm90 does not support clamped or alpha-scaled SwiGLU")

    hidden_states = dispatch_output.hidden_states
    topk_ids, topk_weights = _standard_topk(dispatch_output.topk_output)
    num_tokens, top_k = topk_ids.shape
    num_experts = quant_info.w13.qweight.shape[0]
    intermediate_size = quant_info.w2.qweight.shape[2] * TILE_K

    token_block = select_token_block(
        num_tokens=num_tokens, top_k=top_k, num_experts=num_experts
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, token_block, num_experts
    )
    routing = dict(
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        token_block=token_block,
    )

    num_rows = num_tokens * top_k
    gate_up = hidden_states.new_empty((num_rows, 2 * intermediate_size))
    w4a16_moe_sm90_gemm(
        a=hidden_states,
        out=gate_up,
        weights=quant_info.w13,
        topk_weights=None,
        a_row_divisor=top_k,
        **routing,
    )
    activated = hidden_states.new_empty((num_rows, intermediate_size))
    silu_and_mul(gate_up, activated)

    output = torch.empty_like(hidden_states)
    routed_scaling_factor = runner_config.routed_scaling_factor
    sum_scale = 1.0 if routed_scaling_factor is None else routed_scaling_factor
    down_weights = topk_weights.to(torch.float32).contiguous().view(-1)
    # Under EP most routed rows belong to other ranks, and the fused sum would store
    # a zero partial for each of them, so EP keeps the separate reduce.
    experts_all_local = runner_config.num_local_experts == runner_config.num_experts
    if experts_all_local and num_tokens <= _TOPK_SUM_MAX_TOKENS:
        # The epilogue writes every routed row, zeros for no local expert, and sums them.
        # moe_align_block_size lists every row, -1 ones included (ignore_invalid_expert
        # off); an unlisted row would leave its counter non-zero for later launches.
        w4a16_moe_sm90_gemm(
            a=activated,
            out=hidden_states.new_empty((num_rows, hidden_states.shape[1])),
            weights=quant_info.w2,
            topk_weights=down_weights,
            a_row_divisor=1,
            sum_out=output,
            arrivals=_topk_sum_arrivals(output.device, output.shape[1]),
            sum_scale=sum_scale,
            **routing,
        )
        return StandardCombineInput(hidden_states=output)

    # Rows routed to no local expert are never written, and the top-k sum reads them.
    down = hidden_states.new_zeros((num_rows, hidden_states.shape[1]))
    w4a16_moe_sm90_gemm(
        a=activated,
        out=down,
        weights=quant_info.w2,
        topk_weights=down_weights,
        a_row_divisor=1,
        **routing,
    )
    moe_sum_reduce(down.view(num_tokens, top_k, -1), output, sum_scale)
    return StandardCombineInput(hidden_states=output)
