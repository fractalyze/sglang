from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.moe.w4a16_moe_sm90 import (
    TILE_K,
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
        # Gate-up runs a few waves of tiles, so a partial last wave is a large
        # share of its time; down runs dozens, where the same tail barely matters.
        half_tile_tail=True,
        **routing,
    )
    activated = hidden_states.new_empty((num_rows, intermediate_size))
    silu_and_mul(gate_up, activated)

    # The top-k sum reads every routed row. moe_align_block_size lists them all, -1
    # ones in expert -1 blocks, so with zero_unrouted the GEMM writes each one. Under
    # EP most rows belong to other ranks; the buffer is zero-filled and they are skipped.
    zero_unrouted = runner_config.num_local_experts == runner_config.num_experts
    down_shape = (num_rows, hidden_states.shape[1])
    if zero_unrouted:
        down = hidden_states.new_empty(down_shape)
    else:
        down = hidden_states.new_zeros(down_shape)
    w4a16_moe_sm90_gemm(
        a=activated,
        out=down,
        weights=quant_info.w2,
        topk_weights=topk_weights.to(torch.float32).contiguous().view(-1),
        a_row_divisor=1,
        zero_unrouted=zero_unrouted,
        **routing,
    )

    output = torch.empty_like(hidden_states)
    routed_scaling_factor = runner_config.routed_scaling_factor
    moe_sum_reduce(
        down.view(num_tokens, top_k, -1),
        output,
        1.0 if routed_scaling_factor is None else routed_scaling_factor,
    )
    return StandardCombineInput(hidden_states=output)
