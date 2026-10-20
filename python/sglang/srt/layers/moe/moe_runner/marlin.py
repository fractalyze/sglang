from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    RunnerInput,
    RunnerOutput,
    register_fused_func,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


@triton.jit
def _unpack_packed_topk_kernel(packed_ptr, ids_ptr, w_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    packed = tl.load(
        packed_ptr + offs, mask=mask
    )  # int32 (id << 16) | bf16-weight-bits
    tl.store(ids_ptr + offs, packed >> 16, mask=mask)  # expert id (high 16 bits)
    wbits = (packed & 0xFFFF).to(tl.int16)  # bf16 weight bits (low 16 bits)
    tl.store(
        w_ptr + offs, wbits.to(tl.bfloat16, bitcast=True).to(tl.float32), mask=mask
    )


def _fused_unpack_packed_topk(packed: torch.Tensor):
    """Single-launch inverse of the fused topk pack ((id << 16) | bf16-weight-bits).

    Returns (topk_ids int32, topk_weights float32). Collapses the ~5 elementwise
    ops (shift / mask / int16 / bitcast / cast) the torch reference emits per call
    into one Triton launch. Mirrors _pack_topk_kernel (trtllm_lora_temp/topk_pack).
    """
    packed = packed.contiguous()
    ids = torch.empty_like(packed, dtype=torch.int32)
    w = torch.empty(packed.shape, dtype=torch.float32, device=packed.device)
    numel = packed.numel()
    if numel:
        BLOCK = 1024
        _unpack_packed_topk_kernel[(triton.cdiv(numel, BLOCK),)](
            packed, ids, w, numel, BLOCK=BLOCK
        )
    return ids, w


@dataclass
class MarlinRunnerInput(RunnerInput):
    """Input bundle passed to the Marlin runner core."""

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.MARLIN


@dataclass
class MarlinRunnerOutput(RunnerOutput):
    """Output bundle returned from the Marlin runner core."""

    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.MARLIN


@dataclass
class MarlinMoeQuantInfo(MoeQuantInfo):
    """Quantization payload consumed by the Marlin backend."""

    w13_qweight: torch.Tensor
    w2_qweight: torch.Tensor
    w13_scales: torch.Tensor
    w2_scales: torch.Tensor
    w13_g_idx_sort_indices: Optional[torch.Tensor]
    w2_g_idx_sort_indices: Optional[torch.Tensor]
    weight_bits: int

    # GPTQ specific (Optional)
    w13_g_idx: Optional[torch.Tensor] = None
    w2_g_idx: Optional[torch.Tensor] = None
    is_k_full: bool = True

    # AWQ specific (Optional)
    w13_qzeros: Optional[torch.Tensor] = None
    w2_qzeros: Optional[torch.Tensor] = None

    # Optional
    expert_map: Optional[torch.Tensor] = None
    global_num_experts: int = -1
    w13_global_scale: Optional[torch.Tensor] = None
    w2_global_scale: Optional[torch.Tensor] = None
    w13_bias: Optional[torch.Tensor] = None
    w2_bias: Optional[torch.Tensor] = None


@register_fused_func("none", "marlin")
def fused_experts_none_to_marlin(
    dispatch_output: StandardDispatchOutput,
    quant_info: MarlinMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output

    # The fused gate+topk kernel (SGLANG_OPT_USE_FUSED_GATE_TOPK) emits a
    # PackedTopKOutput -- int32 (expert_id << 16) | bf16-weight-bits -- for the
    # FlashInfer trtllm routed kernel, which unpacks device-side. The Marlin
    # runner reads topk_ids / topk_weights separately, so unpack it here.
    if not hasattr(topk_output, "topk_weights"):
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        topk_ids, topk_weights = _fused_unpack_packed_topk(topk_output.packed_topk_ids)
        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=topk_output.router_logits,
        )

    if runner_config.is_gated:
        assert runner_config.activation in {
            "silu",
            "situ",
        }, f"Only gated SiLU/SiTU is supported, got {runner_config.activation}."
    elif runner_config.activation not in {"silu", "relu2"}:
        raise ValueError(
            f"Unsupported Marlin MoE activation: {runner_config.activation}"
        )

    output = _run_marlin(
        hidden_states=hidden_states,
        topk_ids=topk_output.topk_ids,
        topk_weights=topk_output.topk_weights,
        gating_output=topk_output.router_logits,
        quant_info=quant_info,
        runner_config=runner_config,
        expert_map=quant_info.expert_map,
        global_num_experts=quant_info.global_num_experts,
        routed_scaling_factor=runner_config.routed_scaling_factor,
    )

    return StandardCombineInput(
        hidden_states=output,
    )


def _run_marlin(
    *,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gating_output: torch.Tensor,
    quant_info: MarlinMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    expert_map: Optional[torch.Tensor],
    global_num_experts: int,
    routed_scaling_factor: Optional[float],
    skip_invalid_ids: Optional[bool] = None,
) -> torch.Tensor:
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

    # Use a per-call workspace so captured graphs cannot alias Marlin's
    # inter-block reduction locks and deadlock during capture.
    workspace = marlin_make_workspace(hidden_states.device, max_blocks_per_sm=4)

    marlin_hidden_states = hidden_states
    if (
        quant_info.weight_bits == 4
        and quant_info.w13_qzeros is None
        and quant_info.w2_qzeros is None
        and quant_info.w13_scales.dtype == torch.float8_e8m0fnu
        and quant_info.w2_scales.dtype == torch.float8_e8m0fnu
        and hidden_states.dtype == torch.float16
    ):
        # MXFP4(E8M0) Marlin kernels are only numerically valid on the bf16
        # activation path. The fp16 + E8M0 path is intentionally not generated
        # in sgl-kernel, so upcast activations here and cast the result back.
        marlin_hidden_states = hidden_states.to(torch.bfloat16)

    return fused_marlin_moe(
        hidden_states=marlin_hidden_states,
        w1=quant_info.w13_qweight,
        w2=quant_info.w2_qweight,
        w1_scale=quant_info.w13_scales,
        w2_scale=quant_info.w2_scales,
        gating_output=gating_output,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        skip_invalid_ids=skip_invalid_ids,
        g_idx1=quant_info.w13_g_idx,
        g_idx2=quant_info.w2_g_idx,
        sort_indices1=quant_info.w13_g_idx_sort_indices,
        sort_indices2=quant_info.w2_g_idx_sort_indices,
        w1_zeros=quant_info.w13_qzeros,
        w2_zeros=quant_info.w2_qzeros,
        w1_global_scale=quant_info.w13_global_scale,
        w2_global_scale=quant_info.w2_global_scale,
        w1_bias=quant_info.w13_bias,
        w2_bias=quant_info.w2_bias,
        workspace=workspace,
        num_bits=quant_info.weight_bits,
        is_k_full=quant_info.is_k_full,
        # Avoid aliasing the MoE input buffer until Marlin output semantics are
        # fully validated across shared-expert and overlap paths.
        inplace=False,
        routed_scaling_factor=routed_scaling_factor,
        clamp_limit=(
            runner_config.gemm1_clamp_limit
            if runner_config.gemm1_alpha is not None
            else runner_config.swiglu_limit
        ),
        gemm1_alpha=runner_config.gemm1_alpha,
        activation=runner_config.activation,
        is_gated=runner_config.is_gated,
    ).to(hidden_states.dtype)


def _run_marlin_local_experts(
    *,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    quant_info: MarlinMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> torch.Tensor:
    # DeepEP's ids are already local (-1 = another rank's expert), and the DeepEP
    # paths apply the routed scaling factor outside the MoE runner.
    return _run_marlin(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        gating_output=topk_weights,
        quant_info=quant_info,
        runner_config=runner_config,
        expert_map=None,
        global_num_experts=quant_info.w13_qweight.shape[0],
        routed_scaling_factor=None,
        skip_invalid_ids=True,
    )


# Caps Marlin's scratch (it scales with rows x max(2N, K)) for any prefill size.
_NORMAL_MAX_PAIRS = 16384


def marlin_moe_deepep_normal(
    *,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    quant_info: MarlinMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> torch.Tensor:
    # Only local (token, expert) pairs run; most of a rank's top_k slots are remote.
    # Eager only (dynamic shapes), as DeepEP normal mode is.
    t_idx, k_idx = (topk_ids >= 0).nonzero(as_tuple=True)
    output = torch.zeros_like(hidden_states)
    for start in range(0, t_idx.numel(), _NORMAL_MAX_PAIRS):
        t = t_idx[start : start + _NORMAL_MAX_PAIRS]
        k = k_idx[start : start + _NORMAL_MAX_PAIRS]
        expert_out = _run_marlin_local_experts(
            hidden_states=hidden_states[t],
            topk_ids=topk_ids[t, k].to(torch.int32).unsqueeze(1),
            topk_weights=topk_weights[t, k].to(torch.float32).unsqueeze(1),
            quant_info=quant_info,
            runner_config=runner_config,
        )
        output.index_add_(0, t, expert_out)
    return output


def marlin_moe_deepep_ll(
    *,
    hidden_states: torch.Tensor,
    masked_m: torch.Tensor,
    max_rows: int,
    quant_info: MarlinMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> torch.Tensor:
    # [E, M, H] with masked_m[e] valid rows per expert -> max_rows compact rows,
    # one expert each -> back to [E, M, H] unweighted (the combine weights them).
    # Static shapes, so CUDA-graph safe; max_rows must bound sum(masked_m).
    from sglang.kernels.ops.moe.ep_moe_kernels import (
        expand_to_masked_slab,
        masked_slab_to_expand,
    )

    num_experts, max_m, _ = hidden_states.shape
    psum = torch.cumsum(masked_m, dim=0, dtype=torch.int32)
    compact = masked_slab_to_expand(
        input_tensor=hidden_states,
        psum_num_recv_tokens_per_expert=psum,
        total_expanded_tokens=max_rows,
        expert_alignment=1,
    )
    rows = torch.arange(max_rows, dtype=torch.int32, device=hidden_states.device)
    expert_ids = torch.searchsorted(psum, rows, right=True, out_int32=True)
    expert_ids = torch.where(expert_ids < num_experts, expert_ids, -1)
    compact_out = _run_marlin_local_experts(
        hidden_states=compact,
        topk_ids=expert_ids.unsqueeze(1),
        topk_weights=torch.ones(
            (max_rows, 1), dtype=torch.float32, device=hidden_states.device
        ),
        quant_info=quant_info,
        runner_config=runner_config,
    )
    output, _, _ = expand_to_masked_slab(
        recv_x=compact_out,
        recv_x_scale=None,
        psum_num_recv_tokens_per_expert=psum,
        num_local_experts=num_experts,
        max_m=max_m,
        expert_alignment=1,
    )
    return output.view(num_experts, max_m, -1)


@register_fused_func("deepep", "marlin")
def fused_experts_deepep_to_marlin(
    dispatch_output,
    quant_info: MarlinMoeQuantInfo,
    runner_config: MoeRunnerConfig,
):
    from sglang.srt.distributed.parallel_state import (
        get_moe_expert_parallel_world_size,
    )
    from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLCombineInput,
        DeepEPNormalCombineInput,
    )

    assert runner_config.is_gated and runner_config.activation in {"silu", "situ"}, (
        f"Marlin behind DeepEP supports gated SiLU/SiTU, got {runner_config.activation}."
    )
    assert dispatch_output.hidden_states_scale is None, (
        "Marlin takes unquantized activations; pass --deepep-dispatcher-output-dtype bf16."
    )

    if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
        output = marlin_moe_deepep_normal(
            hidden_states=dispatch_output.hidden_states,
            topk_ids=dispatch_output.topk_ids,
            topk_weights=dispatch_output.topk_weights,
            quant_info=quant_info,
            runner_config=runner_config,
        )
        return DeepEPNormalCombineInput(
            hidden_states=output,
            topk_ids=dispatch_output.topk_ids,
            topk_weights=dispatch_output.topk_weights,
        )

    assert DispatchOutputChecker.format_is_deepep_ll(dispatch_output)
    if torch.cuda.is_current_stream_capturing():
        # A decode CUDA graph runs one batch size on every DP rank, so each
        # rank sends at most num_tokens x top_k rows here.
        num_tokens, top_k = dispatch_output.topk_ids.shape
        max_rows = get_moe_expert_parallel_world_size() * num_tokens * top_k
    else:
        # Eager batches can differ across ranks: size by the exact count.
        max_rows = int(dispatch_output.masked_m.sum().item())
    output = marlin_moe_deepep_ll(
        hidden_states=dispatch_output.hidden_states,
        masked_m=dispatch_output.masked_m,
        max_rows=max(max_rows, 1),
        quant_info=quant_info,
        runner_config=runner_config,
    )
    return DeepEPLLCombineInput(
        hidden_states=output,
        topk_ids=dispatch_output.topk_ids,
        topk_weights=dispatch_output.topk_weights,
    )


# ===== TO BE REFACTORED ====
from sglang.srt.lora.marlin_lora_temp import sgl_backend  # noqa: E402,F401

# ===== END TO BE REFACTORED ====
