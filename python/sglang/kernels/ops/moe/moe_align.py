from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_moe_align_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "moe_align_block_size",
        *args,
        cuda_files=["moe/moe_align_kernel.cu"],
        cuda_wrappers=[
            ("moe_align_block_size", f"MoeAlignBlockSizeKernel<{args}>::run"),
        ],
    )


# Stable placement cuts topk_ids into chunks of at least this many entries, one warp each.
# Arbitrary, not tuned; smaller chunks spread a decode batch over more SMs.
_MIN_CHUNK_ENTRIES = 64
# The scan walks every chunk serially per expert, so the chunk count is capped.
_MAX_CHUNKS = 256


def _num_chunks(numel: int) -> int:
    return min(max(1, triton.cdiv(numel, _MIN_CHUNK_ENTRIES)), _MAX_CHUNKS)


def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    cumsum_buffer: torch.Tensor,
    pad_sorted_token_ids: bool = False,
) -> None:
    """Align and sort expert token ids into block-padded output buffers.

    For num_experts <= 1024 each expert's entries are listed in ascending flat
    index of topk_ids, so the output is identical from run to run.
    """
    module = _jit_moe_align_module(topk_ids.dtype)
    chunk_counts = torch.empty(
        (_num_chunks(topk_ids.numel()), num_experts),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    module.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        chunk_counts,
        pad_sorted_token_ids,
    )
