# SPDX-License-Identifier: Apache-2.0
"""One-shot NVLink push all-gather / reduce-scatter for a GroupCoordinator.

Under DP attention every layer gathers the ranks' hidden states before the MLP
and reduce-scatters them after it, at a few MiB per call. This routes those
equal-chunk bf16 calls to the ``sp_collective`` push kernels: the all-gather
multicast-stores each rank's shard once, and the reduce-scatter writes each
row shard straight to its owner, which sums the ranks in fixed rank order with
fp32 accumulation. The result does not depend on which rank reduces it, and no
host collective runs per call, so the path is deterministic and CUDA-graph
safe.

The kernels stage through a CustomAllReduceV2 push workspace of their own,
sized for a rank's shard; the group's all-reduce workspace is sized for
all-reduce and is far smaller. Which sizes take the push path, and at what
launch grid, comes from the checked-in ``sp_collective`` table for the device;
everything else, and every device without a table, stays on NCCL.
"""

import logging
from typing import Optional

import torch
from torch.distributed import ProcessGroup

from sglang.kernels.ops.communication import sp_collective

logger = logging.getLogger(__name__)

# Per-peer push slot. Holds a rank's shard of 1024 tokens at hidden 7168 bf16
# on 8 ranks; larger calls fall back to NCCL.
_PUSH_SLOT_BYTES = 2 * 1024 * 1024


class PushAllGatherReduceScatter:
    def __init__(self, group: ProcessGroup, device: torch.device) -> None:
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            CustomAllReduceV2,
        )

        self.device = device
        self.comm = CustomAllReduceV2(
            group=group,
            device=device,
            max_push_size=_PUSH_SLOT_BYTES,
            max_pull_size=0,
        )
        self.disabled = self.comm.disabled or not self.comm.has_multicast
        if self.disabled:
            logger.warning(
                "Push all-gather / reduce-scatter needs CustomAllReduceV2 with "
                "multicast on this group; using NCCL."
            )
            return
        self.world_size = self.comm.world_size
        self._dispatches: dict[
            tuple[str, int, int], Optional[sp_collective.Dispatch]
        ] = {}
        sp_collective.register_comm(self.comm.obj)

    def _dispatch(
        self, kind: str, hidden_size: int, num_tokens: int
    ) -> Optional[sp_collective.Dispatch]:
        key = (kind, hidden_size, num_tokens)
        if key not in self._dispatches:
            dispatch = sp_collective.get_dispatch(
                kind, self.world_size, hidden_size, num_tokens, self.device
            )
            if dispatch is not None and dispatch.strategy != "push":
                dispatch = None
            self._dispatches[key] = dispatch
        return self._dispatches[key]

    def _eligible(self, output: torch.Tensor, input: torch.Tensor) -> bool:
        return (
            not self.disabled
            and input.dtype == torch.bfloat16
            and output.dtype == torch.bfloat16
            and input.ndim == 2
            and input.is_contiguous()
            and output.is_contiguous()
        )

    def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> bool:
        """Gather ``input`` rows into ``output``; False leaves it to NCCL."""
        if not self._eligible(output, input) or input.shape[0] == 0:
            return False
        if output.numel() != input.numel() * self.world_size:
            return False
        if input.numel() * input.element_size() > self.comm.max_push_size:
            return False
        global_tokens = input.shape[0] * self.world_size
        dispatch = self._dispatch("all_gather", input.shape[1], global_tokens)
        if dispatch is None:
            return False
        sp_collective.all_gather(self.world_size, input, output, tuning=dispatch.tuning)
        return True

    def reduce_scatter(self, output: torch.Tensor, input: torch.Tensor) -> bool:
        """Sum ``input`` over ranks into this rank's ``output`` rows; False
        leaves it to NCCL."""
        if not self._eligible(output, input) or input.shape[0] == 0:
            return False
        if input.shape[0] % self.world_size != 0:
            return False
        if input.numel() != output.numel() * self.world_size:
            return False
        if output.numel() * output.element_size() > self.comm.max_push_size:
            return False
        dispatch = self._dispatch("reduce_scatter", input.shape[1], input.shape[0])
        if dispatch is None:
            return False
        sp_collective.reduce_scatter_res(
            self.world_size, input, output, tuning=dispatch.tuning
        )
        return True

    def close(self) -> None:
        self.comm.close()
