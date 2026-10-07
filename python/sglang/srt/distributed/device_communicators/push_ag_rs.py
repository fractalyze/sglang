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

The kernels stage through the group's CustomAllReduceV2 push workspace, whose
slot the group enlarges to at least ``PUSH_SLOT_BYTES`` when this path is on;
the all-reduce thresholds do not move with it. Which sizes take the push path, and
at what launch grid, comes from the checked-in ``sp_collective`` table for the
device; everything else, and every device without a table, stays on NCCL.
"""

import logging
from typing import Optional

import torch

from sglang.kernels.ops.communication import sp_collective
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
    default_max_push_size,
)

logger = logging.getLogger(__name__)

# Per-peer push slot. Holds a rank's shard of 1024 tokens at hidden 7168 bf16
# on 8 ranks; larger calls fall back to NCCL.
PUSH_SLOT_BYTES = 2 * 1024 * 1024


def custom_all_reduce_kwargs(ca_class: type, world_size: int) -> dict:
    """Constructor arguments that let a group's communicator carry this path.

    The push all-gather / reduce-scatter stage a rank's whole shard through the
    communicator's push slot. The slot is only ever enlarged: a smaller one
    would lower the all-reduce's one-shot push threshold with it.
    """
    if not issubclass(ca_class, CustomAllReduceV2):
        return {}
    if default_max_push_size(world_size) >= PUSH_SLOT_BYTES:
        return {}
    return {"max_push_size": PUSH_SLOT_BYTES}


def create(comm: Optional[object]) -> Optional["PushAllGatherReduceScatter"]:
    """The push path over ``comm``, or None (NCCL) when ``comm`` cannot carry it."""
    if (
        isinstance(comm, CustomAllReduceV2)
        and not comm.disabled
        and comm.has_multicast
        and comm.max_push_size >= PUSH_SLOT_BYTES
    ):
        return PushAllGatherReduceScatter(comm)
    logger.warning(
        "SGLANG_OPT_USE_PUSH_AG_RS needs CustomAllReduceV2 with multicast on the "
        "TP group; using NCCL."
    )
    return None


class PushAllGatherReduceScatter:
    def __init__(self, comm: CustomAllReduceV2) -> None:
        self.comm = comm
        self.device = comm.device
        self.world_size = comm.world_size
        self._dispatches: dict[
            tuple[str, int, int], Optional[sp_collective.Dispatch]
        ] = {}
        sp_collective.register_comm(comm.obj)

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
            input.dtype == torch.bfloat16
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
