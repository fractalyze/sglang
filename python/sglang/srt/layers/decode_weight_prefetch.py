"""Stage the next layer's weights in L2 while a collective leaves HBM idle.

A decode step reads every weight once, so a GEMM at small M is bound by HBM;
a collective between two GEMMs moves activations over NVLink and leaves HBM
idle. Issuing the next GEMM's weights to L2 on a side stream as the collective
starts overlaps the two, up to what fits in L2.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, List, Optional

import torch

from sglang.kernels.ops.memory.l2_prefetch import l2_prefetch, plan_l2_prefetch


class DecodeWeightPrefetch:
    """One layer's prefetch of `weights()`, at most `budget_bytes` in read order.

    The range table holds raw addresses, so it is built on the first call
    outside graph capture, once weight post-processing has run, and keeps the
    tensors alive. A call inside capture before that skips the prefetch.
    """

    def __init__(
        self,
        *,
        weights: Callable[[], List[torch.Tensor]],
        budget_bytes: int,
        stream: torch.cuda.Stream,
    ):
        self._weights = weights
        self._budget_bytes = budget_bytes
        self._stream = stream
        self._planned: Optional[List[torch.Tensor]] = None
        self._ranges: Optional[torch.Tensor] = None

    @contextmanager
    def overlapping(self) -> Iterator[None]:
        """Prefetch on the side stream for the duration of the block."""
        if self._ranges is None and not torch.cuda.is_current_stream_capturing():
            self._planned = self._weights()
            self._ranges = plan_l2_prefetch(self._planned, self._budget_bytes)
        if self._ranges is None:
            yield
            return
        main = torch.cuda.current_stream()
        self._stream.wait_stream(main)
        with torch.cuda.stream(self._stream):
            l2_prefetch(self._ranges)
        yield
        main.wait_stream(self._stream)


def attach_decode_weight_prefetch(
    layers, layer_ids: List[int], *, budget_bytes: int
) -> None:
    """Give each MoE layer followed by a local layer a prefetch of that layer's
    pre-attention weights, issued during its FFN output collective."""
    stream = torch.cuda.Stream()
    for layer_id, next_id in zip(layer_ids, layer_ids[1:]):
        layer = layers[layer_id]
        if not layer.is_layer_sparse:
            continue
        layer.decode_weight_prefetch = DecodeWeightPrefetch(
            weights=layers[next_id].self_attn.decode_weights_before_core,
            budget_bytes=budget_bytes,
            stream=stream,
        )
