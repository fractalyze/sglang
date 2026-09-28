# SPDX-License-Identifier: Apache-2.0
"""CUDA graphs of a module's pure-GPU forward, one per input shape."""

from collections.abc import Callable

import torch
from torch import nn


class ShapeKeyedCudaGraphs:
    """Capture ``fn(*inputs)`` once per input shape and replay it afterwards.

    ``fn`` must be pure GPU work (no host syncs, no data-dependent shapes)
    returning a tuple of tensors. A replay reads the owner's parameters where
    they were at capture, so the graphs are dropped and recaptured whenever
    any parameter of ``owner`` has moved (a component offloaded to the CPU
    comes back in new memory).
    """

    def __init__(self, owner: nn.Module):
        self._owner = owner
        self._graphs: dict[tuple, tuple] = {}
        self._addresses: tuple[int, ...] | None = None

    def run(self, fn: Callable[..., tuple[torch.Tensor, ...]], *inputs: torch.Tensor):
        addresses = tuple(p.data_ptr() for p in self._owner.parameters())
        if addresses != self._addresses:
            self._graphs.clear()
            self._addresses = addresses
        key = tuple((tuple(t.shape), t.dtype, t.device) for t in inputs)
        if key not in self._graphs:
            self._graphs[key] = self._capture(fn, inputs)
        graph, static_inputs, static_outputs = self._graphs[key]
        for static, value in zip(static_inputs, inputs):
            static.copy_(value)
        graph.replay()
        return tuple(out.clone() for out in static_outputs)

    @staticmethod
    def _capture(fn, inputs):
        static_inputs = tuple(t.clone() for t in inputs)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):  # warm up lazy initialization outside the capture
            for _ in range(2):
                fn(*static_inputs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_outputs = fn(*static_inputs)
        return graph, static_inputs, static_outputs
