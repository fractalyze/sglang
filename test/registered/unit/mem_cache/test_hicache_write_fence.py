"""HiCache write-through must not copy KV that an in-flight forward still writes.

Under the overlap scheduler a finished request is cached while the next forward,
which writes the KV of its last output token, is still queued on the forward
stream. A D2H copy ordered only after the scheduler stream reads that token
half-written (seen on Gemma-4: one token per finished turn, a run of layers),
and a later load-back restores the stale KV.
"""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import CacheOperation
from sglang.srt.mem_cache import l2_transfer as transfer_module
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.l2_transfer import L2TransferEngine
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Stream:
    def __init__(self, name, log):
        self.name = name
        self.log = log

    def wait_stream(self, other):
        self.log.append(f"{self.name} waits {other.name}")


class _Event:
    def __init__(self, enable_timing=False):
        pass

    def record(self):
        pass

    def wait(self, stream):
        pass


class TestHiCacheWriteThroughFence(CustomTestCase):
    def setUp(self):
        transfer_module._timing_events_supported.cache_clear()
        self.addCleanup(transfer_module._timing_events_supported.cache_clear)

    def _start_writing(self):
        """Submit one write-through and return the ordered stream operations."""
        log = []

        class DeviceModule:
            Event = _Event

            @staticmethod
            def Stream():
                return _Stream("d2h", log)

            @staticmethod
            @contextmanager
            def stream(stream):
                yield

        class HostPool:
            def backup_from_device_all_layer(self, *args):
                log.append("copy")

        op = CacheOperation(
            host_indices=torch.arange(0, 4),
            device_indices=torch.arange(4, 8),
            node_id=1,
        )
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.write_queue = [op]
        controller.ack_write_queue = []
        controller.io_backend = "kernel"
        controller.mem_pool_host = SimpleNamespace(layout="layer_first")
        controller.mem_pool_device = None
        controller._num_tokens_by_pool = lambda op: {}
        controller._transfer_num_bytes = lambda op: 0
        controller.load_fence_stream = _Stream("forward", log)
        controller.move_hybrid_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices, None)
        )
        controller._l2_transfers = lambda host, device, transfers: [
            transfer_module.L2Transfer(
                host_pool=HostPool(),
                device_pool=None,
                host_indices=host,
                device_indices=device,
            )
        ]
        with mock.patch.object(transfer_module, "device_module", DeviceModule):
            controller.l2_transfer_engine = L2TransferEngine("kernel")
            controller.start_writing()
        self.assertEqual(len(controller.ack_write_queue), 1)
        return log

    def test_fence_orders_copy_after_queued_forwards(self):
        with envs.SGLANG_OPT_HICACHE_FENCE_WRITE_THROUGH.override(True):
            self.assertEqual(self._start_writing(), ["d2h waits forward", "copy"])

    def test_unfenced_copy_does_not_wait_for_the_forward_stream(self):
        # The default path the switch closes: nothing orders the copy after
        # the forward still writing the node's KV.
        with envs.SGLANG_OPT_HICACHE_FENCE_WRITE_THROUGH.override(False):
            self.assertEqual(self._start_writing(), ["copy"])


if __name__ == "__main__":
    unittest.main()
