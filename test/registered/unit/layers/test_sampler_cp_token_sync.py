"""Attention-CP replicas of one DP shard agree on every sampled token.

The replicas sample from logits that are not bit-identical, so each could
pick a different token; a replica that alone samples EOS would finish the
request a step early and the shard's batches would diverge. CP rank 0's
tokens are broadcast over the CP group whether or not the TP sync is on: it
returns the shard's output, so its logprobs stay those of the emitted token.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import sampler as sampler_module
from sglang.srt.layers.sampler import Sampler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

TP_GROUP = object()
CP_GROUP = object()
CP_LEADER = 6  # not 0, so passing the group-local src would fail


class TestSamplerCpTokenSync(CustomTestCase):
    def sync(self, cp_group, cp_world_size=2, sync_across_tp=False):
        sampler = Sampler.__new__(Sampler)
        torch.nn.Module.__init__(sampler)
        sampler.tp_sync_group = TP_GROUP
        sampler.cp_sync_group = cp_group
        ids = torch.tensor([3, 5])
        calls = []

        def all_reduce(tensor, op, group):
            calls.append(("all_reduce", op, group, tensor is ids))

        def broadcast(tensor, src, group):
            calls.append(("broadcast", src, group, tensor is ids))

        def get_global_rank(group, group_rank):
            self.assertIs(group, CP_GROUP)
            return CP_LEADER + group_rank

        with (
            patch.object(sampler_module, "SYNC_TOKEN_IDS_ACROSS_TP", sync_across_tp),
            patch.object(
                sampler_module.dist, "get_world_size", return_value=cp_world_size
            ),
            patch.object(
                sampler_module.dist, "get_global_rank", side_effect=get_global_rank
            ),
            patch.object(torch.distributed, "all_reduce", side_effect=all_reduce),
            patch.object(torch.distributed, "broadcast", side_effect=broadcast),
        ):
            sampler._sync_token_ids_across_tp(ids, SimpleNamespace(grammars=None))
        return calls

    def test_cp_ranks_take_cp_rank_0s_tokens(self):
        self.assertEqual(
            self.sync(CP_GROUP), [("broadcast", CP_LEADER, CP_GROUP, True)]
        )

    def test_tp_sync_runs_before_the_cp_broadcast(self):
        self.assertEqual(
            self.sync(CP_GROUP, sync_across_tp=True),
            [
                ("all_reduce", torch.distributed.ReduceOp.MIN, TP_GROUP, True),
                ("broadcast", CP_LEADER, CP_GROUP, True),
            ],
        )

    def test_without_cp_nothing_is_broadcast(self):
        self.assertEqual(self.sync(None), [])
        self.assertEqual(self.sync(CP_GROUP, cp_world_size=1), [])


if __name__ == "__main__":
    unittest.main()
