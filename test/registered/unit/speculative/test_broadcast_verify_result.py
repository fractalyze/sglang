"""Every rank of a DP shard keeps its first rank's EAGLE drafts and verify result.

Attention CP splits only prefill. Under attention DP the CP ranks of a DP
shard hold the same requests and the same replicated KV, and each drafts,
verifies and decodes by itself on logits that are not bit-identical. The
ranks can then accept different drafts, their committed lengths diverge, and
the next draft batch has a different row count per rank. The first rank's
drafts and verify result are shared along the attention-CP group, then along
each attention-TP group.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers import dp_attention
from sglang.srt.speculative import eagle_utils, eagle_worker_common
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def parallel(attn_dp_enabled, attn_cp_size, attn_cp_group=None):
    return SimpleNamespace(
        attn_dp_enabled=attn_dp_enabled,
        attn_cp_size=attn_cp_size,
        attn_cp_group=attn_cp_group,
    )


class TestCpReplicasNeedSync(CustomTestCase):
    def test_only_attention_dp_with_cp_needs_sync(self):
        for dp, cp, expected in [
            (True, 2, True),
            (True, 1, False),
            (False, 2, False),
            (False, 1, False),
        ]:
            with (
                self.subTest(attn_dp_enabled=dp, attn_cp_size=cp),
                patch(
                    "sglang.srt.runtime_context.get_parallel",
                    return_value=parallel(dp, cp),
                ),
            ):
                self.assertEqual(eagle_utils._cp_replicas_need_sync(), expected)


class TestBroadcastFromShardLeader(CustomTestCase):
    def run_shard(self, cp_size, tp_size):
        # Global rank of (cp, tp) within one DP shard; each rank starts with
        # its own accept counts.
        rank_of = {
            (cp, tp): cp * tp_size + tp
            for cp in range(cp_size)
            for tp in range(tp_size)
        }
        held = {rank: torch.tensor([rank, rank + 10]) for rank in rank_of.values()}
        sent = {}

        def make_group(name, key, members, rank_in_group):
            def broadcast(tensor, src=0):
                # Stand-in for the collective: the source sends what it holds
                # and every other member overwrites its copy in place.
                if rank_in_group == src:
                    sent[(name, key)] = tensor.clone()
                else:
                    self.assertIn((name, key), sent, "a receiver ran before its source")
                    tensor.copy_(sent[(name, key)])
                return tensor

            return SimpleNamespace(
                world_size=len(members),
                rank_in_group=rank_in_group,
                broadcast=broadcast,
            )

        # Collectives order the ranks for real; here each source runs before
        # the ranks it sends to: attention-TP leaders along CP first, then
        # each attention-TP group.
        order = [(cp, 0) for cp in range(cp_size)]
        order += [(cp, tp) for cp in range(cp_size) for tp in range(1, tp_size)]
        for cp, tp in order:
            rank = rank_of[(cp, tp)]
            tp_group = make_group(
                "attn_tp", cp, [rank_of[(cp, t)] for t in range(tp_size)], tp
            )
            cp_group = make_group(
                "attn_cp", tp, [rank_of[(c, tp)] for c in range(cp_size)], cp
            )
            with (
                patch.object(dp_attention, "get_dp_tp_group", return_value=tp_group),
                patch(
                    "sglang.srt.runtime_context.get_parallel",
                    return_value=parallel(True, cp_size, cp_group),
                ),
            ):
                eagle_utils._broadcast_from_shard_leader(held[rank])
        return held

    def test_every_rank_of_the_shard_ends_with_the_first_ranks_copy(self):
        for cp_size, tp_size in [(2, 1), (2, 2), (4, 2), (1, 2), (1, 1)]:
            with self.subTest(cp_size=cp_size, tp_size=tp_size):
                held = self.run_shard(cp_size, tp_size)
                for rank, tensor in held.items():
                    self.assertEqual(tensor.tolist(), [0, 10], f"rank {rank}")

    def test_without_attention_dp_only_the_tp_group_broadcasts(self):
        # There the CP group is the TP group itself: a CP leg on its leader
        # alone would desynchronize the shared communicator.
        cp_group = SimpleNamespace(
            world_size=4,
            rank_in_group=0,
            broadcast=Mock(side_effect=AssertionError("CP leg without attention DP")),
        )
        tp_group = SimpleNamespace(world_size=4, rank_in_group=0, broadcast=Mock())
        tensor = torch.tensor([1, 2])
        with (
            patch.object(dp_attention, "get_dp_tp_group", return_value=tp_group),
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=parallel(False, 4, cp_group),
            ),
        ):
            eagle_utils._broadcast_from_shard_leader(tensor)
        tp_group.broadcast.assert_called_once_with(tensor, src=0)


class TestGreedyVerifySync(CustomTestCase):
    def verify_greedy(self, need_sync):
        def fake_greedy(**kwargs):
            kwargs["predicts"].fill_(3)
            kwargs["accept_index"].fill_(0)
            kwargs["accept_token_num"].fill_(1)
            return (
                kwargs["predicts"],
                kwargs["accept_index"],
                kwargs["accept_token_num"],
            )

        verify_input = SimpleNamespace(
            draft_token_num=2,
            draft_token=torch.tensor([1, 2], dtype=torch.int32),
            max_tree_depth=2,
            tree_topk=1,
            retrieve_index=torch.zeros((1, 2), dtype=torch.int32),
            retrieve_next_token=torch.zeros((1, 2), dtype=torch.int32),
            retrieve_next_sibling=torch.zeros((1, 2), dtype=torch.int32),
            draft_probs=None,
        )
        batch = SimpleNamespace(
            device="cpu",
            seq_lens=torch.tensor([4], dtype=torch.int32),
            sampling_info=SimpleNamespace(
                acc_additive_penalties=None,
                acc_scaling_penalties=None,
                logit_bias=None,
                is_all_greedy=True,
            ),
            forward_mode=SimpleNamespace(is_idle=lambda: False),
        )
        logits_output = SimpleNamespace(next_token_logits=torch.randn((2, 8)))
        spec_config = SimpleNamespace(speculative_use_rejection_sampling=False)
        share = Mock()
        with (
            patch.object(eagle_utils, "_is_hip", False),
            patch.object(eagle_utils, "get_spec", return_value=spec_config),
            patch.object(
                eagle_utils, "verify_tree_greedy_func", side_effect=fake_greedy
            ),
            patch.object(eagle_utils, "_cp_replicas_need_sync", return_value=need_sync),
            patch.object(eagle_utils, "_broadcast_from_shard_leader", share),
        ):
            eagle_utils.eagle_sample(verify_input, batch, logits_output)
        return share

    def test_cp_replicas_share_the_greedy_result(self):
        share = self.verify_greedy(need_sync=True)
        share.assert_called_once()
        predict, accept_index, num_correct_drafts = share.call_args.args
        self.assertEqual(accept_index.shape, (1, 2))
        self.assertEqual(num_correct_drafts.shape, (1,))

    def test_greedy_result_is_not_shared_without_cp(self):
        self.verify_greedy(need_sync=False).assert_not_called()


class TestBroadcastDraftTree(CustomTestCase):
    def share(self, need_sync, topk, num_steps, num_draft_tokens, tree):
        shared = []

        def fake_share(*tensors):
            shared.append(tensors)
            for t in tensors:
                t.fill_(7)  # stand-in for the shard leader's values

        with (
            patch.object(eagle_utils, "_cp_replicas_need_sync", return_value=need_sync),
            patch.object(
                eagle_utils, "_broadcast_from_shard_leader", side_effect=fake_share
            ),
        ):
            out = eagle_utils.broadcast_draft_tree(
                *tree,
                topk=topk,
                num_steps=num_steps,
                num_draft_tokens=num_draft_tokens,
            )
        return out, shared

    @staticmethod
    def tree(num_drafted=3, parents=2):
        """(parent_list, top_scores_index, draft_tokens) of one request."""
        return (
            torch.zeros((1, parents), dtype=torch.int64),
            torch.arange(num_drafted).view(1, -1),
            torch.arange(num_drafted).view(1, -1) + 10,
        )

    def test_without_cp_the_drafts_are_untouched(self):
        tree = self.tree()
        out, shared = self.share(False, 1, 3, 4, tree)
        self.assertEqual(shared, [])
        for got, given in zip(out, tree):
            self.assertIs(got, given)

    def test_a_full_topk1_chain_shares_only_its_tokens(self):
        tree = self.tree()
        out, shared = self.share(True, 1, 3, 4, tree)
        self.assertEqual(len(shared), 1)
        (only,) = shared[0]
        self.assertIs(only, tree[2])
        self.assertEqual(out[2].tolist(), [[7, 7, 7]])
        self.assertEqual(out[1].tolist(), [[0, 1, 2]])

    def test_a_tree_or_a_cut_chain_also_shares_its_topology(self):
        for topk, num_steps, num_draft_tokens in [(2, 3, 4), (1, 3, 3)]:
            with self.subTest(topk=topk, num_draft_tokens=num_draft_tokens):
                tree = self.tree(num_drafted=num_draft_tokens - 1)
                out, shared = self.share(True, topk, num_steps, num_draft_tokens, tree)
                self.assertEqual(len(shared), 1)
                self.assertEqual(
                    [t.data_ptr() for t in shared[0]],
                    [tree[2].data_ptr(), tree[0].data_ptr(), tree[1].data_ptr()],
                )
                for t in out:
                    self.assertTrue(torch.all(t == 7))

    def test_an_empty_parent_list_is_not_sent(self):
        tree = self.tree(parents=0)
        out, shared = self.share(True, 2, 1, 4, tree)
        self.assertEqual(
            [t.data_ptr() for t in shared[0]], [tree[2].data_ptr(), tree[1].data_ptr()]
        )

    def test_a_non_contiguous_draft_is_shared_as_a_copy(self):
        draft_tokens = torch.arange(6).view(3, 2).t()
        self.assertFalse(draft_tokens.is_contiguous())
        tree = (
            torch.zeros((2, 2), dtype=torch.int64),
            torch.zeros((2, 3), dtype=torch.int64),
            draft_tokens,
        )
        out, _ = self.share(True, 1, 3, 4, tree)
        self.assertTrue(out[2].is_contiguous())
        self.assertTrue(torch.all(out[2] == 7))


class TestVerifyInputUsesSharedDrafts(CustomTestCase):
    def test_the_tree_is_built_from_the_shared_drafts(self):
        shared = (
            torch.zeros((1, 2), dtype=torch.int64),
            torch.arange(3).view(1, 3),
            torch.tensor([[21, 22, 23]]),
        )
        built = {}

        def fake_tree(_bonus, parent_list, top_scores_index, draft_tokens, *_, **__):
            built.update(
                parent_list=parent_list,
                top_scores_index=top_scores_index,
                draft_tokens=draft_tokens,
            )
            return None, None, None, None, None, draft_tokens.flatten()

        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            seq_lens=torch.tensor([4]),
            seq_lens_sum=4,
        )
        target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                attn_backend=SimpleNamespace(verify_mask=None, max_context_len=16)
            )
        )
        with (
            patch.object(
                eagle_worker_common, "broadcast_draft_tree", return_value=shared
            ) as share,
            patch.object(
                eagle_worker_common,
                "build_tree_kernel_efficient",
                side_effect=fake_tree,
            ),
            patch.object(
                eagle_worker_common,
                "EagleVerifyInput",
                side_effect=lambda **kw: SimpleNamespace(**kw),
            ),
        ):
            verify_input = eagle_worker_common.build_eagle_verify_input(
                batch,
                SimpleNamespace(bonus_tokens=torch.tensor([5])),
                torch.zeros((1, 2), dtype=torch.int64),
                torch.zeros((1, 3), dtype=torch.int64),
                torch.tensor([[1, 2, 3]]),
                None,
                target_worker=target_worker,
                topk=1,
                num_steps=3,
                num_draft_tokens=4,
                tree_mask_mode=eagle_utils.TreeMaskMode.FULL_MASK,
                device="cpu",
            )

        share.assert_called_once()
        self.assertEqual(
            share.call_args.kwargs, dict(topk=1, num_steps=3, num_draft_tokens=4)
        )
        self.assertIs(built["parent_list"], shared[0])
        self.assertIs(built["top_scores_index"], shared[1])
        self.assertIs(built["draft_tokens"], shared[2])
        self.assertEqual(verify_input.draft_token.tolist(), [21, 22, 23])


if __name__ == "__main__":
    unittest.main()
