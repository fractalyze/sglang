import unittest
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    DSAFlashMLAMetadata,
    DSAMetadata,
    _restore_trtllm_decode_dp_padding,
    _trim_trtllm_decode_dp_padding,
)
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSABackendDPPadding(unittest.TestCase):
    def test_occupied_uneven_decode_with_idle_rank_selects_max_len(self):
        with (
            patch(
                "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
                return_value=MoeA2ABackend.NONE,
            ),
            patch(
                "sglang.srt.layers.dp_attention.dp_gather_width",
                return_value=4,
            ),
        ):
            mode = DpPaddingMode.get_dp_padding_mode(
                is_extend_in_batch=False,
                global_num_tokens=[8, 7, 6, 0],
            )
        self.assertEqual(mode, DpPaddingMode.MAX_LEN)

    def test_trim_and_restore_active_eager_decode_padding(self):
        q_all = torch.arange(8 * 2 * 3).view(8, 2, 3)
        topk_indices = torch.arange(8 * 4, dtype=torch.int32).view(8, 4)

        real_q, real_topk, num_padding_rows = _trim_trtllm_decode_dp_padding(
            q_all,
            topk_indices,
            real_batch_size=6,
        )

        self.assertTrue(torch.equal(real_q, q_all[:6]))
        self.assertTrue(torch.equal(real_topk, topk_indices[:6]))
        self.assertEqual(num_padding_rows, 2)

        real_output = torch.ones((6, 1, 2, 5), dtype=torch.bfloat16)
        output = _restore_trtllm_decode_dp_padding(real_output, num_padding_rows)
        self.assertEqual(output.shape, (8, 1, 2, 5))
        self.assertTrue(torch.equal(output[:6], real_output))
        self.assertTrue(torch.all(output[6:] == 0))

    def test_no_padding_preserves_existing_tensors(self):
        q_all = torch.empty((2, 2, 3))
        topk_indices = torch.empty((2, 8), dtype=torch.int32)

        real_q, real_topk, num_padding_rows = _trim_trtllm_decode_dp_padding(
            q_all,
            topk_indices,
            real_batch_size=2,
        )

        self.assertIs(real_q, q_all)
        self.assertIs(real_topk, topk_indices)
        self.assertEqual(num_padding_rows, 0)
        self.assertIs(
            _restore_trtllm_decode_dp_padding(real_q, num_padding_rows),
            real_q,
        )

    def test_rejects_metadata_larger_than_physical_batch(self):
        with self.assertRaisesRegex(
            AssertionError, "metadata batch size \\(3\\) exceeds q batch size \\(2\\)"
        ):
            _trim_trtllm_decode_dp_padding(
                torch.empty((2, 2, 3)),
                torch.empty((2, 8), dtype=torch.int32),
                real_batch_size=3,
            )

    def test_trtllm_decode_runs_real_rows_then_restores_physical_batch(self):
        metadata = SimpleNamespace(
            cache_seqlens_int32=torch.tensor([8, 12], dtype=torch.int32),
            page_table_1=torch.zeros((2, 12), dtype=torch.int32),
            max_seq_len_k=12,
        )
        backend = SimpleNamespace(
            forward_metadata=metadata,
            kv_cache_dtype=torch.bfloat16,
            token_to_kv_pool=SimpleNamespace(
                get_key_buffer=lambda _layer_id: torch.zeros((24, 3))
            ),
            real_page_size=1,
            kv_cache_dim=3,
            use_fused_topk=False,
            qk_nope_head_dim=2,
            kv_lora_rank=2,
            qk_rope_head_dim=1,
            workspace_buffer=None,
            dsa_index_topk=2,
            _multi_ctas_kv_counter_buffer=None,
            device="cpu",
            num_q_heads=2,
        )
        backend._pad_topk_indices = MethodType(
            DeepseekSparseAttnBackend._pad_topk_indices, backend
        )
        backend._pad_trtllm_sparse_page_table = MethodType(
            DeepseekSparseAttnBackend._pad_trtllm_sparse_page_table, backend
        )
        backend._multi_ctas_kv_counter_for = MethodType(
            DeepseekSparseAttnBackend._multi_ctas_kv_counter_for, backend
        )

        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=2,
            head_dim=3,
            k_scale_float=None,
            scaling=1.0,
        )
        forward_batch = SimpleNamespace()
        q = torch.arange(4 * 2 * 3, dtype=torch.float32).view(4, 2, 3)
        topk_indices = torch.arange(4 * 2, dtype=torch.int32).view(4, 2)

        flashinfer = ModuleType("flashinfer")
        flashinfer_decode = ModuleType("flashinfer.decode")
        captured = {}

        def fake_decode(**kwargs):
            captured.update(kwargs)
            return torch.ones((2, 1, 2, 2), dtype=torch.bfloat16)

        flashinfer_decode.trtllm_batch_decode_with_kv_cache_mla = fake_decode
        flashinfer.decode = flashinfer_decode

        def fake_transform(*, page_table, topk_indices, page_size):
            self.assertEqual(page_table.shape[0], 2)
            self.assertEqual(topk_indices.shape[0], 2)
            self.assertEqual(page_size, 1)
            return torch.zeros((2, 2), dtype=torch.int32)

        with (
            patch.dict(
                "sys.modules",
                {
                    "flashinfer": flashinfer,
                    "flashinfer.decode": flashinfer_decode,
                },
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend.transform_index_page_table_decode",
                side_effect=fake_transform,
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend.dsa_use_prefill_cp",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend.grow_multi_ctas_kv_counter_buffer_if_needed",
                return_value=None,
            ),
        ):
            output = DeepseekSparseAttnBackend._forward_trtllm(
                backend,
                q=q,
                k=torch.empty((4, 1, 2)),
                v=torch.empty((4, 1, 2)),
                layer=layer,
                forward_batch=forward_batch,
                seq_lens=metadata.cache_seqlens_int32,
                save_kv_cache=False,
                topk_indices=topk_indices,
            )

        self.assertEqual(captured["query"].shape, (2, 1, 2, 3))
        self.assertEqual(output.shape, (4, 1, 2, 2))
        self.assertTrue(torch.all(output[:2] == 1))
        self.assertTrue(torch.all(output[2:] == 0))

    def _flashmla_kv_backend(self, *, num_sm_parts=0, index_kpool=1):
        backend = SimpleNamespace(
            _flashmla_kv_q_row_fit=None,
            _flashmla_kv_num_sm_parts=num_sm_parts,
            flashmla_kv_num_q_heads=2,
            real_page_size=64,
            kv_cache_dim=3,
            dsa_kv_cache_store_fp8=True,
            dsa_index_topk=4,
            dsa_index_kpool=index_kpool,
        )
        for method in (
            "_flashmla_kv_topk_length",
            "_flashmla_kv_skips_padding",
            "_compute_flashmla_row_per_part_metadata",
        ):
            setattr(
                backend,
                method,
                MethodType(getattr(DeepseekSparseAttnBackend, method), backend),
            )
        metadata_rows = []

        def fake_compute_flashmla_metadata(*, cache_seqlens, seq_len_q):
            self.assertEqual(seq_len_q, 1)
            metadata_rows.append(cache_seqlens.shape[0])
            return DSAFlashMLAMetadata(
                flashmla_metadata=torch.empty(0),
                num_splits=torch.zeros(cache_seqlens.shape[0] + 1, dtype=torch.int32),
            )

        backend._compute_flashmla_metadata = fake_compute_flashmla_metadata
        backend._fit_flashmla_kv_metadata_to_q_rows = MethodType(
            DeepseekSparseAttnBackend._fit_flashmla_kv_metadata_to_q_rows, backend
        )
        return backend, metadata_rows

    @staticmethod
    def _flashmla_kv_metadata(dsa_cache_seqlens):
        empty = torch.empty(0, dtype=torch.int32)
        return DSAMetadata(
            page_size=64,
            cache_seqlens_int32=empty,
            max_seq_len_q=1,
            max_seq_len_k=1,
            cu_seqlens_q=empty,
            cu_seqlens_k=empty,
            page_table_1=None,
            real_page_table=empty,
            dsa_cache_seqlens_int32=dsa_cache_seqlens,
            dsa_cu_seqlens_q=empty,
            dsa_cu_seqlens_k=empty,
            dsa_extend_seq_lens_list=[],
            dsa_seqlens_expanded=empty,
            flashmla_metadata=DSAFlashMLAMetadata(
                flashmla_metadata=torch.empty(0),
                num_splits=torch.zeros(
                    dsa_cache_seqlens.shape[0] + 1, dtype=torch.int32
                ),
            ),
        )

    def _run_flashmla_kv_layers(self, *, dsa_cache_seqlens, num_q_rows, num_sm_parts=0):
        """Two layers of one forward through _forward_flashmla_kv with q of
        num_q_rows; returns the seqlens FlashMLA got and the metadata rebuilds."""
        backend, metadata_rows = self._flashmla_kv_backend(num_sm_parts=num_sm_parts)
        metadata = self._flashmla_kv_metadata(dsa_cache_seqlens)
        layer = SimpleNamespace(tp_q_head_num=2, head_dim=3)

        flash_mla = ModuleType("sgl_kernel.flash_mla")
        kernel_calls = []

        def fake_flash_mla_with_kvcache(
            *, q, cache_seqlens, tile_scheduler_metadata, num_splits=None, **kwargs
        ):
            if num_q_rows <= num_sm_parts:
                # The schedule rides in the sched-meta object, beside the
                # topk_length it was built from.
                self.assertIsNone(num_splits)
                num_splits = tile_scheduler_metadata.num_splits
                self.assertIs(kwargs["topk_length"], cache_seqlens)
            else:
                self.assertNotIn("topk_length", kwargs)
            # The check that raised "num_splits must have shape (b+1)".
            self.assertEqual(num_splits.shape[0], q.shape[0] + 1)
            self.assertEqual(cache_seqlens.shape[0], q.shape[0])
            kernel_calls.append((cache_seqlens, num_splits))
            return torch.zeros((q.shape[0], 1, 2, 2)), None

        flash_mla.FlashMLASchedMeta = SimpleNamespace
        flash_mla.flash_mla_with_kvcache = fake_flash_mla_with_kvcache
        with (
            patch.dict("sys.modules", {"sgl_kernel.flash_mla": flash_mla}),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        ):
            for _ in range(2):
                output = DeepseekSparseAttnBackend._forward_flashmla_kv(
                    backend,
                    q_all=torch.zeros((num_q_rows, 2, 3)),
                    kv_cache=torch.zeros((64, 3)),
                    v_head_dim=2,
                    sm_scale=1.0,
                    layer=layer,
                    metadata=metadata,
                    page_table_1=torch.zeros((num_q_rows, 4), dtype=torch.int32),
                )

        self.assertEqual(output.shape, (num_q_rows, 1, 2, 2))
        # Rebuilt once per forward, not per layer.
        self.assertIs(kernel_calls[0][1], kernel_calls[1][1])
        # The shared forward metadata the indexer reads is left untouched.
        self.assertIs(metadata.dsa_cache_seqlens_int32, dsa_cache_seqlens)
        return kernel_calls[0][0], metadata_rows

    def test_flashmla_kv_fits_metadata_to_narrowed_prefill_graph_q(self):
        """Prefill-graph replay narrows q to the rank's real tokens, fewer than
        the DP-padded seqlens; FlashMLA must still get q rows + 1 splits."""
        cache_seqlens, metadata_rows = self._run_flashmla_kv_layers(
            dsa_cache_seqlens=torch.tensor([1, 2, 3, 4, 0, 0, 0, 0], dtype=torch.int32),
            num_q_rows=4,
        )
        self.assertEqual(metadata_rows, [4])
        self.assertEqual(cache_seqlens.tolist(), [1, 2, 3, 4])

    def test_flashmla_kv_pads_metadata_to_longer_q_with_empty_rows(self):
        cache_seqlens, metadata_rows = self._run_flashmla_kv_layers(
            dsa_cache_seqlens=torch.tensor([1, 2, 3, 4, 0, 0], dtype=torch.int32),
            num_q_rows=8,
        )
        self.assertEqual(metadata_rows, [8])
        self.assertEqual(cache_seqlens.tolist(), [1, 2, 3, 4, 0, 0, 0, 0])

    def test_flashmla_kv_skips_padding_when_each_row_has_a_part(self):
        """With a SM part per q row, FlashMLA gets each row's valid top-k length
        beside the schedule."""
        cache_seqlens, metadata_rows = self._run_flashmla_kv_layers(
            dsa_cache_seqlens=torch.tensor([1, 2, 3, 4, 0, 0], dtype=torch.int32),
            num_q_rows=8,
            num_sm_parts=8,
        )
        self.assertEqual(metadata_rows, [8])
        self.assertEqual(cache_seqlens.tolist(), [1, 2, 3, 4, 0, 0, 0, 0])

    def test_flashmla_kv_keeps_full_topk_with_more_rows_than_parts(self):
        cache_seqlens, _ = self._run_flashmla_kv_layers(
            dsa_cache_seqlens=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
            num_q_rows=4,
            num_sm_parts=3,
        )
        self.assertEqual(cache_seqlens.tolist(), [1, 2, 3, 4])

    def test_flashmla_kv_row_per_part_schedule(self):
        backend, _ = self._flashmla_kv_backend(num_sm_parts=5)
        metadata = backend._compute_flashmla_row_per_part_metadata(
            torch.tensor([0, 64, 65, 4], dtype=torch.int32)
        )
        # begin_req, end_req, begin_block, end_block, then split fields and pad.
        self.assertEqual(
            metadata.flashmla_metadata.tolist(),
            [
                [0, 0, 0, 1, 0, 0, 0, 0],
                [1, 1, 0, 1, 0, 0, 0, 0],
                [2, 2, 0, 2, 0, 0, 0, 0],
                [3, 3, 0, 1, 0, 0, 0, 0],
                [4, 3, 0, 0, 0, 0, 0, 0],
            ],
        )
        self.assertEqual(metadata.num_splits.tolist(), [0, 1, 2, 3, 4])

    def test_flashmla_kv_topk_length_clamps_pooled_tail_to_topk(self):
        seqlens = torch.tensor([2, 4, 6], dtype=torch.int32)
        backend, _ = self._flashmla_kv_backend()
        self.assertIs(backend._flashmla_kv_topk_length(seqlens), seqlens)
        pooled, _ = self._flashmla_kv_backend(index_kpool=4)
        self.assertEqual(pooled._flashmla_kv_topk_length(seqlens).tolist(), [2, 4, 4])

    def test_flashmla_kv_metadata_matching_q_rows_is_unchanged(self):
        backend, metadata_rows = self._flashmla_kv_backend()
        metadata = self._flashmla_kv_metadata(torch.ones(4, dtype=torch.int32))

        self.assertIs(
            backend._fit_flashmla_kv_metadata_to_q_rows(
                metadata=metadata, num_q_rows=4
            ),
            metadata,
        )
        self.assertEqual(metadata_rows, [])

    def test_flashmla_kv_metadata_mismatch_rejected_under_graph_capture(self):
        backend, _ = self._flashmla_kv_backend()
        metadata = self._flashmla_kv_metadata(torch.ones(4, dtype=torch.int32))

        with (
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=True),
            self.assertRaisesRegex(AssertionError, "CUDA graph capture"),
        ):
            backend._fit_flashmla_kv_metadata_to_q_rows(metadata=metadata, num_q_rows=8)


if __name__ == "__main__":
    unittest.main()
