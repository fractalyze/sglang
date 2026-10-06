"""Triton decode replay on a static SWA pool, SGLANG_OPT_SWA_DECODE_NO_HOST_SYNC.

``update_sliding_window_buffer`` fills the CUDA-graph window-KV buffer with full-pool ids and, for a
static SWA pool, maps them to SWA ids. The default path slices the buffer by the GPU scalar
``window_kv_indptr[-1]``, which makes the host wait for the previous forward on every decode replay.
The switch translates over a host-side bound (``bs * sliding_window_size``) with a device-side mask
instead. These tests pin that the switch writes the same ids as the default path, leaves the stale
tail of the buffer alone, and never turns a tensor into a host value.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.triton_backend import update_sliding_window_buffer
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

WINDOW = 8
FULL_POOL = 200
MAX_BS = 4


class _Translator:
    """Writes each request's last min(seq_len, window) full ids from a fixed req_to_token table."""

    reads_are_translated = False

    def __init__(self, req_to_token: torch.Tensor):
        self.req_to_token = req_to_token

    def fill_packed_read_stream(self, *, req_pool_indices, seq_lens, indptr, total_tokens, out,
                                kv_start_idx=None, sliding_window=False):
        for i in range(seq_lens.numel()):
            start = int(kv_start_idx[i])
            ids = self.req_to_token[int(req_pool_indices[i]), start : start + int(seq_lens[i])]
            out[int(indptr[i]) : int(indptr[i]) + ids.numel()] = ids
        return False


def _pool(mapping: torch.Tensor):
    pool = mock.MagicMock(spec=BaseSWAKVPool)
    pool.translate_loc_from_full_to_swa.side_effect = lambda idx: mapping[idx]
    return pool


def _inputs(seed: int):
    g = torch.Generator().manual_seed(seed)
    req_to_token = torch.randint(1, FULL_POOL, (MAX_BS + 1, 64), generator=g)
    mapping = torch.randint(0, 50, (FULL_POOL + 1,), generator=g)
    seq_lens = torch.tensor([3, 20, 8, 1])[: MAX_BS]
    req_pool_indices = torch.tensor([2, 0, 4, 1])
    return req_to_token, mapping, seq_lens, req_pool_indices


def _run(no_sync: bool, seed: int, stale: int):
    req_to_token, mapping, seq_lens, req_pool_indices = _inputs(seed)
    indptr = torch.zeros(MAX_BS + 1, dtype=torch.int64)
    buf = torch.full((MAX_BS * WINDOW,), stale, dtype=torch.int64)
    with envs.SGLANG_OPT_SWA_DECODE_NO_HOST_SYNC.override(no_sync):
        indptr_out, indices, lens, _ = update_sliding_window_buffer(
            indptr, _Translator(req_to_token), req_pool_indices, WINDOW, seq_lens, MAX_BS,
            token_to_kv_pool=_pool(mapping), window_kv_indices=buf,
        )
    return int(indptr_out[-1]), indices, lens


class TestSWAWindowNoHostSync(CustomTestCase):
    def test_same_ids_as_the_synced_path(self):
        for seed in range(5):
            n_ref, ref, _ = _run(False, seed, stale=7)
            n, got, _ = _run(True, seed, stale=7)
            self.assertEqual(n, n_ref)
            self.assertTrue(torch.equal(got[:n], ref[:n]))

    def test_stale_tail_is_left_alone(self):
        n, got, lens = _run(True, 0, stale=7)
        self.assertEqual(n, int(lens.sum()))
        self.assertLess(n, got.numel())
        self.assertTrue(torch.all(got[n:] == 7))

    def test_no_tensor_becomes_a_host_value(self):
        def boom(*args, **kwargs):
            raise AssertionError("host sync: a tensor was read on the host")

        req_to_token, mapping, seq_lens, req_pool_indices = _inputs(0)
        translator, pool = _Translator(req_to_token), _pool(mapping)
        # The test translator reads ids on the host; only the code under test is watched.
        with mock.patch.object(translator, "fill_packed_read_stream", return_value=False):
            for no_sync in (True, False):
                indptr = torch.zeros(MAX_BS + 1, dtype=torch.int64)
                buf = torch.zeros(MAX_BS * WINDOW, dtype=torch.int64)
                with envs.SGLANG_OPT_SWA_DECODE_NO_HOST_SYNC.override(no_sync), mock.patch.object(
                    torch.Tensor, "__index__", boom
                ), mock.patch.object(torch.Tensor, "item", boom):
                    call = lambda: update_sliding_window_buffer(
                        indptr, translator, req_pool_indices, WINDOW, seq_lens, MAX_BS,
                        token_to_kv_pool=pool, window_kv_indices=buf,
                    )
                    if no_sync:
                        call()
                    else:
                        # The default path slices by a GPU scalar: the guard must catch it.
                        with self.assertRaises(AssertionError):
                            call()


if __name__ == "__main__":
    unittest.main()
