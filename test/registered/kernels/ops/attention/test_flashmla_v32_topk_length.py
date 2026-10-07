from types import MethodType, SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or torch.cuda.get_device_capability()[0] != 9,
    reason="the V3.2 sparse decode topk_length path is the SM90 kernel's",
)

# DeepSeek-V3.2 decode on one DP-attention rank: 128 q heads, top-k 2048, page 64,
# the 656-byte fp8 KV token.
H_Q, D_QK, D_V, TOPK, PAGE = 128, 576, 512, 2048, 64
NUM_TOKENS = 64 * 1024
BLOCK_SIZE_N = 64
# FlashMLA's DecodingSchedMeta ends in a `_pad` word it never writes.
DEFINED = slice(0, 7)


def _inputs(lengths, seed=0):
    from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache

    g = torch.Generator(device="cuda").manual_seed(seed)
    b = lengths.shape[0]
    kv = torch.randn(
        (NUM_TOKENS // PAGE, PAGE, 1, D_QK),
        device="cuda",
        dtype=torch.bfloat16,
        generator=g,
    )
    q = torch.randn((b, 1, H_Q, D_QK), device="cuda", dtype=torch.bfloat16, generator=g)
    # Valid-first top-k rows, as the DSA indexer emits them: `length` distinct
    # token slots, then -1 padding.
    indices = torch.full((b, 1, TOPK), -1, device="cuda", dtype=torch.int32)
    for i, length in enumerate(lengths.tolist()):
        slots = torch.randperm(NUM_TOKENS, device="cuda", generator=g)[:length]
        indices[i, 0, :length] = slots.int()
    return q, quantize_k_cache(kv), indices


def _backend_schedule(topk_length):
    """The schedule DeepseekSparseAttnBackend builds for flashmla_kv."""
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

    backend = SimpleNamespace(
        _num_sms=torch.cuda.get_device_properties(0).multi_processor_count,
        flashmla_kv_num_q_heads=H_Q,
        dsa_index_topk=TOPK,
        dsa_index_kpool=1,
    )
    backend._flashmla_kv_topk_length = MethodType(
        DeepseekSparseAttnBackend._flashmla_kv_topk_length, backend
    )
    metadata = DeepseekSparseAttnBackend._compute_flashmla_topk_length_metadata(
        backend, topk_length, seq_len_q=1
    )
    return metadata.flashmla_metadata, metadata.num_splits


def _decode(q, kv, indices, topk_length, sched=None):
    """Sparse decode; with no `sched`, FlashMLA schedules the call itself."""
    import sgl_kernel.flash_mla as flash_mla

    b = q.shape[0]
    if sched is None:
        sched = flash_mla.FlashMLASchedMeta()
    out, _ = flash_mla.flash_mla_with_kvcache(
        q,
        kv,
        torch.empty((b, 0), device="cuda", dtype=torch.int32),
        None,
        D_V,
        sched,
        softmax_scale=D_QK**-0.5,
        indices=indices,
        topk_length=topk_length,
        is_fp8_kvcache=True,
    )
    return out


def _backend_decode(q, kv, indices, topk_length):
    import sgl_kernel.flash_mla as flash_mla

    meta, splits = _backend_schedule(topk_length)
    sched = flash_mla.FlashMLASchedMeta(tile_scheduler_metadata=meta, num_splits=splits)
    return _decode(q, kv, indices, topk_length, sched)


def _lengths(b, mode):
    if mode == "full":
        return torch.full((b,), TOPK, device="cuda", dtype=torch.int32)
    if mode == "short":
        return torch.full((b,), 1242, device="cuda", dtype=torch.int32)
    # Every block-boundary case, plus the zero-length rows DP padding adds.
    pattern = [0, 1, 63, 64, 65, 1242, TOPK - 1, TOPK]
    return torch.tensor(
        [pattern[i % len(pattern)] for i in range(b)], device="cuda", dtype=torch.int32
    )


@pytest.mark.parametrize("b", [8, 64])
@pytest.mark.parametrize("mode", ["full", "short", "mixed"])
def test_topk_length_matches_full_topk(b, mode):
    """Stopping a row at its valid length gives the full-top-k row's output: the
    skipped slots are -1 padding the full run masks out."""
    lengths = _lengths(b, mode)
    q, kv, indices = _inputs(lengths)
    out = _backend_decode(q, kv, indices, lengths)
    full = _decode(q, kv, indices, None)
    valid = lengths > 0
    torch.testing.assert_close(out[valid], full[valid], atol=2e-3, rtol=2e-3)


def test_topk_length_is_the_truncated_top_k():
    """Rows that share a length run exactly the blocks of top-k rows cut to that
    length, so the outputs are bitwise equal."""
    lengths = _lengths(64, "short")
    q, kv, indices = _inputs(lengths)
    cut = (1242 + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N * BLOCK_SIZE_N
    truncated = indices[..., :cut].contiguous()
    assert torch.equal(
        _backend_decode(q, kv, indices, lengths), _decode(q, kv, truncated, None)
    )


@pytest.mark.parametrize("mode", ["full", "short", "mixed"])
def test_backend_schedule_is_flashmla_own(mode):
    """The backend's schedule is the one FlashMLA computes for itself from the
    same topk_length, so its shape and split counts match the kernel's."""
    import sgl_kernel.flash_mla as flash_mla

    lengths = _lengths(64, mode)
    q, kv, indices = _inputs(lengths)
    own = flash_mla.FlashMLASchedMeta()
    _decode(q, kv, indices, lengths, own)
    meta, splits = _backend_schedule(lengths)
    assert meta.shape == own.tile_scheduler_metadata.shape
    assert torch.equal(meta[:, DEFINED], own.tile_scheduler_metadata[:, DEFINED])
    assert torch.equal(splits, own.num_splits)


def test_topk_length_is_deterministic_and_graph_safe():
    lengths = _lengths(64, "mixed")
    q, kv, indices = _inputs(lengths)
    eager = _backend_decode(q, kv, indices, lengths)
    assert torch.equal(eager, _backend_decode(q, kv, indices, lengths))

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _backend_decode(q, kv, indices, lengths)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        replayed = _backend_decode(q, kv, indices, lengths)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(replayed, eager)
