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


def _backend():
    """The flashmla_kv pieces of DeepseekSparseAttnBackend on this GPU."""
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    backend = SimpleNamespace(
        _flashmla_kv_num_sm_parts=num_sms // (H_Q // 64),
        dsa_index_topk=TOPK,
        dsa_index_kpool=1,
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
    return backend


def _decode(q, kv, indices, topk_length=None):
    """Sparse decode: today's full-top-k path with no `topk_length`, else the
    backend's row-per-part schedule with it."""
    import sgl_kernel.flash_mla as flash_mla

    b = q.shape[0]
    sched = flash_mla.FlashMLASchedMeta()
    if topk_length is not None:
        backend = _backend()
        assert backend._flashmla_kv_skips_padding(num_rows=b)
        metadata = backend._compute_flashmla_row_per_part_metadata(topk_length)
        sched.tile_scheduler_metadata = metadata.flashmla_metadata
        sched.num_splits = metadata.num_splits
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
def test_padding_skip_is_bitwise_full_topk(b, mode):
    """Stopping each row at its valid length gives exactly today's output: the
    skipped blocks hold only -1 slots, which the full run masks out."""
    lengths = _lengths(b, mode)
    q, kv, indices = _inputs(lengths)
    out = _decode(q, kv, indices, lengths)
    full = _decode(q, kv, indices)
    valid = lengths > 0
    assert torch.equal(out[valid], full[valid])


def test_padding_skip_is_the_truncated_top_k():
    """A row runs exactly the blocks of a top-k row cut to its length."""
    lengths = _lengths(64, "short")
    q, kv, indices = _inputs(lengths)
    cut = (1242 + PAGE - 1) // PAGE * PAGE
    truncated = indices[..., :cut].contiguous()
    assert torch.equal(_decode(q, kv, indices, lengths), _decode(q, kv, truncated))


def test_padding_skip_is_deterministic_and_graph_safe():
    lengths = _lengths(64, "mixed")
    q, kv, indices = _inputs(lengths)
    eager = _decode(q, kv, indices, lengths)
    assert torch.equal(eager, _decode(q, kv, indices, lengths))

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _decode(q, kv, indices, lengths)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        replayed = _decode(q, kv, indices, lengths)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(replayed, eager)
