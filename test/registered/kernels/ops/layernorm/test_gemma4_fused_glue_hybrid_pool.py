"""Byte-level test of the Gemma4 fused KV store through a hybrid SWA pool.

``SGLANG_OPT_GEMMA4_FUSED_GLUE >= 1`` makes ``Gemma4Attention`` normalize q/k/v,
apply RoPE and store K/V in one kernel at the rows ``_fused_kv_write_target``
picks. The kernel itself is tested against a standalone pool in
``test_gemma4_fused_qkv_rope_kv.py``. This test covers the routing on the pool a
served Gemma-4 actually has: ``SWAKVPool`` with FP8 E4M3 sub-pools of different
geometry (sliding 8 kv heads x 256, full 2 kv heads x 512), sliding layers
written at the attention backend's ``swa_out_cache_loc`` and full layers at
``out_cache_loc``.

The reference is what ``TritonAttnBackend.forward_decode`` does unfused:
``gemma_qkv_rmsnorm`` -> RoPE -> ``SWAKVPool.set_kv_buffer`` with a
``KVWriteLoc`` and the layer's KV scales. Every byte of both sub-pools must match,
so a write to the wrong sub-pool, layer, slot or head fails as well as a wrong
value. Full layers project V with K's weights (``attention_k_eq_v``), so their v
input equals their k input here.

Requires a CUDA GPU with FP8 support; skips otherwise.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="FP8 E4M3 KV cache needs a CUDA GPU with SM 8.9+",
)

_FULL_SLOTS = 4096
# Smaller than the full pool, as served, but above the largest M so every token gets its own SWA slot.
_SWA_SLOTS = 3072
# Served with max_position_embeddings 16384, so positions reach the last cache row.
_MAX_POS = 16384
_EPS = 1e-6
_HQ = 16
# Gemma-4-26B-A4B's pattern over one period: five sliding layers, then a full one.
_SWA_LAYERS = [0, 1, 2, 3, 4]
_FULL_LAYERS = [5]
_GEOMETRY = {
    # layer kind: (kv heads, head_dim, (rope kind, base, rotated fraction))
    "sliding": (8, 256, ("default", 10000.0, 1.0)),
    "full": (2, 512, ("proportional", 1000000.0, 0.25)),
}


def _rope(head_dim: int, rope):
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
    from sglang.srt.layers.rotary_embedding.rope_variant import Gemma4RotaryEmbedding

    kind, base, partial = rope
    if kind == "proportional":
        emb = Gemma4RotaryEmbedding(
            head_dim, int(head_dim * partial), _MAX_POS, base, True, torch.bfloat16
        )
    else:
        emb = RotaryEmbedding(head_dim, head_dim, _MAX_POS, base, True, torch.bfloat16)
    return emb.to("cuda")


def _pool(dtype=torch.float8_e4m3fn):
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

    full_kv, full_d, _ = _GEOMETRY["full"]
    swa_kv, swa_d, _ = _GEOMETRY["sliding"]
    return SWAKVPool(
        size=_FULL_SLOTS,
        size_swa=_SWA_SLOTS,
        page_size=1,
        dtype=dtype,
        head_num=full_kv,
        head_dim=full_d,
        swa_attention_layer_ids=_SWA_LAYERS,
        full_attention_layer_ids=_FULL_LAYERS,
        device="cuda",
        enable_alt_stream=False,
        swa_head_num=swa_kv,
        swa_head_dim=swa_d,
        swa_v_head_dim=swa_d,
        v_head_dim=full_d,
    )


def _triton_backend(swa_loc, full_physical=None):
    """A TritonAttnBackend carrying only the per-forward metadata the write target reads."""
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

    backend = TritonAttnBackend.__new__(TritonAttnBackend)
    backend.dcp_size = 1
    backend.forward_metadata = SimpleNamespace(
        swa_out_cache_loc=swa_loc, out_cache_loc_full_physical=full_physical
    )
    return backend


def _inputs(M, hkv, d, k_eq_v, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    qkv = torch.randn(M, (_HQ + 2 * hkv) * d, device="cuda", generator=g) * 3.0
    qkv = qkv.to(torch.bfloat16)
    if k_eq_v:
        qkv[:, (_HQ + hkv) * d :] = qkv[:, _HQ * d : (_HQ + hkv) * d]
    q_w = (1.0 + 0.5 * torch.randn(d, device="cuda", generator=g)).to(torch.bfloat16)
    k_w = (1.0 + 0.5 * torch.randn(d, device="cuda", generator=g)).to(torch.bfloat16)
    pos = torch.randint(0, _MAX_POS, (M,), device="cuda", generator=g)
    pos[0] = _MAX_POS - 1
    # Distinct slots in each sub-pool: the request's full-pool rows and their SWA translation. Slot 0 is
    # the CUDA-graph padding slot, never allocated; the backend's store skips it and the fused kernel does not.
    loc = torch.randperm(_FULL_SLOTS - 1, device="cuda", generator=g)[:M] + 1
    swa_loc = torch.randperm(_SWA_SLOTS - 1, device="cuda", generator=g)[:M] + 1
    assert loc.numel() == swa_loc.numel() == M
    return qkv, q_w, k_w, pos, loc, swa_loc


def _split(qkv, hkv, d):
    return qkv.split([_HQ * d, hkv * d, hkv * d], dim=-1)


def _unfused_store(
    pool, layer_id, qkv, q_w, k_w, pos, loc, swa_loc, hkv, d, rope, scale
):
    from sglang.kernels.ops.layernorm.gemma4_fused_ops import gemma_qkv_rmsnorm
    from sglang.srt.mem_cache.memory_pool import KVWriteLoc

    q, k, v = _split(qkv, hkv, d)
    gemma_qkv_rmsnorm(
        q, k, v, q_w, k_w, num_q_heads=_HQ, num_kv_heads=hkv, head_dim=d, eps=_EPS
    )
    q, k = rope.forward_cuda(pos, q, k)
    k = k.reshape(-1, hkv, d)
    v = v.reshape(-1, hkv, d)
    pool.set_kv_buffer(
        SimpleNamespace(layer_id=layer_id),
        KVWriteLoc(loc, swa_loc, full_loc=None),
        k,
        v,
        scale,
        scale,
    )


def _fused_store(pool, layer_id, qkv, q_w, k_w, pos, loc, swa_loc, hkv, d, rope, scale):
    from sglang.kernels.ops.layernorm.gemma4_fused_ops import (
        gemma_qkv_norm_rope_store_kv,
    )
    from sglang.srt.models import gemma4_causal

    with (
        mock.patch.object(
            gemma4_causal, "get_attn_backend", return_value=_triton_backend(swa_loc)
        ),
        mock.patch.object(gemma4_causal, "get_token_to_kv_pool", return_value=pool),
    ):
        target = gemma4_causal._fused_kv_write_target(
            layer_id, SimpleNamespace(out_cache_loc=loc)
        )
    assert target is not None, "the FP8 hybrid pool must take the fused path"
    k_cache, v_cache, target_loc = target
    q, k, v = _split(qkv, hkv, d)
    gemma_qkv_norm_rope_store_kv(
        q,
        k,
        v,
        q_w,
        k_w,
        rope.cos_sin_cache,
        pos,
        target_loc,
        k_cache,
        v_cache,
        scale,
        scale,
        num_q_heads=_HQ,
        num_kv_heads=hkv,
        head_dim=d,
        eps=_EPS,
    )


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _assert_pools_equal(pool, ref):
    for name in ("swa_kv_pool", "full_kv_pool"):
        sub, ref_sub = getattr(pool, name), getattr(ref, name)
        for i in range(len(ref_sub.k_buffer)):
            assert torch.equal(_bits(sub.k_buffer[i]), _bits(ref_sub.k_buffer[i])), (
                name,
                "k",
                i,
            )
            assert torch.equal(_bits(sub.v_buffer[i]), _bits(ref_sub.v_buffer[i])), (
                name,
                "v",
                i,
            )


# Sliding layer 3 sits at index 3 of the SWA sub-pool, so a wrong layer mapping shows; 12 rows is
# a decode batch and 2048 a prefill chunk at --chunked-prefill-size 2048.
@pytest.mark.parametrize("layer_id,kind", [(0, "sliding"), (3, "sliding"), (5, "full")])
@pytest.mark.parametrize("M", [12, 2048])
def test_fused_store_matches_backend_store_byte_for_byte(layer_id, kind, M):
    hkv, d, rope_cfg = _GEOMETRY[kind]
    rope = _rope(d, rope_cfg)
    # The compressed-tensors FP8 checkpoint has no KV scales; its layers carry a 1.0 scale tensor.
    scale = torch.tensor(1.0, device="cuda")
    qkv, q_w, k_w, pos, loc, swa_loc = _inputs(
        M, hkv, d, k_eq_v=kind == "full", seed=M * 31 + layer_id
    )

    ref = _pool()
    _unfused_store(
        ref, layer_id, qkv.clone(), q_w, k_w, pos, loc, swa_loc, hkv, d, rope, scale
    )
    pool = _pool()
    _fused_store(
        pool, layer_id, qkv.clone(), q_w, k_w, pos, loc, swa_loc, hkv, d, rope, scale
    )

    _assert_pools_equal(pool, ref)
    # The write landed: the target rows are not still the zero-initialized buffer.
    written = (
        ref.swa_kv_pool.k_buffer[_SWA_LAYERS.index(layer_id)][swa_loc]
        if kind == "sliding"
        else ref.full_kv_pool.k_buffer[_FULL_LAYERS.index(layer_id)][loc]
    )
    assert _bits(written).any()


def test_write_target_falls_back_off_the_static_fp8_triton_path():
    from sglang.srt.models import gemma4_causal

    loc = torch.arange(4, device="cuda")
    batch = SimpleNamespace(out_cache_loc=loc)

    def target(backend, pool):
        with (
            mock.patch.object(gemma4_causal, "get_attn_backend", return_value=backend),
            mock.patch.object(gemma4_causal, "get_token_to_kv_pool", return_value=pool),
        ):
            return gemma4_causal._fused_kv_write_target(5, batch)

    fp8 = _pool()
    assert target(_triton_backend(loc), fp8) is not None
    # A translating (unified) pool carries a physical full-pool loc: stay unfused.
    assert target(_triton_backend(loc, full_physical=loc), fp8) is None
    # Not the Triton backend.
    assert target(SimpleNamespace(dcp_size=1, forward_metadata=None), fp8) is None
    # A bf16 pool: the fused kernel only writes E4M3.
    assert target(_triton_backend(loc), _pool(dtype=torch.bfloat16)) is None
