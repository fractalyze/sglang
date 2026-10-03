"""Byte-level tests for ``gemma_qkv_norm_rope_store_kv`` (Gemma4 fused decode glue, group A).

The reference is the unfused production path: ``gemma_qkv_rmsnorm`` ->
``apply_rope_with_cos_sin_cache_inplace`` -> ``MHATokenToKVPool.set_kv_buffer``
on an FP8 E4M3 pool. The fused kernel must reproduce the q/k/v activations and
every byte of the K and V cache bit for bit, for both Gemma4 attention shapes:
sliding (16 q / 8 kv heads, head_dim 256, default RoPE) and full (16 q / 2 kv
heads, head_dim 512, proportional RoPE over a quarter of the head, separate K
and V copies even though V is projected with K's weights).

Requires a CUDA GPU with FP8 support; skips otherwise.
"""

from __future__ import annotations

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="FP8 E4M3 KV cache needs a CUDA GPU with SM 8.9+",
)

_SLOTS = 4096
_MAX_POS = 8192
_EPS = 1e-6

# (name, q heads, kv heads, head_dim, rope)
_SHAPES = [
    ("sliding", 16, 8, 256, ("default", 10000.0, 1.0)),
    ("full", 16, 2, 512, ("proportional", 1000000.0, 0.25)),
]


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


def _pool(num_kv_heads: int, head_dim: int):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    return MHATokenToKVPool(
        size=_SLOTS,
        page_size=1,
        dtype=torch.float8_e4m3fn,
        head_num=num_kv_heads,
        head_dim=head_dim,
        layer_num=1,
        device="cuda",
        enable_memory_saver=False,
        enable_alt_stream=False,
    )


class _Layer:
    """The attributes set_kv_buffer reads off a RadixAttention layer."""

    layer_id = 0


def _inputs(M, hq, hkv, d, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    qkv = torch.randn(M, (hq + 2 * hkv) * d, device="cuda", generator=g).to(
        torch.bfloat16
    )
    qkv *= 3.0
    q_w = (1.0 + 0.5 * torch.randn(d, device="cuda", generator=g)).to(torch.bfloat16)
    k_w = (1.0 + 0.5 * torch.randn(d, device="cuda", generator=g)).to(torch.bfloat16)
    pos = torch.randint(
        0, _MAX_POS, (M,), device="cuda", generator=g, dtype=torch.int64
    )
    # Distinct, scattered slots: stands in for the full->SWA ring translation of out_cache_loc.
    loc = torch.randperm(_SLOTS, device="cuda", generator=g)[:M].to(torch.int64)
    return qkv, q_w, k_w, pos, loc


def _split(qkv, hq, hkv, d):
    return qkv.split([hq * d, hkv * d, hkv * d], dim=-1)


def _reference(qkv, q_w, k_w, pos, loc, hq, hkv, d, rope, scale):
    from sglang.kernels.ops.layernorm.gemma4_fused_ops import gemma_qkv_rmsnorm

    q, k, v = _split(qkv, hq, hkv, d)
    gemma_qkv_rmsnorm(
        q, k, v, q_w, k_w, num_q_heads=hq, num_kv_heads=hkv, head_dim=d, eps=_EPS
    )
    q, k = rope.forward_cuda(pos, q, k)
    k = k.reshape(-1, hkv, d)
    v = v.reshape(-1, hkv, d)
    roped_k = k.clone()
    pool = _pool(hkv, d)
    pool.set_kv_buffer(_Layer(), loc, k, v, scale, scale)
    return q, roped_k, v, pool


def _fused(qkv, q_w, k_w, pos, loc, hq, hkv, d, rope, scale):
    from sglang.kernels.ops.layernorm.gemma4_fused_ops import (
        gemma_qkv_norm_rope_store_kv,
    )

    q, k, v = _split(qkv, hq, hkv, d)
    pool = _pool(hkv, d)
    gemma_qkv_norm_rope_store_kv(
        q,
        k,
        v,
        q_w,
        k_w,
        rope.cos_sin_cache,
        pos,
        loc,
        pool.get_key_buffer(0),
        pool.get_value_buffer(0),
        scale,
        scale,
        num_q_heads=hq,
        num_kv_heads=hkv,
        head_dim=d,
        eps=_EPS,
    )
    return q, k.reshape(-1, hkv, d), v.reshape(-1, hkv, d), pool


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8 if t.element_size() == 1 else torch.int16)


@pytest.mark.parametrize("shape", _SHAPES, ids=[s[0] for s in _SHAPES])
@pytest.mark.parametrize("M", [1, 8, 22, 32, 300])
@pytest.mark.parametrize("scale_value", [None, 1.0, 0.37])
def test_bit_exact_against_unfused_path(shape, M, scale_value):
    _, hq, hkv, d, rope_cfg = shape
    rope = _rope(d, rope_cfg)
    scale = None if scale_value is None else torch.tensor(scale_value, device="cuda")
    qkv, q_w, k_w, pos, loc = _inputs(M, hq, hkv, d, seed=M * 7 + hkv)

    ref_q, ref_k, ref_v, ref_pool = _reference(
        qkv.clone(), q_w, k_w, pos, loc, hq, hkv, d, rope, scale
    )
    q, k, v, pool = _fused(qkv.clone(), q_w, k_w, pos, loc, hq, hkv, d, rope, scale)

    assert torch.equal(_bits(q), _bits(ref_q))
    # The unfused set_kv_buffer divides k and v in place by their scales before the cast;
    # the fused kernel leaves the activations undivided, so compare the pre-division values.
    assert torch.equal(_bits(k), _bits(ref_k))
    if scale_value in (None, 1.0):
        assert torch.equal(_bits(v), _bits(ref_v))
    # Whole buffers, so a write to a wrong slot or head shows up as well as a wrong byte.
    assert torch.equal(_bits(pool.k_buffer[0]), _bits(ref_pool.k_buffer[0]))
    assert torch.equal(_bits(pool.v_buffer[0]), _bits(ref_pool.v_buffer[0]))


def test_unfused_kv_write_launches_four_elementwise_kernels():
    """Pins the attribution of the four ATen elementwise kernels in the decode profile."""
    _, hq, hkv, d, rope_cfg = _SHAPES[0]
    rope = _rope(d, rope_cfg)
    scale = torch.tensor(1.0, device="cuda")
    qkv, q_w, k_w, pos, loc = _inputs(8, hq, hkv, d, seed=1)
    q, k, v = _split(qkv, hq, hkv, d)
    pool = _pool(hkv, d)
    k = k.reshape(-1, hkv, d)
    v = v.reshape(-1, hkv, d)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        pool.set_kv_buffer(_Layer(), loc, k, v, scale, scale)
        torch.cuda.synchronize()
    names = [
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]
    elementwise = [n for n in names if "elementwise_kernel" in n]
    assert len(elementwise) == 4, names
