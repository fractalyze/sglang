"""Fused triton kernels for Gemma4 decoder layer operations.

Fuses standard RMSNorm + residual-add (+ optional scalar multiply) into
a single kernel pass to reduce kernel launch overhead.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _gemma_rmsnorm_residual_kernel(
    X_ptr,
    W_ptr,
    Residual_ptr,
    Scalar_ptr,
    Out_ptr,
    stride_x,
    stride_r,
    stride_o,
    N,
    eps,
    HAS_SCALAR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: out = rmsnorm(x, w) + residual [* scalar]

    When HAS_SCALAR is True, also multiplies by a scalar loaded from Scalar_ptr.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(Residual_ptr + row * stride_r + cols, mask=mask, other=0.0).to(
        tl.float32
    )

    var = tl.sum(x * x, axis=0) / N
    rrms = tl.rsqrt(var + eps)
    out = x * rrms * w + r

    if HAS_SCALAR:
        scalar = tl.load(Scalar_ptr).to(tl.float32)
        out = out * scalar

    tl.store(Out_ptr + row * stride_o + cols, out.to(x.dtype), mask=mask)


def gemma_rmsnorm_residual_scalar(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    scalar: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused (rmsnorm(x) + residual) * scalar."""
    assert x.dim() == 2 and x.stride(-1) == 1, "Expected contiguous 2D input"
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    out = torch.empty_like(x)

    _gemma_rmsnorm_residual_kernel[(M,)](
        x,
        weight,
        residual,
        scalar,
        out,
        x.stride(0),
        residual.stride(0),
        out.stride(0),
        N,
        eps,
        HAS_SCALAR=True,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


@triton.jit
def _gemma_dual_rmsnorm_residual_kernel(
    X1_ptr,
    W1_ptr,
    X2_ptr,
    W2_ptr,
    W3_ptr,
    Residual_ptr,
    Scalar_ptr,
    Out_ptr,
    stride_x1,
    stride_x2,
    stride_r,
    stride_o,
    N,
    eps1,
    eps2,
    eps3,
    NextW_ptr,
    NextOut_ptr,
    stride_n,
    eps_next,
    BLOCK_SIZE: tl.constexpr,
    HAS_NEXT: tl.constexpr,
):
    """Fused: out = (rmsnorm(rmsnorm(x1,w1) + rmsnorm(x2,w2), w3) + residual) * scalar

    With HAS_NEXT, also next_out = rmsnorm(out, next_w): the next layer's input norm
    (or the final model norm), computed from the stored activation-dtype `out`.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x1 = tl.load(X1_ptr + row * stride_x1 + cols, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(W1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(X2_ptr + row * stride_x2 + cols, mask=mask, other=0.0).to(tl.float32)
    w2 = tl.load(W2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w3 = tl.load(W3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(Residual_ptr + row * stride_r + cols, mask=mask, other=0.0).to(
        tl.float32
    )

    var1 = tl.sum(x1 * x1, axis=0) / N
    norm1 = x1 * tl.rsqrt(var1 + eps1) * w1

    var2 = tl.sum(x2 * x2, axis=0) / N
    norm2 = x2 * tl.rsqrt(var2 + eps2) * w2

    combined = norm1 + norm2

    var3 = tl.sum(combined * combined, axis=0) / N
    norm3 = combined * tl.rsqrt(var3 + eps3) * w3

    scalar = tl.load(Scalar_ptr).to(tl.float32)
    out = ((norm3 + r) * scalar).to(Out_ptr.dtype.element_ty)

    tl.store(Out_ptr + row * stride_o + cols, out, mask=mask)

    if HAS_NEXT:
        y = out.to(tl.float32)
        wn = tl.load(NextW_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        rrms = tl.rsqrt(tl.sum(y * y, axis=0) / N + eps_next)
        next_out = (y * rrms * wn).to(NextOut_ptr.dtype.element_ty)
        tl.store(NextOut_ptr + row * stride_n + cols, next_out, mask=mask)


@triton.jit
def _gemma_qkv_rmsnorm_store(
    X_ptr,
    W_ptr,
    stride_m,
    m,
    h,
    cols,
    mask,
    HEAD_DIM: tl.constexpr,
    eps,
    HAS_WEIGHT: tl.constexpr,
):
    off = m * stride_m + h * HEAD_DIM + cols
    x = tl.load(X_ptr + off, mask=mask, other=0.0).to(tl.float32)
    rrms = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)
    out = x * rrms
    if HAS_WEIGHT:
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = out * w
    tl.store(X_ptr + off, out.to(X_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _gemma_qkv_rmsnorm_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    Q_w_ptr,
    K_w_ptr,
    stride_q_m,
    stride_k_m,
    stride_v_m,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    eps,
    HAS_KV: tl.constexpr,
    BY_HEAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fused per-head RMSNorm for Q, K, V.

    The same kernel supports two launch shapes:
    - BY_HEAD=True: grid is (M, total_heads), one program per token/head.
    - BY_HEAD=False: grid is (M,), one program per token looping over heads.

    Layout assumption: each tensor's last dim packs (num_heads, head_dim)
    contiguously so per-head offset is `h * HEAD_DIM`. The token (M) stride is
    taken from stride_*_m so the kernel works on strided views (e.g. slices of a
    larger qkv buffer produced by `qkv.split`) without requiring `.contiguous()`
    copies. V uses `weight=ones` semantics so the multiply-by-weight is omitted.
    """
    m = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < HEAD_DIM

    if BY_HEAD:
        h_all = tl.program_id(1)
        if h_all < NUM_Q_HEADS:
            _gemma_qkv_rmsnorm_store(
                Q_ptr, Q_w_ptr, stride_q_m, m, h_all, cols, mask, HEAD_DIM, eps, True
            )
        elif HAS_KV and h_all < NUM_Q_HEADS + NUM_KV_HEADS:
            h = h_all - NUM_Q_HEADS
            _gemma_qkv_rmsnorm_store(
                K_ptr, K_w_ptr, stride_k_m, m, h, cols, mask, HEAD_DIM, eps, True
            )
        elif HAS_KV:
            h = h_all - NUM_Q_HEADS - NUM_KV_HEADS
            _gemma_qkv_rmsnorm_store(
                V_ptr, Q_w_ptr, stride_v_m, m, h, cols, mask, HEAD_DIM, eps, False
            )
    else:
        for h in tl.static_range(NUM_Q_HEADS):
            _gemma_qkv_rmsnorm_store(
                Q_ptr, Q_w_ptr, stride_q_m, m, h, cols, mask, HEAD_DIM, eps, True
            )

        if HAS_KV:
            for h in tl.static_range(NUM_KV_HEADS):
                _gemma_qkv_rmsnorm_store(
                    K_ptr,
                    K_w_ptr,
                    stride_k_m,
                    m,
                    h,
                    cols,
                    mask,
                    HEAD_DIM,
                    eps,
                    True,
                )

            for h in tl.static_range(NUM_KV_HEADS):
                _gemma_qkv_rmsnorm_store(
                    V_ptr,
                    Q_w_ptr,
                    stride_v_m,
                    m,
                    h,
                    cols,
                    mask,
                    HEAD_DIM,
                    eps,
                    False,
                )


def gemma_qkv_rmsnorm(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    q_weight: torch.Tensor,
    k_weight: Optional[torch.Tensor],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float = 1e-6,
) -> None:
    """In-place fused RMSNorm on Q, K, V for Gemma4 attention.

    All three norms compute `x * rsqrt(mean(x^2) + eps)` independently per head.
    Q is scaled by `q_weight`, K by `k_weight`, V by 1 (Gemma4's V-norm has
    `with_scale=False`).

    Inputs may be 2D `(M, num_heads * head_dim)` or strided views of a larger
    buffer (such as q/k/v slices from `qkv.split`). The kernel uses the actual
    `stride(0)` so no `.contiguous()` copy is required. Within a token, the
    last dim must be contiguous so heads pack as `h * head_dim` offsets.

    If k and v are both None (KV-shared layer), only Q is normalized.
    """
    assert q.is_cuda or q.is_xpu
    assert q.stride(-1) == 1, "Q's last dim must be contiguous"
    assert q_weight.shape[-1] == head_dim
    M = q.shape[0] if q.dim() >= 2 else 1
    BLOCK = triton.next_power_of_2(head_dim)

    has_kv = k is not None and v is not None
    if has_kv:
        assert (k.is_cuda and v.is_cuda) or (k.is_xpu and v.is_xpu)
        assert k.stride(-1) == 1 and v.stride(-1) == 1
        assert k_weight is not None and k_weight.shape[-1] == head_dim

    if M <= 256:
        total_heads = num_q_heads + (2 * num_kv_heads if has_kv else 0)
        _gemma_qkv_rmsnorm_kernel[(M, total_heads)](
            q,
            k if has_kv else q,
            v if has_kv else q,
            q_weight,
            k_weight if has_kv else q_weight,
            q.stride(0),
            k.stride(0) if has_kv else 0,
            v.stride(0) if has_kv else 0,
            NUM_Q_HEADS=num_q_heads,
            NUM_KV_HEADS=num_kv_heads if has_kv else 0,
            HEAD_DIM=head_dim,
            eps=eps,
            HAS_KV=has_kv,
            BY_HEAD=True,
            BLOCK=BLOCK,
        )
        return

    _gemma_qkv_rmsnorm_kernel[(M,)](
        q,
        k if has_kv else q,
        v if has_kv else q,
        q_weight,
        k_weight if has_kv else q_weight,
        q.stride(0),
        k.stride(0) if has_kv else 0,
        v.stride(0) if has_kv else 0,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads if has_kv else 0,
        HEAD_DIM=head_dim,
        eps=eps,
        HAS_KV=has_kv,
        BY_HEAD=False,
        BLOCK=BLOCK,
    )


@triton.jit
def _gemma_routing_post_topk_kernel(
    Logits_ptr,
    Ids_ptr,
    Scale_ptr,
    Out_weights_ptr,
    Out_ids_ptr,
    stride_l,
    stride_ow,
    stride_oi,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused: softmax(topk_logits) * per_expert_scale[topk_ids] → float32 weights, int32 ids.

    One program per token. K is the number of top-k experts (e.g. 8).
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    mask = cols < K

    logits = tl.load(
        Logits_ptr + row * stride_l + cols, mask=mask, other=float("-inf")
    ).to(tl.float32)
    ids_i64 = tl.load(Ids_ptr + row * stride_l + cols, mask=mask, other=0)

    # Stable softmax
    max_val = tl.max(logits, axis=0)
    exp_val = tl.exp(logits - max_val)
    sum_exp = tl.sum(exp_val, axis=0)
    weights = exp_val / sum_exp

    # Gather per_expert_scale and multiply
    scale = tl.load(Scale_ptr + ids_i64, mask=mask, other=1.0).to(tl.float32)
    weights = weights * scale

    tl.store(Out_weights_ptr + row * stride_ow + cols, weights, mask=mask)
    tl.store(Out_ids_ptr + row * stride_oi + cols, ids_i64.to(tl.int32), mask=mask)


def gemma_routing_post_topk(
    topk_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    per_expert_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused softmax + scale-gather + casts for Gemma4 routing.

    Replaces: softmax(topk_logits) * per_expert_scale[topk_ids] → (f32, i32).
    """
    B, K = topk_logits.shape
    BLOCK_K = triton.next_power_of_2(K)
    out_weights = torch.empty((B, K), dtype=torch.float32, device=topk_logits.device)
    out_ids = torch.empty((B, K), dtype=torch.int32, device=topk_logits.device)

    _gemma_routing_post_topk_kernel[(B,)](
        topk_logits,
        topk_ids,
        per_expert_scale,
        out_weights,
        out_ids,
        topk_logits.stride(0),
        out_weights.stride(0),
        out_ids.stride(0),
        K=K,
        BLOCK_K=BLOCK_K,
    )
    return out_weights, out_ids


def gemma_dual_rmsnorm_residual_scalar(
    x1: torch.Tensor,
    weight1: torch.Tensor,
    x2: torch.Tensor,
    weight2: torch.Tensor,
    weight3: torch.Tensor,
    residual: torch.Tensor,
    scalar: torch.Tensor,
    eps1: float = 1e-6,
    eps2: float = 1e-6,
    eps3: float = 1e-6,
) -> torch.Tensor:
    """Fused (rmsnorm(rmsnorm(x1,w1) + rmsnorm(x2,w2), w3) + residual) * scalar."""
    out, _ = _launch_dual_rmsnorm_residual(
        x1, weight1, x2, weight2, weight3, residual, scalar, eps1, eps2, eps3, None, 0.0
    )
    return out


def gemma_dual_rmsnorm_residual_scalar_next_norm(
    x1: torch.Tensor,
    weight1: torch.Tensor,
    x2: torch.Tensor,
    weight2: torch.Tensor,
    weight3: torch.Tensor,
    residual: torch.Tensor,
    scalar: torch.Tensor,
    next_weight: torch.Tensor,
    eps1: float = 1e-6,
    eps2: float = 1e-6,
    eps3: float = 1e-6,
    next_eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """gemma_dual_rmsnorm_residual_scalar plus rmsnorm(out, next_weight) in the same pass."""
    return _launch_dual_rmsnorm_residual(
        x1,
        weight1,
        x2,
        weight2,
        weight3,
        residual,
        scalar,
        eps1,
        eps2,
        eps3,
        next_weight,
        next_eps,
    )


def _launch_dual_rmsnorm_residual(
    x1,
    weight1,
    x2,
    weight2,
    weight3,
    residual,
    scalar,
    eps1,
    eps2,
    eps3,
    next_weight,
    next_eps,
):
    assert x1.dim() == 2 and x1.stride(-1) == 1
    M, N = x1.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    out = torch.empty_like(x1)
    next_out = torch.empty_like(x1) if next_weight is not None else None

    _gemma_dual_rmsnorm_residual_kernel[(M,)](
        x1,
        weight1,
        x2,
        weight2,
        weight3,
        residual,
        scalar,
        out,
        x1.stride(0),
        x2.stride(0),
        residual.stride(0),
        out.stride(0),
        N,
        eps1,
        eps2,
        eps3,
        next_weight if next_weight is not None else weight3,
        next_out if next_out is not None else out,
        out.stride(0),
        next_eps,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_NEXT=next_weight is not None,
    )
    return out, next_out


@triton.jit
def _gemma4_routing_kernel(
    gating_ptr,  # [T, E] router logits, any float dtype
    per_expert_scale_ptr,  # [E] per-expert scale (any float dtype)
    topk_weights_ptr,  # [T, K] fp32 out
    topk_ids_ptr,  # [T, K] int32 out
    stride_g_t,  # stride of gating in the token dim
    E: tl.constexpr,
    K: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    valid = offs_e < E

    logits = tl.load(
        gating_ptr + pid * stride_g_t + offs_e,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)

    # Pack (sort_key, expert_id) into one int64 so a single signed-ascending
    # tl.sort yields logits in descending float order. The key bijection is
    # anti-monotone on the float value, and the <<32 shift moves its high bit
    # into the int64 sign bit. Ties break by expert id ascending. Invalid
    # lanes use a max key so they sort last.
    MIN32 = -2147483648
    logit_bits = logits.to(tl.int32, bitcast=True)
    sign = logit_bits >> 31
    key = tl.where(sign == 0, logit_bits ^ -1, logit_bits ^ MIN32)
    key = tl.where(valid, key, 0x7FFFFFFF)
    sk64 = key.to(tl.int64) & 0x00000000FFFFFFFF
    packed = (sk64 << 32) | offs_e.to(tl.int64)

    sorted_p = tl.sort(packed, descending=False)
    all_keys = ((sorted_p >> 32) & 0x00000000FFFFFFFF).to(tl.int32)
    all_ids = (sorted_p & 0x00000000FFFFFFFF).to(tl.int32)

    # Invert the key bijection to recover the original logit value.
    sign_k = all_keys >> 31
    all_bits = tl.where(sign_k < 0, all_keys ^ -1, all_keys ^ MIN32)
    all_logits = all_bits.to(tl.float32, bitcast=True)

    # softmax over the top-K logits; max sits at index 0 (sorted descending).
    top_mask = offs_e < K
    max_l = tl.max(tl.where(top_mask, all_logits, -float("inf")), axis=0)
    raw_exp = tl.where(top_mask, tl.exp(all_logits - max_l), 0.0)

    denom = tl.sum(raw_exp, axis=0)
    denom = tl.where(denom > 0.0, denom, 1.0)
    weights = raw_exp / denom

    scales = tl.load(
        per_expert_scale_ptr + all_ids.to(tl.int64),
        mask=top_mask,
        other=1.0,
    ).to(tl.float32)
    weights = weights * scales

    base_off = pid * K + offs_e
    tl.store(topk_weights_ptr + base_off, weights, mask=top_mask)
    tl.store(topk_ids_ptr + base_off, all_ids, mask=top_mask)


def gemma4_fused_routing(
    gating_output: torch.Tensor,
    per_expert_scale: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-launch Gemma4 router.

    Args:
        gating_output: [T, E] router logits in any floating dtype; will be
            cast to fp32 inside the kernel.
        per_expert_scale: [E] per-expert scale, any floating dtype.
        topk: number of experts to keep per token.

    Returns:
        topk_weights: [T, topk] fp32 (matches SGLang TopK contract).
        topk_ids: [T, topk] int32 (matches SGLang TopK contract).
    """
    assert gating_output.dim() == 2, "expected [T, E] router logits"
    assert per_expert_scale.dim() == 1
    assert per_expert_scale.shape[0] == gating_output.shape[1]
    T, E = gating_output.shape
    assert topk <= E, f"topk ({topk}) must be <= E ({E})"
    assert E <= 1024, f"gemma4_fused_routing only supports E<=1024, got E={E}"

    gating_output = gating_output.contiguous()
    per_expert_scale = per_expert_scale.contiguous()

    BLOCK_E = triton.next_power_of_2(E)
    topk_weights = torch.empty(
        (T, topk), dtype=torch.float32, device=gating_output.device
    )
    topk_ids = torch.empty((T, topk), dtype=torch.int32, device=gating_output.device)

    if T == 0:
        return topk_weights, topk_ids

    _gemma4_routing_kernel[(T,)](
        gating_output,
        per_expert_scale,
        topk_weights,
        topk_ids,
        gating_output.stride(0),
        E,
        topk,
        BLOCK_E,
        num_warps=1,
    )
    return topk_weights, topk_ids


@triton.jit
def _gemma_head_rms(X_ptr, off, HEAD_DIM: tl.constexpr, eps):
    # Same load shape and reduction as _gemma_qkv_rmsnorm_store, so rrms is bit-identical.
    cols = tl.arange(0, HEAD_DIM)
    x = tl.load(X_ptr + off + cols).to(tl.float32)
    return tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)


@triton.jit
def _gemma_norm_half(
    X_ptr, W_ptr, off, w_off, half_cols, rrms, HAS_WEIGHT: tl.constexpr
):
    # Rounds to the activation dtype, as the unfused norm's store does before RoPE reads it.
    x = tl.load(X_ptr + off + half_cols).to(tl.float32) * rrms
    if HAS_WEIGHT:
        x = x * tl.load(W_ptr + w_off + half_cols).to(tl.float32)
    return x.to(X_ptr.dtype.element_ty).to(tl.float32)


@triton.jit
def _gemma_store_fp8(Cache_ptr, Scale_ptr, slot_off, cols, x, HAS_SCALE: tl.constexpr):
    # Mirrors set_kv_buffer: bf16 x.div_(fp32 0-dim scale) casts the scale to bf16 (type
    # promotion), divides in fp32 and rounds to bf16; then .to(float8_e4m3fn) rounds to nearest even.
    if HAS_SCALE:
        scale = tl.load(Scale_ptr).to(tl.bfloat16).to(tl.float32)
        x = tl.div_rn(x, scale).to(tl.bfloat16).to(tl.float32)
    tl.store(Cache_ptr + slot_off + cols, x.to(Cache_ptr.dtype.element_ty))


@triton.jit
def _gemma_qkv_norm_rope_store_kv_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    Q_w_ptr,
    K_w_ptr,
    CosSin_ptr,
    Pos_ptr,
    Loc_ptr,
    K_cache_ptr,
    V_cache_ptr,
    K_scale_ptr,
    V_scale_ptr,
    stride_q_m,
    stride_k_m,
    stride_v_m,
    stride_cache_slot,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    eps,
    HAS_K_SCALE: tl.constexpr,
    HAS_V_SCALE: tl.constexpr,
):
    """One program per (token, head): RMSNorm, then neox RoPE on q/k, then the FP8 KV-cache store.

    Reproduces the unfused sequence gemma_qkv_rmsnorm -> apply_rope_with_cos_sin_cache_inplace
    -> MHATokenToKVPool.set_kv_buffer: q and k are written back roped, v normed, all in the
    activation dtype; k and v also land in the cache row Loc[m] as E4M3.
    """
    m = tl.program_id(0)
    h_all = tl.program_id(1)
    HALF: tl.constexpr = HEAD_DIM // 2
    half_cols = tl.arange(0, HALF)
    slot_off = tl.load(Loc_ptr + m).to(tl.int64) * stride_cache_slot

    if h_all < NUM_Q_HEADS + NUM_KV_HEADS:
        if h_all < NUM_Q_HEADS:
            X_ptr = Q_ptr
            W_ptr = Q_w_ptr
            off = m * stride_q_m + h_all * HEAD_DIM
        else:
            X_ptr = K_ptr
            W_ptr = K_w_ptr
            off = m * stride_k_m + (h_all - NUM_Q_HEADS) * HEAD_DIM
        rrms = _gemma_head_rms(X_ptr, off, HEAD_DIM, eps)
        x1 = _gemma_norm_half(X_ptr, W_ptr, off, 0, half_cols, rrms, True)
        x2 = _gemma_norm_half(X_ptr, W_ptr, off + HALF, HALF, half_cols, rrms, True)
        # cos_sin_cache row: [cos(HALF) | sin(HALF)], fp32. The FMA forms are the ones nvcc
        # contracts rope.cuh's neox `x*cos - y*sin`, `x*sin + y*cos` into (matched bit for bit).
        cs = CosSin_ptr + tl.load(Pos_ptr + m).to(tl.int64) * HEAD_DIM
        cos = tl.load(cs + half_cols)
        sin = tl.load(cs + HALF + half_cols)
        o1 = tl.fma(x1, cos, -(x2 * sin)).to(X_ptr.dtype.element_ty)
        o2 = tl.fma(x2, cos, x1 * sin).to(X_ptr.dtype.element_ty)
        tl.store(X_ptr + off + half_cols, o1)
        tl.store(X_ptr + off + HALF + half_cols, o2)
        if h_all >= NUM_Q_HEADS:
            c_off = slot_off + (h_all - NUM_Q_HEADS) * HEAD_DIM
            _gemma_store_fp8(
                K_cache_ptr,
                K_scale_ptr,
                c_off,
                half_cols,
                o1.to(tl.float32),
                HAS_K_SCALE,
            )
            _gemma_store_fp8(
                K_cache_ptr,
                K_scale_ptr,
                c_off + HALF,
                half_cols,
                o2.to(tl.float32),
                HAS_K_SCALE,
            )
    else:
        h = h_all - NUM_Q_HEADS - NUM_KV_HEADS
        off = m * stride_v_m + h * HEAD_DIM
        rrms = _gemma_head_rms(V_ptr, off, HEAD_DIM, eps)
        c_off = slot_off + h * HEAD_DIM
        for part in tl.static_range(2):
            v = _gemma_norm_half(
                V_ptr, V_ptr, off + part * HALF, 0, half_cols, rrms, False
            )
            tl.store(
                V_ptr + off + part * HALF + half_cols, v.to(V_ptr.dtype.element_ty)
            )
            _gemma_store_fp8(
                V_cache_ptr, V_scale_ptr, c_off + part * HALF, half_cols, v, HAS_V_SCALE
            )


def gemma_qkv_norm_rope_store_kv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    loc: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: Optional[torch.Tensor],
    v_scale: Optional[torch.Tensor],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float = 1e-6,
) -> None:
    """In place: q, k <- RoPE(RMSNorm(.)), v <- RMSNorm(v); k, v -> FP8 cache rows `loc`.

    q/k/v are [M, heads * head_dim] (strided slices of the qkv buffer are fine);
    k_cache/v_cache are one layer's [slots, num_kv_heads, head_dim] float8_e4m3fn
    buffers; cos_sin_cache is the fp32 [max_pos, head_dim] neox cache covering the
    whole head (Gemma4 pads non-rotated dims with cos=1, sin=0).
    """
    assert q.dtype == torch.bfloat16 and k.dtype == q.dtype and v.dtype == q.dtype
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    assert cos_sin_cache.dtype == torch.float32 and cos_sin_cache.shape[-1] == head_dim
    assert cos_sin_cache.is_contiguous()
    assert k_cache.dtype == torch.float8_e4m3fn and v_cache.dtype == k_cache.dtype
    assert (
        k_cache.shape[1:] == (num_kv_heads, head_dim)
        and k_cache.stride() == v_cache.stride()
    )
    assert k_cache.stride(1) == head_dim and k_cache.stride(2) == 1
    M = q.shape[0]
    if M == 0:
        return
    _gemma_qkv_norm_rope_store_kv_kernel[(M, num_q_heads + 2 * num_kv_heads)](
        q,
        k,
        v,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        loc,
        k_cache,
        v_cache,
        k_scale if k_scale is not None else k_cache,
        v_scale if v_scale is not None else v_cache,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        k_cache.stride(0),
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        eps=eps,
        HAS_K_SCALE=k_scale is not None,
        HAS_V_SCALE=v_scale is not None,
    )


@triton.jit
def _gemma_rmsnorm_add_rmsnorm_kernel(
    X_ptr,
    W1_ptr,
    Residual_ptr,
    W2_ptr,
    stride_x,
    stride_r,
    N,
    eps1,
    eps2,
    BLOCK_SIZE: tl.constexpr,
):
    """In place: residual += rmsnorm(x, w1); x = rmsnorm(residual, w2).

    Reproduces RMSNorm(x) followed by FusedAddRMSNorm(., residual): the first norm's
    output is rounded to the activation dtype, the sum feeds the second norm in fp32
    and is stored back as the new residual.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(W1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    h = (x * tl.rsqrt(tl.sum(x * x, axis=0) / N + eps1) * w1).to(X_ptr.dtype.element_ty)
    r = tl.load(Residual_ptr + row * stride_r + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    s = h.to(tl.float32) + r
    tl.store(
        Residual_ptr + row * stride_r + cols,
        s.to(Residual_ptr.dtype.element_ty),
        mask=mask,
    )
    w2 = tl.load(W2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = s * tl.rsqrt(tl.sum(s * s, axis=0) / N + eps2) * w2
    tl.store(X_ptr + row * stride_x + cols, out.to(X_ptr.dtype.element_ty), mask=mask)


def gemma_rmsnorm_add_rmsnorm(
    x: torch.Tensor,
    weight1: torch.Tensor,
    residual: torch.Tensor,
    weight2: torch.Tensor,
    eps1: float = 1e-6,
    eps2: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """In place on x and residual; returns (x, residual) like FusedAddRMSNorm."""
    assert x.dim() == 2 and x.stride(-1) == 1 and residual.stride(-1) == 1
    M, N = x.shape
    if M == 0:
        return x, residual
    _gemma_rmsnorm_add_rmsnorm_kernel[(M,)](
        x,
        weight1,
        residual,
        weight2,
        x.stride(0),
        residual.stride(0),
        N,
        eps1,
        eps2,
        BLOCK_SIZE=triton.next_power_of_2(N),
    )
    return x, residual


@triton.jit
def _gemma_dual_output_rmsnorm_kernel(
    X_ptr,
    W1_ptr,
    W2_ptr,
    Out1_ptr,
    Out2_ptr,
    stride_x,
    stride_o,
    N,
    eps1,
    eps2,
    BLOCK_SIZE: tl.constexpr,
):
    """out1 = rmsnorm(x, w1), out2 = rmsnorm(x, w2) from one read of x."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    ss = tl.sum(x * x, axis=0) / N
    w1 = tl.load(W1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w2 = tl.load(W2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out1 = x * tl.rsqrt(ss + eps1) * w1
    out2 = x * tl.rsqrt(ss + eps2) * w2
    tl.store(
        Out1_ptr + row * stride_o + cols, out1.to(X_ptr.dtype.element_ty), mask=mask
    )
    tl.store(
        Out2_ptr + row * stride_o + cols, out2.to(X_ptr.dtype.element_ty), mask=mask
    )


def gemma_dual_output_rmsnorm(
    x: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    eps1: float = 1e-6,
    eps2: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(rmsnorm(x, w1), rmsnorm(x, w2)) in one kernel: the router norm and pre-FF norm 2."""
    assert x.dim() == 2 and x.stride(-1) == 1
    M, N = x.shape
    out1 = torch.empty_like(x)
    out2 = torch.empty_like(x)
    if M == 0:
        return out1, out2
    _gemma_dual_output_rmsnorm_kernel[(M,)](
        x,
        weight1,
        weight2,
        out1,
        out2,
        x.stride(0),
        out1.stride(0),
        N,
        eps1,
        eps2,
        BLOCK_SIZE=triton.next_power_of_2(N),
    )
    return out1, out2
