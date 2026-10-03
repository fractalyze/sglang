"""Per-shape AWQ MoE GEMM benchmark on one H100: w4a16_sm90 vs Marlin vs Humming.

Shapes are DeepSeek-V3.2 at TP8 (256 experts, top-8, hidden 7168, intermediate
256 per partition). Bandwidth is the AWQ checkpoint bytes the GEMM must read
(4-bit weights, bf16 scales, 4-bit zeros) over the measured time. Each call
reads at least 235 MB of weights against a 50 MB L2, so CUDA-graph replay of
fixed inputs still measures cold weight traffic.

    python test/manual/kernels/bench_w4a16_moe_sm90.py --out results.jsonl
"""

import argparse
import json
import sys
from types import SimpleNamespace

import torch

from sglang.kernels.ops.moe.w4a16_moe_sm90 import (
    TOKEN_BLOCKS,
    repack_awq_moe_weights,
    select_token_block,
    w4a16_moe_sm90_gemm,
)
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

NUM_EXPERTS = 256
TOP_K = 8
HIDDEN = 7168
INTERMEDIATE = 256
GROUP = 128
# H100 SXM5 HBM3 peak.
PEAK_BYTES_PER_S = 3.35e12
# (name, K, N, a_row_divisor)
PROJECTIONS = [
    ("gate_up", HIDDEN, 2 * INTERMEDIATE, TOP_K),
    ("down", INTERMEDIATE, HIDDEN, 1),
]


def _random_awq(k: int, n: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    kw = dict(dtype=torch.int32, device="cuda", generator=gen)
    qweight = torch.randint(-(2**31), 2**31 - 1, (NUM_EXPERTS, k, n // 8), **kw)
    qzeros = torch.randint(-(2**31), 2**31 - 1, (NUM_EXPERTS, k // GROUP, n // 8), **kw)
    scales = torch.rand((NUM_EXPERTS, k // GROUP, n), device="cuda", generator=gen)
    return qweight, (scales * 0.02 + 0.002).to(torch.bfloat16), qzeros


def _routing(num_tokens: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.rand(num_tokens, NUM_EXPERTS, device="cuda", generator=gen)
    topk_weights, topk_ids = scores.topk(TOP_K, dim=-1)
    return topk_ids.to(torch.int32), (topk_weights / topk_weights.sum(-1, True)).float()


def _floor_bytes(k: int, n: int, active_experts: int, a_bytes: int) -> int:
    per_expert = n * k // 2 + n * (k // GROUP) * 2 + n * (k // GROUP) // 2
    return active_experts * per_expert + a_bytes


def _time_us(fn, iters: int) -> float:
    graph = torch.cuda.CUDAGraph()
    fn()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(3):
        graph.replay()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters


def _marlin_gemm(qweight, scales, qzeros, k, n):
    from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
    from sglang.srt.hardware_backend.gpu.quantization.awq_kernels import AWQMoEKernel
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
        get_scalar_type,
    )

    layer = torch.nn.Module()
    for name, t in (("qweight", qweight), ("scales", scales), ("qzeros", qzeros)):
        layer.register_parameter(
            f"w13_{name}", torch.nn.Parameter(t.clone(), requires_grad=False)
        )
        layer.register_parameter(
            f"w2_{name}", torch.nn.Parameter(t[:1].clone(), requires_grad=False)
        )
    layer.intermediate_size_per_partition = INTERMEDIATE
    quant = SimpleNamespace(pack_factor=8, weight_bits=4, group_size=GROUP)
    AWQMoEKernel(quant).process_weights_after_loading(layer)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    workspace = torch.zeros(sms * 4, dtype=torch.int32, device="cuda")
    scalar_type = get_scalar_type(4, True, layer.w13_scales, None)

    def run(a, out, routing, topk_weights, block, top_k, num_tokens):
        sorted_ids, expert_ids, num_post = routing
        moe_wna16_marlin_gemm(
            a,
            out,
            layer.w13_qweight,
            None,
            layer.w13_scales,
            None,
            layer.w13_qzeros,
            None,
            layer.w13_g_idx_sort_indices,
            workspace,
            sorted_ids,
            expert_ids,
            num_post,
            topk_weights,
            moe_block_size=block,
            top_k=top_k,
            mul_topk_weights=False,
            is_ep=False,
            b_q_type=scalar_type,
            size_m=num_tokens,
            size_n=n,
            size_k=k,
            is_k_full=True,
            use_atomic_add=True,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

    return run


def _marlin_block(num_tokens: int) -> int:
    # Same selection as fused_marlin_moe.
    for block in [8, 16, 32, 48, 64]:
        if num_tokens * TOP_K / NUM_EXPERTS / block < 0.9:
            return block
    return 64


def _humming_gemm(qweight, scales, qzeros, k, n):
    from humming.config import GemmType
    from humming.layer import HummingLayer, HummingMethod

    layer = HummingLayer(
        shape_n=n,
        shape_k=k,
        weight_config={"quant_method": "awq", "bits": 4, "group_size": GROUP},
        num_experts=NUM_EXPERTS,
        torch_dtype=torch.bfloat16,
    ).cuda()
    layer.load_from_tensors({"qweight": qweight, "scales": scales, "qzeros": qzeros})
    layer.transform()
    # Same configs the sglang humming runner builds for its indexed MoE GEMM.
    compute_config = json.dumps({"use_f16_accum": False, "gemm_type": "indexed"})
    tuning = HummingMethod.get_default_tuning_configs(
        layer=layer, use_f16_accum=False, gemm_type=GemmType.INDEXED, sublayer_name=""
    )

    def block_for(valid_shape_m):
        for low, high, config in tuning:
            if low < valid_shape_m <= high:
                return config["block_shape"][0]
        raise ValueError(f"no humming tuning entry for shape_m={valid_shape_m}")

    def run(a, out, routing, top_k, num_tokens):
        sorted_ids, expert_ids, num_post = routing
        layer(
            a,
            outputs=out,
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_padded=num_post,
            top_k=top_k,
            valid_shape_m=num_tokens * top_k,
            compute_config=compute_config,
            tuning_config=json.dumps(tuning),
        )

    return run, block_for


def bench_projection(
    name, k, n, a_row_divisor, tokens_per_expert, iters, sweep_blocks, results
):
    qweight, scales, qzeros = _random_awq(k, n, seed=k + n)
    ours_weights = repack_awq_moe_weights(qweight, scales, qzeros, GROUP)
    marlin = _marlin_gemm(qweight, scales, qzeros, k, n)
    try:
        humming, humming_block = _humming_gemm(qweight, scales, qzeros, k, n)
    except Exception as exc:  # report and keep the other legs
        humming = None
        results.append({"projection": name, "method": "humming", "error": repr(exc)})

    for tpe in tokens_per_expert:
        num_tokens = tpe * NUM_EXPERTS // TOP_K
        topk_ids, topk_weights = _routing(num_tokens, seed=tpe)
        active = int(topk_ids.unique().numel())
        a_rows = num_tokens if a_row_divisor == TOP_K else num_tokens * TOP_K
        a = torch.randn(a_rows, k, device="cuda").to(torch.bfloat16)
        out = torch.empty(num_tokens * TOP_K, n, device="cuda", dtype=torch.bfloat16)
        floor = _floor_bytes(k, n, active, a.numel() * 2 + out.numel() * 2)
        top_k = TOP_K if a_row_divisor == TOP_K else 1

        legs = {}

        def ours(block):
            routing = moe_align_block_size(topk_ids, block, NUM_EXPERTS)
            return lambda: w4a16_moe_sm90_gemm(
                a=a,
                out=out,
                weights=ours_weights,
                sorted_token_ids=routing[0],
                expert_ids=routing[1],
                num_tokens_post_padded=routing[2],
                topk_weights=None,
                token_block=block,
                a_row_divisor=a_row_divisor,
            )

        selected = select_token_block(
            num_tokens=num_tokens, top_k=TOP_K, num_experts=NUM_EXPERTS
        )
        legs["w4a16_sm90"] = ours(selected)
        if sweep_blocks:
            for block in TOKEN_BLOCKS:
                if block != selected:
                    legs[f"w4a16_sm90_b{block}"] = ours(block)
        m_block = _marlin_block(num_tokens)
        m_routing = moe_align_block_size(topk_ids, m_block, NUM_EXPERTS)
        legs["marlin"] = lambda: marlin(
            a, out, m_routing, topk_weights, m_block, top_k, a_rows
        )
        if humming is not None:
            h_block = humming_block(a_rows * top_k)
            h_routing = moe_align_block_size(topk_ids, h_block, NUM_EXPERTS)
            legs["humming"] = lambda: humming(a, out, h_routing, top_k, a_rows)

        for method, fn in legs.items():
            row = {
                "projection": name,
                "method": method,
                "tokens_per_expert": tpe,
                "num_tokens": num_tokens,
                "k": k,
                "n": n,
            }
            try:
                us = _time_us(fn, iters)
                row.update(
                    us=round(us, 2),
                    gbps=round(floor / us / 1e3, 1),
                    pct_peak=round(100 * floor / (us * 1e-6) / PEAK_BYTES_PER_S, 1),
                )
            except Exception as exc:
                row["error"] = repr(exc)
            results.append(row)
            print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--tokens-per-expert", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128]
    )
    parser.add_argument(
        "--sweep-blocks",
        action="store_true",
        help="also time w4a16_sm90 at every token block, not only the selected one",
    )
    args = parser.parse_args()
    if torch.cuda.get_device_capability()[0] != 9:
        sys.exit("needs an SM90 (Hopper) GPU")

    results = []
    for name, k, n, divisor in PROJECTIONS:
        bench_projection(
            name=name,
            k=k,
            n=n,
            a_row_divisor=divisor,
            tokens_per_expert=args.tokens_per_expert,
            iters=args.iters,
            sweep_blocks=args.sweep_blocks,
            results=results,
        )
    with open(args.out, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
