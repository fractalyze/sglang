"""Per-shape large-M AWQ MoE GEMM benchmark on one H100: w4fp8_sm90 vs w4a16_sm90, Marlin and Humming W4A8.

Shapes and weights are those of bench_w4a16_moe_sm90 (DeepSeek-V3.2 at TP8). Each
row reports the time, the achieved TFLOP/s, and the floor: the larger of the FP8
tensor-core time and the AWQ byte-streaming time. ``w4fp8_sm90+quant`` includes the
activation quantisation the runner adds in front of the GEMM. As in that bench,
every call streams at least 235 MB of expert weights against a 50 MB L2, so
CUDA-graph replay of fixed inputs still measures cold weight traffic.

    python test/manual/kernels/bench_w4fp8_moe_sm90.py --out results.jsonl
"""

import argparse
import json
import sys

import torch
from bench_w4a16_moe_sm90 import (
    GROUP,
    NUM_EXPERTS,
    PEAK_BYTES_PER_S,
    PROJECTIONS,
    TOP_K,
    _floor_bytes,
    _marlin_block,
    _marlin_gemm,
    _random_awq,
    _routing,
    _time_us,
)

from sglang.kernels.ops.moe.w4a16_moe_sm90 import (
    repack_awq_moe_weights,
    select_token_block,
    w4a16_moe_sm90_gemm,
)
from sglang.kernels.ops.moe.w4fp8_moe_sm90 import (
    TOKEN_BLOCK,
    quantize_activations,
    w4fp8_moe_sm90_gemm,
)
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

# H100 SXM5 dense FP8 tensor-core peak.
PEAK_FP8_FLOPS = 989.4e12 * 2
# The documented SGLANG_HUMMING_INPUT_QUANT_CONFIG value for FP8 activations.
HUMMING_W4A8_INPUT_CONFIG = {"dtype": "float8e4m3"}


def _humming_w4a8_legs(*, qweight, scales, qzeros, k, n):
    """Humming W4A8 on the same AWQ weights, as its MoE runner serves it.

    Returns {gemm type: run(a, out, topk_ids, a_row_divisor)}; each leg includes
    the input quantisation, and grouped_contiguous gate-up also its token permute.
    """
    from humming.config import GemmType
    from humming.layer import HummingLayer, HummingMethod

    from sglang.kernels.ops.moe.ep_moe_kernels import moe_permute

    layer = HummingLayer(
        shape_n=n,
        shape_k=k,
        weight_config={"quant_method": "awq", "bits": 4, "group_size": GROUP},
        input_config=dict(HUMMING_W4A8_INPUT_CONFIG),
        num_experts=NUM_EXPERTS,
        torch_dtype=torch.bfloat16,
    ).cuda()
    layer.load_from_tensors({"qweight": qweight, "scales": scales, "qzeros": qzeros})
    layer.transform()

    def configs(gemm_type):
        compute = json.dumps({"use_f16_accum": False, "gemm_type": gemm_type.value})
        tuning = HummingMethod.get_default_tuning_configs(
            layer=layer, use_f16_accum=False, gemm_type=gemm_type, sublayer_name=""
        )
        return compute, tuning

    def indexed(a, out, topk_ids, a_row_divisor):
        compute, tuning = configs(GemmType.INDEXED)
        valid_shape_m = topk_ids.numel()
        block = next(
            c["block_shape"][0] for lo, hi, c in tuning if lo < valid_shape_m <= hi
        )
        sorted_ids, expert_ids, num_padded = moe_align_block_size(
            topk_ids, block, NUM_EXPERTS
        )
        tuning_str = json.dumps(tuning)

        def run():
            q, q_scale = HummingMethod.may_quant_input(layer=layer, inputs=a)
            layer(
                q,
                outputs=out,
                input_scale=q_scale,
                sorted_ids=sorted_ids,
                expert_ids=expert_ids,
                num_tokens_padded=num_padded,
                top_k=TOP_K if a_row_divisor == TOP_K else 1,
                valid_shape_m=valid_shape_m,
                compute_config=compute,
                tuning_config=tuning_str,
            )

        return run

    def grouped(a, out, topk_ids, a_row_divisor):
        compute, tuning = configs(GemmType.GROUPED_CONTIGUOUS)
        tuning_str = json.dumps(tuning)
        valid_shape_m = topk_ids.numel()
        # Down's input is gate-up's output, already in expert order.
        _, _, expert_offsets = moe_permute(
            inputs=a[: topk_ids.shape[0]], topk_ids=topk_ids, num_experts=NUM_EXPERTS
        )

        def run():
            rows = a
            if a_row_divisor == TOP_K:
                rows, _, _ = moe_permute(
                    inputs=a, topk_ids=topk_ids, num_experts=NUM_EXPERTS
                )
            q, q_scale = HummingMethod.may_quant_input(layer=layer, inputs=rows)
            layer(
                q,
                outputs=out,
                input_scale=q_scale,
                expert_layout=expert_offsets,
                valid_shape_m=valid_shape_m,
                compute_config=compute,
                tuning_config=tuning_str,
            )

        return run

    return {"indexed": indexed, "grouped": grouped}


def _legs(*, a_row_divisor, a, out, topk_ids, topk_weights, weights, marlin, humming):
    num_tokens = topk_ids.shape[0]
    a_rows = a.shape[0]
    top_k = TOP_K if a_row_divisor == TOP_K else 1
    a_q, a_scales = quantize_activations(a)
    fp8_routing = moe_align_block_size(topk_ids, TOKEN_BLOCK, NUM_EXPERTS)

    def fp8_gemm():
        w4fp8_moe_sm90_gemm(
            a=a_q,
            a_scales=a_scales,
            out=out,
            weights=weights,
            sorted_token_ids=fp8_routing[0],
            expert_ids=fp8_routing[1],
            num_tokens_post_padded=fp8_routing[2],
            topk_weights=None,
            a_row_divisor=a_row_divisor,
        )

    def fp8_gemm_with_quant():
        quantize_activations(a)
        fp8_gemm()

    block = select_token_block(
        num_tokens=num_tokens, top_k=TOP_K, num_experts=NUM_EXPERTS
    )
    w4a16_routing = moe_align_block_size(topk_ids, block, NUM_EXPERTS)
    m_block = _marlin_block(num_tokens)
    m_routing = moe_align_block_size(topk_ids, m_block, NUM_EXPERTS)
    legs = {
        "w4fp8_sm90": fp8_gemm,
        "w4fp8_sm90+quant": fp8_gemm_with_quant,
        "w4a16_sm90": lambda: w4a16_moe_sm90_gemm(
            a=a,
            out=out,
            weights=weights,
            sorted_token_ids=w4a16_routing[0],
            expert_ids=w4a16_routing[1],
            num_tokens_post_padded=w4a16_routing[2],
            topk_weights=None,
            token_block=block,
            a_row_divisor=a_row_divisor,
        ),
        "marlin": lambda: marlin(
            a, out, m_routing, topk_weights, m_block, top_k, a_rows
        ),
    }
    for gemm_type, make in humming.items():
        legs[f"humming_w4a8_{gemm_type}"] = make(a, out, topk_ids, a_row_divisor)
    return legs


def bench_projection(name, k, n, a_row_divisor, tokens_per_expert, iters, results):
    qweight, scales, qzeros = _random_awq(k, n, seed=k + n)
    weights = repack_awq_moe_weights(qweight, scales, qzeros, GROUP)
    marlin = _marlin_gemm(qweight, scales, qzeros, k, n)
    humming = _humming_w4a8_legs(
        qweight=qweight, scales=scales, qzeros=qzeros, k=k, n=n
    )

    for tpe in tokens_per_expert:
        num_tokens = tpe * NUM_EXPERTS // TOP_K
        topk_ids, topk_weights = _routing(num_tokens, seed=tpe)
        active = int(topk_ids.unique().numel())
        a_rows = num_tokens if a_row_divisor == TOP_K else num_tokens * TOP_K
        a = torch.randn(a_rows, k, device="cuda").to(torch.bfloat16)
        out = torch.empty(num_tokens * TOP_K, n, device="cuda", dtype=torch.bfloat16)
        flops = 2 * num_tokens * TOP_K * n * k
        floor_bytes = _floor_bytes(k, n, active, a.numel() + out.numel() * 2)
        floor_us = 1e6 * max(flops / PEAK_FP8_FLOPS, floor_bytes / PEAK_BYTES_PER_S)

        legs = _legs(
            a_row_divisor=a_row_divisor,
            a=a,
            out=out,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            weights=weights,
            marlin=marlin,
            humming=humming,
        )
        for method, fn in legs.items():
            row = {
                "projection": name,
                "method": method,
                "tokens_per_expert": tpe,
                "num_tokens": num_tokens,
                "k": k,
                "n": n,
                "floor_us": round(floor_us, 2),
            }
            try:
                us = _time_us(fn, iters)
                row.update(
                    us=round(us, 2),
                    tflops=round(flops / us / 1e6, 1),
                    pct_floor=round(100 * floor_us / us, 1),
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
        "--tokens-per-expert", type=int, nargs="+", default=[32, 64, 128, 256, 1024]
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
            results=results,
        )
    with open(args.out, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
