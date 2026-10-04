"""SM90 W4A16 GEMM vs the stock dense AWQ paths (Marlin, Humming) on AWQ g128 weights.

The `hbm_floor` column is the time to move the kernel's bytes (weights, scales,
zeros, activations, output) once at the device's measured copy bandwidth. Every
tensor argument is cloned per replay, so weights are read L2-cold as in serving.
The pinned operating points are M = 48 (c128 EAGLE), 64 (c512) and 96 (c256 EAGLE).

Run on a Hopper (SM90) GPU:
    python test/registered/kernels/benchmark/gemm/bench_w4a16_sm90.py
"""

import json

import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.gemm.gptq_marlin import gptq_marlin_gemm
from sglang.kernels.ops.gemm.w4a16_sm90 import GROUP_SIZE, w4a16_sm90_gemm
from sglang.srt.layers.quantization.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT,
    marlin_make_workspace,
    should_use_atomic_add_reduce,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=90, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

# (N, K) per GPU of DeepSeek V3.2's dense AWQ projections under DP-attention
# (attention unsharded) and TP8 shared experts.
SHAPES = [
    (2112, 7168),  # fused q_a + kv_a
    (24576, 1536),  # q_b
    (7168, 16384),  # o_proj
    (8192, 1536),  # indexer wq_b
    (128, 7168),  # indexer wk
    (512, 7168),  # shared expert gate_up
    (7168, 256),  # shared expert down
]
M_VALUES = [1, 8, 16, 32, 48, 64, 96, 128, 192]
# The Humming call SGLang makes for a dense linear (--quantization humming).
_HUMMING_COMPUTE = json.dumps({"use_f16_accum": False, "gemm_type": "dense"})


@cache_once
def _copy_bytes_per_s() -> float:
    """The device's achievable DRAM bandwidth: a 1 GiB device-to-device copy."""
    src = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    result = marker.do_bench(
        lambda: dst.copy_(src),
        metrics=(0.5,),
        graph_clone_args=None,
        disable_log_bandwidth=True,
    )
    # `times` holds one value per metric: here the median. A copy reads and
    # writes every byte.
    return 2 * src.numel() / result.times[0]


def _random_int32(*shape: int) -> torch.Tensor:
    return torch.randint(-(2**31), 2**31 - 1, shape, dtype=torch.int32, device="cuda")


def _marlin_operands(n: int, k: int):
    # Random bits are a valid Marlin-layout operand set; values do not matter here.
    qweight = _random_int32(k // 16, 2 * n)
    scales = torch.randn((k // GROUP_SIZE, n), dtype=torch.bfloat16, device="cuda")
    qzeros = _random_int32(k // GROUP_SIZE, n // 8)
    return qweight, scales, qzeros


@cache_once
def _humming_layer(n: int, k: int):
    try:
        from humming.layer import HummingLayer
    except ImportError:
        return None
    layer = HummingLayer(
        shape_n=n,
        shape_k=k,
        weight_config={
            "quant_method": "awq",
            "bits": 4,
            "group_size": GROUP_SIZE,
            "zero_point": True,
        },
        pad_n_to_multiple=256,
        pad_k_to_multiple=128,
        torch_dtype=torch.bfloat16,
    )
    num_groups = k // GROUP_SIZE
    layer.load_from_tensors(
        {
            "qweight": _random_int32(k, n // 8).cpu(),
            "scales": torch.randn((num_groups, n), dtype=torch.bfloat16),
            "qzeros": _random_int32(num_groups, n // 8).cpu(),
        }
    )
    layer = layer.cuda()
    layer.transform()
    return layer


def _bench_humming(a: torch.Tensor, n: int, k: int):
    layer = _humming_layer(n, k)
    if layer is None:
        marker.skip("humming is not installed")
    from humming.forward import humming_forward

    def run(a, weight, weight_scale, zero_point):
        return humming_forward(
            layer.humming_config,
            inputs=a,
            weight=weight,
            weight_scale=weight_scale,
            zero_point=zero_point,
            locks=layer.locks,
            compute_config=_HUMMING_COMPUTE,
        )

    return marker.do_bench(
        run, input_args=(a, layer.weight, layer.weight_scale, layer.zero_point)
    )


def _bench_marlin(a: torch.Tensor, n: int, k: int):
    m = a.shape[0]
    use_atomic_add = should_use_atomic_add_reduce(
        m=m, n=n, k=k, device=a.device, dtype=a.dtype
    )
    workspace = marlin_make_workspace(a.device)

    def run(a, qweight, scales, qzeros):
        return gptq_marlin_gemm(
            a,
            None,
            qweight,
            scales,
            None,
            qzeros,
            None,
            None,
            workspace,
            scalar_types.uint4,
            size_m=m,
            size_n=n,
            size_k=k,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=USE_FP32_REDUCE_DEFAULT,
        )

    return marker.do_bench(run, input_args=(a, *_marlin_operands(n, k)))


@marker.parametrize("shape", SHAPES, [(7168, 16384), (128, 7168)])
@marker.parametrize("m", M_VALUES, [8, 64])
@marker.benchmark("impl", ["w4a16_sm90", "marlin", "humming", "hbm_floor"])
def benchmark(shape, m, impl):
    n, k = shape
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    if impl == "hbm_floor":
        nbytes = sum(t.nbytes for t in (a, *_marlin_operands(n, k))) + m * n * 2
        return marker.BenchResult(
            metrics=(0.5, "avg"),
            times=[nbytes / _copy_bytes_per_s()],
            memory_footprint=nbytes,
        )
    if impl == "w4a16_sm90":
        return marker.do_bench(
            w4a16_sm90_gemm, input_args=(a, *_marlin_operands(n, k), n)
        )
    if impl == "humming":
        return _bench_humming(a, n, k)
    return _bench_marlin(a, n, k)


if __name__ == "__main__":
    benchmark.run()
