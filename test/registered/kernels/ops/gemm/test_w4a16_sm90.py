import sys
from types import SimpleNamespace

import pytest
import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.kernels.ops.gemm.gptq_marlin import gptq_marlin_gemm
from sglang.kernels.ops.gemm.w4a16_sm90 import (
    _SMEM_LIMIT,
    _TOKEN_TILES,
    GROUP_SIZE,
    MAX_M,
    _fits_registers,
    _flags,
    _flags_for,
    _jit_w4a16_sm90_module,
    _launch_config,
    _smem_bytes,
    supports_w4a16_sm90,
    w4a16_sm90_gemm,
)
from sglang.kernels.ops.quantization.awq_marlin_repack import awq_marlin_repack
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.gpu.quantization.awq_kernels import (
    AWQMarlinLinearKernel,
)
from sglang.srt.layers.quantization.marlin_utils import (
    awq_to_marlin_zero_points,
    marlin_make_workspace,
    marlin_permute_scales,
)
from sglang.srt.layers.quantization.utils import quantize_weights
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_marlin_utils import awq_pack

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="w4a16_sm90_gemm requires SM90",
)

# (N, K) per GPU of DeepSeek V3.2's dense AWQ projections under DP-attention
# (attention unsharded) and the TP8 MLP, plus the smallest legal shape.
SHAPES = [
    (64, 256),
    (2112, 7168),  # fused q_a + kv_a
    (24576, 1536),  # q_b
    (7168, 16384),  # o_proj
    (8192, 1536),  # indexer wq_b
    (128, 7168),  # indexer wk
    (512, 7168),  # shared expert gate_up
    (7168, 256),  # shared expert down
    (4608, 7168),  # dense MLP gate_up (layers 0-2)
    (7168, 2304),  # dense MLP down (layers 0-2)
]
# The attention side runs at M = 48 (c128 EAGLE), 64 (c512) and 96 (c256
# EAGLE); the MLP side, after the DP all-gather, at that summed over the DP
# ranks, up to MAX_M (c1024).
M_VALUES = [1, 3, 8, 16, 17, 33, 48, 64, 96, 100, 128, 192, 193, 256, 384, 500]
M_VALUES += [512, 768, MAX_M]


def _make_layer(n: int, k: int):
    """AWQ-packed weights put through AWQMarlinLinearKernel's load-time transforms."""
    w = torch.randn((k, n), dtype=torch.bfloat16, device="cuda")
    w_ref, q_w, s, zp = quantize_weights(
        w, scalar_types.uint4, GROUP_SIZE, zero_points=True
    )
    num_groups = k // GROUP_SIZE
    qweight = awq_marlin_repack(awq_pack(q_w, 4, k, n), k, n, 4)
    scales = marlin_permute_scales(s, k, n, GROUP_SIZE)
    qzeros = awq_to_marlin_zero_points(awq_pack(zp, 4, num_groups, n), num_groups, n, 4)
    return w_ref, qweight, scales, qzeros


def _marlin(a, qweight, scales, qzeros, n, k):
    return gptq_marlin_gemm(
        a,
        None,
        qweight,
        scales,
        None,
        qzeros,
        None,
        None,
        marlin_make_workspace(a.device),
        scalar_types.uint4,
        size_m=a.shape[0],
        size_n=n,
        size_k=k,
    )


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("m", M_VALUES)
def test_matches_dequantized_reference_and_marlin(m, n, k):
    torch.manual_seed(m * 7 + n + k)
    w_ref, qweight, scales, qzeros = _make_layer(n, k)
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")

    out = w4a16_sm90_gemm(a, qweight, scales, qzeros, n)

    ref = a.float() @ w_ref.float()
    marlin = _marlin(a, qweight, scales, qzeros, n, k)
    # Both kernels dequantize to the same bf16 weights; ours sums in fp32 and
    # rounds once, so it stays within one bf16 ulp of the fp32 reference, while
    # Marlin's bf16 split-K reduction does not.
    torch.testing.assert_close(out.float(), ref, rtol=2**-7, atol=1e-2)
    ours_err = (out.float() - ref).abs().max()
    marlin_err = (marlin.float() - ref).abs().max()
    assert ours_err <= marlin_err + 1e-2, (ours_err, marlin_err)


def test_strided_activation():
    n, k, m = 2112, 7168, 5
    _, qweight, scales, qzeros = _make_layer(n, k)
    wide = torch.randn((m, k + 64), dtype=torch.bfloat16, device="cuda")
    a = wide[:, :k]

    out = w4a16_sm90_gemm(a, qweight, scales, qzeros, n)
    expected = w4a16_sm90_gemm(a.contiguous(), qweight, scales, qzeros, n)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


# One explicit launch per reduction mode, independent of the launch heuristic:
# (token_tile, tiles, ping, cluster_k, stages, min_groups_per_cta).
MODES = {
    "unsplit": (64, 2, 1, 1, 2, 0),
    "cluster": (64, 2, 1, 4, 2, 0),
    "stream_k": (64, 1, 2, 0, 2, 2),
    "unsplit_192": (192, 1, 1, 1, 2, 0),
    "cluster_192": (192, 1, 1, 2, 2, 0),
    "stream_k_192": (192, 1, 2, 0, 2, 2),
}


def _run_mode(mode, a, qweight, scales, qzeros, n):
    *config, min_groups = MODES[mode]
    out = torch.empty((a.shape[0], n), dtype=a.dtype, device=a.device)
    _jit_w4a16_sm90_module(*config).run(
        out, a, qweight, scales, qzeros, _flags_for(a.device), min_groups
    )
    return out


@pytest.mark.parametrize("mode", sorted(MODES))
# 40 fits one token block of every mode; 600 spans several, the last one partial.
@pytest.mark.parametrize("m", [40, 600])
def test_reduction_mode_exact_deterministic_and_graph_safe(mode, m):
    """Each split-K reduction matches the reference, repeats bit-exactly and replays in a CUDA graph."""
    n, k = 1024, 7168
    w_ref, qweight, scales, qzeros = _make_layer(n, k)
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")

    eager = _run_mode(mode, a, qweight, scales, qzeros, n)
    torch.testing.assert_close(
        eager.float(), a.float() @ w_ref.float(), rtol=2**-7, atol=1e-2
    )
    for _ in range(3):
        torch.testing.assert_close(
            _run_mode(mode, a, qweight, scales, qzeros, n), eager, rtol=0, atol=0
        )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run_mode(mode, a, qweight, scales, qzeros, n)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, eager, rtol=0, atol=0)


def test_stream_k_leaves_flags_zero():
    """A stream's cached flags are all zero after a stream-K launch, so the next launch can reuse them."""
    n, k, m = 1024, 7168, 40
    _, qweight, scales, qzeros = _make_layer(n, k)
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")

    _run_mode("stream_k", a, qweight, scales, qzeros, n)
    torch.cuda.synchronize()
    key = (a.device.index, torch.cuda.current_stream().cuda_stream)
    assert int(_flags[key].count_nonzero()) == 0


def test_stream_k_first_launch_in_capture_keeps_its_own_flags():
    """Graphs whose stream had no flags yet each zero their own buffer, so any replay order is exact."""
    n, k, m = 1024, 7168, 40
    w_ref, qweight, scales, qzeros = _make_layer(n, k)
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    ref = a.float() @ w_ref.float()
    stream = torch.cuda.Stream()

    graphs, outs = [], []
    for _ in range(2):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outs.append(_run_mode("stream_k", a, qweight, scales, qzeros, n))
        graphs.append(graph)
    assert (a.device.index, stream.cuda_stream) not in _flags

    for graph, out in zip(reversed(graphs), reversed(outs)):
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), ref, rtol=2**-7, atol=1e-2)


def _awq_layer(n: int, k: int) -> torch.nn.Module:
    """A linear layer holding raw AWQ tensors, as the checkpoint loader leaves it."""
    w = torch.randn((k, n), dtype=torch.bfloat16, device="cuda")
    _, q_w, s, zp = quantize_weights(
        w, scalar_types.uint4, GROUP_SIZE, zero_points=True
    )
    num_groups = k // GROUP_SIZE
    layer = torch.nn.Module()
    layer.qweight = torch.nn.Parameter(awq_pack(q_w, 4, k, n), requires_grad=False)
    layer.qzeros = torch.nn.Parameter(
        awq_pack(zp, 4, num_groups, n), requires_grad=False
    )
    layer.scales = torch.nn.Parameter(s, requires_grad=False)
    layer.input_size_per_partition = k
    layer.output_size_per_partition = n
    layer.num_groups = num_groups
    return layer


def test_awq_marlin_linear_routes_m_up_to_max_m_only():
    n, k = 2112, 7168
    quant_config = SimpleNamespace(group_size=GROUP_SIZE, quant_type=scalar_types.uint4)
    with envs.SGLANG_USE_W4A16_SM90_GEMM.override(True):
        kernel = AWQMarlinLinearKernel(quant_config)
    layer = _awq_layer(n, k)
    kernel.process_weights_after_loading(layer)
    marlin_only = AWQMarlinLinearKernel(quant_config)
    bias = torch.randn(n, dtype=torch.bfloat16, device="cuda")

    for tokens in (8, MAX_M):
        x = torch.randn((2, tokens // 2, k), dtype=torch.bfloat16, device="cuda")
        expected = (
            w4a16_sm90_gemm(
                x.reshape(-1, k), layer.qweight, layer.scales, layer.qzeros, n
            ).reshape(2, tokens // 2, n)
            + bias
        )
        torch.testing.assert_close(
            kernel.apply(layer, x, bias), expected, rtol=0, atol=0
        )

    large = torch.randn((MAX_M + 1, k), dtype=torch.bfloat16, device="cuda")
    torch.testing.assert_close(
        kernel.apply(layer, large, bias),
        marlin_only.apply(layer, large, bias),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("n", [64, 128, 512, 2112, 4608, 7168, 8192, 24576])
@pytest.mark.parametrize("k", [256, 1536, 2304, 7168, 16384])
def test_launch_config_is_launchable(n, k):
    """Every supported shape gets a config the kernel's own checks accept."""
    for m in range(1, MAX_M + 1):
        token_tile, tiles, ping, cluster_k, stages, _ = _launch_config(m, n, k, 0)
        assert token_tile in _TOKEN_TILES
        assert stages % ping == 0
        assert n % (64 * tiles) == 0
        assert _fits_registers(token_tile, tiles, ping)
        assert _smem_bytes(token_tile, tiles, ping, cluster_k, stages) <= _SMEM_LIMIT
        assert cluster_k <= k // GROUP_SIZE
        assert (64 * tiles // max(cluster_k, 1)) % 8 == 0


@pytest.mark.parametrize(
    "m,n,k,group_size,dtype,expected",
    [
        (1, 7168, 2048, 128, torch.bfloat16, True),
        (MAX_M, 7168, 2048, 128, torch.bfloat16, True),
        (MAX_M + 1, 7168, 2048, 128, torch.bfloat16, False),
        (0, 7168, 2048, 128, torch.bfloat16, False),
        (8, 7168, 2048, 64, torch.bfloat16, False),
        (8, 7168, 2048, 128, torch.float16, False),
        (8, 7200, 2048, 128, torch.bfloat16, False),
        (8, 7168, 128, 128, torch.bfloat16, False),
    ],
)
def test_supports(m, n, k, group_size, dtype, expected):
    assert supports_w4a16_sm90(m, n, k, group_size, dtype) is expected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
