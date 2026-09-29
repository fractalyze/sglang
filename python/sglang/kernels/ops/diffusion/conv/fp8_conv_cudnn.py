# SPDX-License-Identifier: Apache-2.0
"""FP8 e4m3 3x3 convolutions through cuDNN frontend graphs, for VAE decoders.

The convolution runs on e4m3 activations and e4m3 weights with fp32
accumulation and applies ``descale[k] = w_scale[k] * act_scale`` and the bias
(and optionally a bf16 residual) in fp32 before its one bf16 rounding.
Weights take per-output-channel scales. Activations come from two producers
that write e4m3 directly, so quantization adds no pass:

- ``norm_silu_fp8``: channel RMSNorm + SiLU of a channels_last input (the
  rounding of channel_rmsnorm_nhwc) divided by a static scale. After the
  norm, ``|x_c| <= sqrt(C) * max|gamma|`` and SiLU does not grow magnitudes,
  so ``sqrt(C) * max|gamma| / 448`` never overflows;
- ``upsample2x_fp8``: nearest-2x upsampling with a dynamic per-tensor scale
  (amax / 448, computed on the device, no host sync).

Each graph builds every cuDNN execution plan and keeps the fastest one, timed
on its first execution: the heuristic's first plan is up to 1.6x slower at
full resolution on an RTX 5090. This is a precision change (approx): FP8
activations and weights replace bf16.

Needs the cuDNN frontend Python package (``cudnn``); ``is_available`` says
whether it imports.
"""

from __future__ import annotations

import functools
import os

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

E4M3_MAX = 448.0


@functools.cache
def is_available() -> bool:
    """Whether the cuDNN frontend Python package imports."""
    try:
        _cudnn()
    except ImportError:
        return False
    return True


@functools.cache
def _cudnn():
    # The frontend dlopens the CUDA runtime by name; torch's cu13 wheels ship libcudart.so.13.
    if torch.version.cuda and torch.version.cuda.startswith("13"):
        os.environ.setdefault("CUDNN_FRONTEND_CUDART_LIB_NAME", "libcudart.so.13")
    import cudnn

    return cudnn


_HANDLES: dict[int, object] = {}


def _handle():
    cudnn = _cudnn()
    device = torch.cuda.current_device()
    if device not in _HANDLES:
        _HANDLES[device] = cudnn.create_handle()
    handle = _HANDLES[device]
    cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream().cuda_stream)
    return handle


@triton.jit
def _norm_silu_fp8_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    PIXELS,
    CHANNELS: tl.constexpr,
    SCALE: tl.constexpr,
    INV_ACT_SCALE: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pix = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    ch = tl.arange(0, BLOCK_C)
    mask = (pix[:, None] < PIXELS) & (ch[None, :] < CHANNELS)
    offs = pix[:, None] * CHANNELS + ch[None, :]
    value = tl.load(x_ptr + offs, mask, 0).to(tl.float32)
    norm = tl.maximum(tl.sqrt_rn(tl.sum(value * value, axis=1)), 1.0e-12)
    weight = tl.load(weight_ptr + ch, ch < CHANNELS, 0).to(tl.float32)
    value = tl.div_rn(value, norm[:, None]).to(tl.bfloat16).to(tl.float32)
    value = (value * SCALE).to(tl.bfloat16).to(tl.float32)
    value = (value * weight[None, :]).to(tl.bfloat16).to(tl.float32)
    value = tl.div_rn(value, 1.0 + libdevice.exp(-value)).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + offs, (value * INV_ACT_SCALE).to(tl.float8e4nv), mask)


def norm_act_scale(gamma: torch.Tensor) -> float:
    """Static activation scale of norm_silu_fp8 for a norm with weight ``gamma``."""
    return (gamma.numel() ** 0.5) * gamma.float().abs().max().item() / E4M3_MAX


def norm_silu_fp8(
    x: torch.Tensor, gamma: torch.Tensor, norm_scale: float, act_scale: float
) -> torch.Tensor:
    """bf16 ``x`` [1, C, 1, H, W] channels_last_3d -> e4m3 [1, C, H, W] channels_last:
    ``silu(rmsnorm(x) * norm_scale * gamma) / act_scale``."""
    channels = x.shape[1]
    pixels = x.numel() // channels
    out = torch.empty(
        (x.shape[0], channels, x.shape[3], x.shape[4]),
        device=x.device,
        dtype=torch.float8_e4m3fn,
        memory_format=torch.channels_last,
    )
    block_c = triton.next_power_of_2(channels)
    block_p = max(1, 4096 // block_c)
    _norm_silu_fp8_kernel[(triton.cdiv(pixels, block_p),)](
        x,
        gamma,
        out,
        pixels,
        channels,
        norm_scale,
        1.0 / act_scale,
        block_p,
        block_c,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _upsample2x_fp8_kernel(
    x_ptr, scale_ptr, out_ptr, H, W, C: tl.constexpr, N_OUT, BLOCK: tl.constexpr
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < N_OUT
    c = index % C
    pix = index // C
    ow = pix % (2 * W)
    oh = pix // (2 * W)
    src = ((oh // 2) * W + ow // 2) * C + c
    inv = 1.0 / tl.load(scale_ptr)
    value = tl.load(x_ptr + src, mask, 0).to(tl.float32)
    tl.store(out_ptr + index, (value * inv).to(tl.float8e4nv), mask)


def upsample2x_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 ``x`` [1, C, H, W] channels_last -> (e4m3 [1, C, 2H, 2W] channels_last, scale [1])."""
    c, h, w = x.shape[1], x.shape[2], x.shape[3]
    scale = (x.abs().amax().float() / E4M3_MAX).clamp(min=1e-12).reshape(1)
    out = torch.empty(
        (1, c, 2 * h, 2 * w),
        device=x.device,
        dtype=torch.float8_e4m3fn,
        memory_format=torch.channels_last,
    )
    n = out.numel()
    _upsample2x_fp8_kernel[(triton.cdiv(n, 2048),)](x, scale, out, h, w, c, n, 2048)
    return out, scale


class _TunedGraph:
    """A cuDNN graph with every execution plan built; the fastest runs, chosen on first use."""

    def __init__(self, graph):
        cudnn = _cudnn()
        graph.build_plans(cudnn.build_plan_policy.ALL)
        self.graph = graph
        self.count = graph.get_execution_plan_count()
        size = max(graph.get_workspace_size_plan_at_index(i) for i in range(self.count))
        self.workspace = torch.empty(max(size, 1), device="cuda", dtype=torch.uint8)
        self.best = None

    def _time(self, pack, index, handle):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        self.graph.execute_plan_at_index(pack, self.workspace, index, handle=handle)
        start.record()
        self.graph.execute_plan_at_index(pack, self.workspace, index, handle=handle)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    def execute(self, pack, handle):
        if self.best is None:
            timings = []
            for index in range(self.count):
                try:
                    timings.append((self._time(pack, index, handle), index))
                except RuntimeError:  # a plan that cannot run these tensors
                    continue
            self.best = min(timings)[1]
        self.graph.execute_plan_at_index(pack, self.workspace, self.best, handle=handle)


class Fp8Conv3x3:
    """A 3x3, stride-1, padding-1 ``nn.Conv2d`` (with bias) run as an FP8 cuDNN graph.

    ``act_scale`` is a static activation scale, or None for a scale passed per
    call as a device tensor. One graph per (input size, with residual).
    """

    def __init__(self, conv: torch.nn.Conv2d, act_scale: float | None = None):
        weight = conv.weight.detach().float()
        w_scale = weight.abs().amax(dim=(1, 2, 3)).clamp(min=1e-12) / E4M3_MAX
        self.weight = (
            (weight / w_scale.view(-1, 1, 1, 1))
            .to(torch.float8_e4m3fn)
            .contiguous(memory_format=torch.channels_last)
        )
        self.w_scale = w_scale.view(1, -1, 1, 1).contiguous()
        self.descale = self.w_scale * (1.0 if act_scale is None else act_scale)
        self.bias = conv.bias.detach().float().view(1, -1, 1, 1).contiguous()
        self.out_channels, self.in_channels = weight.shape[0], weight.shape[1]
        self.graphs = {}

    def _graph(self, h: int, w: int, residual: bool):
        key = (h, w, residual)
        if key not in self.graphs:
            cudnn = _cudnn()
            c, k = self.in_channels, self.out_channels
            g = cudnn.pygraph(
                io_data_type=cudnn.data_type.FP8_E4M3,
                intermediate_data_type=cudnn.data_type.FLOAT,
                compute_data_type=cudnn.data_type.FLOAT,
                handle=_handle(),
            )
            x = g.tensor(
                name="x",
                dim=[1, c, h, w],
                stride=[c * h * w, 1, w * c, c],
                data_type=cudnn.data_type.FP8_E4M3,
            )
            wt = g.tensor(
                name="w",
                dim=[k, c, 3, 3],
                stride=[c * 9, 1, 3 * c, c],
                data_type=cudnn.data_type.FP8_E4M3,
            )
            s = g.tensor(
                name="s",
                dim=[1, k, 1, 1],
                stride=[k, 1, 1, 1],
                data_type=cudnn.data_type.FLOAT,
            )
            b = g.tensor(
                name="b",
                dim=[1, k, 1, 1],
                stride=[k, 1, 1, 1],
                data_type=cudnn.data_type.FLOAT,
            )
            y = g.conv_fprop(
                image=x,
                weight=wt,
                padding=[1, 1],
                stride=[1, 1],
                dilation=[1, 1],
                compute_data_type=cudnn.data_type.FLOAT,
            )
            out = g.add(a=g.mul(a=y, b=s), b=b)
            tensors = [x, wt, s, b]
            if residual:
                r = g.tensor(
                    name="r",
                    dim=[1, k, h, w],
                    stride=[k * h * w, 1, w * k, k],
                    data_type=cudnn.data_type.BFLOAT16,
                )
                out = g.add(a=out, b=r)
                tensors.append(r)
            out.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
            g.validate()
            g.build_operation_graph()
            g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
            g.check_support()
            self.graphs[key] = (_TunedGraph(g), tensors + [out])
        return self.graphs[key]

    def __call__(
        self,
        x8: torch.Tensor,
        act_scale: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """e4m3 ``x8`` [1, C, H, W] channels_last -> bf16 [1, K, 1, H, W] channels_last_3d.

        ``act_scale``: the [1] device scale of a dynamic-scale conv. ``residual``:
        bf16 [1, K, 1, H, W] channels_last_3d added before the output rounding.
        """
        if act_scale is not None:
            torch.mul(self.w_scale, act_scale, out=self.descale)
        h, w = x8.shape[2], x8.shape[3]
        graph, tensors = self._graph(h, w, residual is not None)
        y = torch.empty(
            (1, self.out_channels, h, w),
            device=x8.device,
            dtype=torch.bfloat16,
            memory_format=torch.channels_last,
        )
        values = [x8, self.weight, self.descale, self.bias]
        if residual is not None:
            values.append(residual.squeeze(2))
        graph.execute(dict(zip(tensors, values + [y])), _handle())
        return y.unsqueeze(2)
