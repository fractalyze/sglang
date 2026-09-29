# Copyright 2026 Qwen Team and The HuggingFace Team
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.kernels.ops.diffusion import dup_up3d_add
from sglang.kernels.ops.diffusion.conv import fp8_conv_cudnn
from sglang.kernels.ops.diffusion.norm.channel_rmsnorm_nhwc import (
    MAX_CHANNELS as NHWC_NORM_MAX_CHANNELS,
)
from sglang.kernels.ops.diffusion.norm.channel_rmsnorm_nhwc import (
    can_use_channel_rmsnorm_nhwc,
    channel_rmsnorm_nhwc,
)
from sglang.kernels.ops.diffusion.norm.channel_rmsnorm_preserve_reduction import (
    can_use_channel_rmsnorm,
    channel_rmsnorm_preserve_reduction,
)
from sglang.kernels.ops.diffusion.sites.bitexact_gate import BitExactFusionGate
from sglang.multimodal_gen.configs.models.vaes.qwenimage21 import QwenImage21VAEConfig
from sglang.multimodal_gen.runtime.distributed import (
    get_decode_parallel_rank,
    get_decode_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.parallel_conv import (
    SpatialParallelConv2d,
    chunk_height_by_sizes,
    disable_spatial_parallel_decode,
    gather_and_trim_height,
    gather_variable_height,
    split_height_for_parallel_decode,
)
from sglang.multimodal_gen.runtime.models.vaes.common import (
    ParallelTiledVAE,
    can_install_spatial_shard_parallel_decode,
    should_run_spatial_shard_parallel_decode,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.environ import envs

logger = init_logger(__name__)
_CHANNEL_RMSNORM_FUSION = BitExactFusionGate("Qwen-Image 2.1 VAE channel RMSNorm")
_CHANNEL_RMSNORM_SILU_FUSION = BitExactFusionGate(
    "Qwen-Image 2.1 VAE channel RMSNorm + SiLU"
)


def get_activation(name):
    if name != "silu":
        raise ValueError(f"unsupported VAE activation: {name}")
    return nn.SiLU()


class QwenImage21AvgDown3D(nn.Module):
    def __init__(self, in_channels, out_channels, factor_t, factor_s=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s
        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        pad = (0, 0, 0, 0, pad_t, 0)
        x = F.pad(x, pad)
        B, C, T, H, W = x.shape
        x = x.view(
            B,
            C,
            T // self.factor_t,
            self.factor_t,
            H // self.factor_s,
            self.factor_s,
            W // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            B,
            C * self.factor,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.view(
            B,
            self.out_channels,
            self.group_size,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.mean(dim=2)
        return x


class QwenImage21DupUp3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor_t, factor_s=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s
        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk=False) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]
        return x


class QwenImage21CausalConv3d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int | int | int],
        stride: int | tuple[int | int | int] = 1,
        padding: int | tuple[int | int | int] = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        # F.pad order (left, right, top, bottom); the spatial-parallel decode reads it
        self._padding = (
            self.padding[1],
            self.padding[1],
            self.padding[0],
            self.padding[0],
        )

    def forward(self, x, cache_x=None):
        assert cache_x is None
        # The causal pad is symmetric zero padding, so the conv pads implicitly
        # instead of materializing a padded copy (a copy and a fill per conv).
        return super().forward(x.squeeze(2)).unsqueeze(2)


class QwenImage21RMS_norm(nn.Module):
    def __init__(
        self,
        dim: int,
        channel_first: bool = True,
        images: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x, silu: bool = False):
        """RMSNorm over channels, then SiLU when ``silu`` (one pass on the fused path)."""
        plain = (
            self.channel_first
            and isinstance(self.bias, (int, float))
            and self.bias == 0
        )
        if plain and can_use_channel_rmsnorm_nhwc(x, self.gamma):
            # channels_last decode (SGLANG_ENABLE_QWEN_IMAGE21_VAE_CHANNELS_LAST)
            return channel_rmsnorm_nhwc(x, self.gamma, self.scale, silu)
        gate = _CHANNEL_RMSNORM_SILU_FUSION if silu else _CHANNEL_RMSNORM_FUSION
        fused = None
        if (
            self.channel_first
            and isinstance(self.bias, (int, float))
            and self.bias == 0
            and can_use_channel_rmsnorm(x, self.gamma)
            and gate.can_attempt_once()
        ):
            fused = channel_rmsnorm_preserve_reduction(x, self.gamma, self.scale, silu)
            if gate.verified:
                return fused
        normalized = F.normalize(
            x if x.dtype == torch.float64 else x.float(),
            dim=1 if self.channel_first else -1,
        ).to(x.dtype)
        out = normalized * self.scale * self.gamma + self.bias
        if silu:
            out = F.silu(out)
        if fused is not None:
            return gate.accept_or_fallback(fused, out, logger=logger)
        return out


class QwenImage21Upsample(nn.Upsample):
    def forward(self, x):
        # Nearest interpolation copies values; no FP32 arithmetic is needed.
        if self.mode == "nearest-exact" and x.dtype in (torch.float16, torch.bfloat16):
            return super().forward(x)
        return super().forward(x.float()).type_as(x)


class QwenImage21Resample(nn.Module):
    def __init__(self, dim: int, mode: str, upsample_out_dim: int = None) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode
        if upsample_out_dim is None:
            upsample_out_dim = dim // 2
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                QwenImage21Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, upsample_out_dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                QwenImage21Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, upsample_out_dim, 3, padding=1),
            )
            self.time_conv = QwenImage21CausalConv3d(
                dim, dim * 2, (1, 1), padding=(0, 0)
            )
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
            self.time_conv = QwenImage21CausalConv3d(
                dim, dim, (1, 1), stride=(1, 1), padding=(0, 0)
            )
        else:
            self.resample = nn.Identity()
        # The upsample conv as an FP8 cuDNN graph (SGLANG_ENABLE_QWEN_IMAGE21_VAE_FP8_CONV).
        self.fp8_conv = None

    def forward(self, x, feat_cache=None, feat_idx=None):
        b, c, t, h, w = x.size()
        if t == 1 and self.fp8_conv is not None:
            x8, act_scale = fp8_conv_cudnn.upsample2x_fp8(
                x.squeeze(2).contiguous(memory_format=torch.channels_last)
            )
            return self.fp8_conv(x8, act_scale=act_scale)
        if t == 1:
            # squeeze keeps a channels_last_3d input channels_last; the reshape
            # below would copy it back to NCHW
            return self.resample(x.squeeze(2)).unsqueeze(2)
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)
        return x


class QwenImage21ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinearity = get_activation(non_linearity)
        self.norm1 = QwenImage21RMS_norm(in_dim, images=False)
        self.conv1 = QwenImage21CausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = QwenImage21RMS_norm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = QwenImage21CausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = (
            QwenImage21CausalConv3d(in_dim, out_dim, 1)
            if in_dim != out_dim
            else nn.Identity()
        )
        # (scale1, conv1, scale2, conv2) of the FP8 path, set by prepare_fp8_convs.
        self.fp8_convs = None

    def forward(self, x, feat_cache=None, feat_idx=None):
        if self.fp8_convs is not None and x.shape[2] == 1:
            return self._fp8_forward(x)
        h = self.conv_shortcut(x)
        x = self.conv1(self.norm1(x, silu=True))
        x = self.dropout(self.norm2(x, silu=True))
        x = self.conv2(x)
        return x + h

    def _fp8_forward(self, x):
        """norm + SiLU write e4m3, conv1 and conv2 run FP8; conv2 adds the shortcut in fp32."""
        # The mid block's attention hands over an NCHW tensor; the FP8 path is channels_last.
        x = x.contiguous(memory_format=torch.channels_last_3d)
        h = self.conv_shortcut(x).contiguous(memory_format=torch.channels_last_3d)
        scale1, conv1, scale2, conv2 = self.fp8_convs
        y = conv1(
            fp8_conv_cudnn.norm_silu_fp8(x, self.norm1.gamma, self.norm1.scale, scale1)
        )
        y = fp8_conv_cudnn.norm_silu_fp8(
            self.dropout(y), self.norm2.gamma, self.norm2.scale, scale2
        )
        return conv2(y, residual=h)

    def prepare_fp8_convs(self):
        """Quantize conv1 / conv2 for _fp8_forward (static activation scales from the norms)."""
        scale1 = fp8_conv_cudnn.norm_act_scale(self.norm1.gamma)
        scale2 = fp8_conv_cudnn.norm_act_scale(self.norm2.gamma)
        self.fp8_convs = (
            scale1,
            fp8_conv_cudnn.Fp8Conv3x3(self.conv1, scale1),
            scale2,
            fp8_conv_cudnn.Fp8Conv3x3(self.conv2, scale2),
        )


class QwenImage21AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.spatial_parallel = False
        self.norm = QwenImage21RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        if self.spatial_parallel:
            x, heights = gather_variable_height(x)
        identity = x
        batch_size, channels, time, height, width = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
        x = self.norm(x)
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)
        x = (
            x.squeeze(1)
            .permute(0, 2, 1)
            .reshape(batch_size * time, channels, height, width)
        )
        x = self.proj(x)
        x = x.view(batch_size, time, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x + identity
        return chunk_height_by_sizes(x, heights) if self.spatial_parallel else x


class QwenImage21MidBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        num_layers: int = 1,
    ):
        super().__init__()
        self.dim = dim
        resnets = [QwenImage21ResidualBlock(dim, dim, dropout, non_linearity)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(QwenImage21AttentionBlock(dim))
            resnets.append(QwenImage21ResidualBlock(dim, dim, dropout, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None):
        x = self.resnets[0](x, feat_cache=feat_cache, feat_idx=feat_idx)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
        return x


class QwenImage21ResidualDownBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dropout,
        num_res_blocks,
        temperal_downsample=False,
        down_flag=False,
    ):
        super().__init__()
        self.avg_shortcut = QwenImage21AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )
        resnets = []
        for _ in range(num_res_blocks):
            resnets.append(QwenImage21ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            self.downsampler = QwenImage21Resample(out_dim, mode=mode)
        else:
            self.downsampler = None

    def forward(self, x, feat_cache=None, feat_idx=None):
        x_copy = x
        for resnet in self.resnets:
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
        if self.downsampler is not None:
            x = self.downsampler(x, feat_cache=feat_cache, feat_idx=feat_idx)
        return x + self.avg_shortcut(x_copy)


class QwenImage21Encoder3d(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
        non_linearity: str = "silu",
        is_residual: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.nonlinearity = get_activation(non_linearity)
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0
        self.conv_in = QwenImage21CausalConv3d(in_channels, dims[0], 3, padding=1)
        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if is_residual:
                self.down_blocks.append(
                    QwenImage21ResidualDownBlock(
                        in_dim,
                        out_dim,
                        dropout,
                        num_res_blocks,
                        temperal_downsample=(
                            temperal_downsample[i] if i != len(dim_mult) - 1 else False
                        ),
                        down_flag=i != len(dim_mult) - 1,
                    )
                )
            else:
                for _ in range(num_res_blocks):
                    self.down_blocks.append(
                        QwenImage21ResidualBlock(in_dim, out_dim, dropout)
                    )
                    if scale in attn_scales:
                        self.down_blocks.append(QwenImage21AttentionBlock(out_dim))
                    in_dim = out_dim
                if i != len(dim_mult) - 1:
                    mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                    self.down_blocks.append(QwenImage21Resample(out_dim, mode=mode))
                    scale /= 2.0
        self.mid_block = QwenImage21MidBlock(
            out_dim, dropout, non_linearity, num_layers=1
        )
        self.norm_out = QwenImage21RMS_norm(out_dim, images=False)
        self.conv_out = QwenImage21CausalConv3d(out_dim, z_dim, 3, padding=1)
        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None):
        x = self.conv_in(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
        return self.conv_out(self.norm_out(x, silu=True))


class QwenImage21ResidualUpBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        temperal_upsample: bool = False,
        up_flag: bool = False,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if up_flag:
            self.avg_shortcut = QwenImage21DupUp3D(
                in_dim, out_dim, factor_t=2 if temperal_upsample else 1, factor_s=2
            )
        else:
            self.avg_shortcut = None
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(
                QwenImage21ResidualBlock(current_dim, out_dim, dropout, non_linearity)
            )
            current_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        if up_flag:
            upsample_mode = "upsample3d" if temperal_upsample else "upsample2d"
            self.upsampler = QwenImage21Resample(
                out_dim, mode=upsample_mode, upsample_out_dim=out_dim
            )
        else:
            self.upsampler = None
        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        x_copy = x
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsampler is not None:
            x = self.upsampler(x)
        if self.avg_shortcut is not None:
            shortcut = self.avg_shortcut
            if (
                type(shortcut) is QwenImage21DupUp3D
                and x.is_cuda
                and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
                and not torch.compiler.is_compiling()
            ):
                fused = dup_up3d_add(
                    x,
                    x_copy,
                    shortcut.factor_t,
                    shortcut.factor_s,
                    shortcut.repeats,
                    first_chunk,
                )
                if fused is not None:
                    return fused
            x = x + self.avg_shortcut(x_copy, first_chunk=first_chunk)
        return x


class QwenImage21UpBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: str | None = None,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(
                QwenImage21ResidualBlock(current_dim, out_dim, dropout, non_linearity)
            )
            current_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList(
                [QwenImage21Resample(out_dim, mode=upsample_mode)]
            )
        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=None):
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class QwenImage21Decoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
        non_linearity: str = "silu",
        out_channels: int = 3,
        is_residual: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample
        self.nonlinearity = get_activation(non_linearity)
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        self.conv_in = QwenImage21CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.mid_block = QwenImage21MidBlock(
            dims[0], dropout, non_linearity, num_layers=1
        )
        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i > 0 and (not is_residual):
                in_dim = in_dim // 2
            up_flag = i != len(dim_mult) - 1
            upsample_mode = None
            if up_flag and temperal_upsample[i]:
                upsample_mode = "upsample3d"
            elif up_flag:
                upsample_mode = "upsample2d"
            if is_residual:
                up_block = QwenImage21ResidualUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    temperal_upsample=temperal_upsample[i] if up_flag else False,
                    up_flag=up_flag,
                    non_linearity=non_linearity,
                )
            else:
                up_block = QwenImage21UpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    upsample_mode=upsample_mode,
                    non_linearity=non_linearity,
                )
            self.up_blocks.append(up_block)
        self.norm_out = QwenImage21RMS_norm(out_dim, images=False)
        self.conv_out = QwenImage21CausalConv3d(out_dim, out_channels, 3, padding=1)
        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        x = self.conv_in(x)
        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
        for up_block in self.up_blocks:
            x = up_block(
                x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk
            )
        return self.conv_out(self.norm_out(x, silu=True))


def _patchify(x, patch_size):
    if patch_size == 1:
        return x
    if x.dim() != 5:
        raise ValueError(f"Invalid input shape: {x.shape}")
    batch_size, channels, frames, height, width = x.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Height ({height}) and width ({width}) must be divisible by patch_size ({patch_size})"
        )
    x = x.view(
        batch_size,
        channels,
        frames,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(
        batch_size,
        channels * patch_size * patch_size,
        frames,
        height // patch_size,
        width // patch_size,
    )
    return x


def _unpatchify(x, patch_size):
    if patch_size == 1:
        return x
    if x.dim() != 5:
        raise ValueError(f"Invalid input shape: {x.shape}")
    batch_size, c_patches, frames, height, width = x.shape
    channels = c_patches // (patch_size * patch_size)
    x = x.view(batch_size, channels, patch_size, patch_size, frames, height, width)
    x = x.permute(0, 1, 4, 5, 3, 6, 2).contiguous()
    x = x.view(batch_size, channels, frames, height * patch_size, width * patch_size)
    return x


class QwenImage21SpatialConv3d(SpatialParallelConv2d):
    def forward(self, x, cache_x=None):
        assert cache_x is None
        return super().forward(x.squeeze(2)).unsqueeze(2)


def enable_qwen21_spatial_decode(module):
    for name, child in list(module.named_children()):
        if isinstance(child, QwenImage21AttentionBlock):
            # attention needs the full image; its pointwise projections stay local
            child.spatial_parallel = True
        elif isinstance(child, nn.Conv2d):
            causal = isinstance(child, QwenImage21CausalConv3d)
            conv_cls = QwenImage21SpatialConv3d if causal else SpatialParallelConv2d
            padding = (
                (child._padding[2], child._padding[0]) if causal else child.padding
            )
            conv = conv_cls(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                stride=child.stride,
                padding=padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
            )
            conv.weight, conv.bias = child.weight, child.bias
            setattr(module, name, conv)
        else:
            enable_qwen21_spatial_decode(child)


def _is_fp8_conv3x3(conv) -> bool:
    return (
        isinstance(conv, nn.Conv2d)
        and conv.kernel_size == (3, 3)
        and conv.stride == (1, 1)
        and conv.padding == (1, 1)
        and conv.bias is not None
    )


def _fp8_residual_eligible(block: QwenImage21ResidualBlock) -> bool:
    norms = (block.norm1, block.norm2)
    return (
        _is_fp8_conv3x3(block.conv1)
        and _is_fp8_conv3x3(block.conv2)
        and all(
            n.channel_first and isinstance(n.bias, (int, float)) and n.bias == 0
            for n in norms
        )
        # the norm kernel holds a whole channel row in registers
        and max(block.in_dim, block.out_dim) <= NHWC_NORM_MAX_CHANNELS
    )


def _fp8_upsample_eligible(resample: QwenImage21Resample) -> bool:
    return resample.mode in ("upsample2d", "upsample3d") and _is_fp8_conv3x3(
        resample.resample[1]
    )


class AutoencoderKLQwenImage21(ParallelTiledVAE):
    layer_names = [
        *ParallelTiledVAE.layer_names,
        "encoder.mid_block.resnets",
        "encoder.mid_block.attentions",
        "decoder.mid_block.resnets",
        "decoder.mid_block.attentions",
    ]

    def __init__(self, config: QwenImage21VAEConfig, **kwargs):
        super().__init__(config, **kwargs)
        ac = config.arch_config
        shared = dict(
            z_dim=ac.z_dim,
            dim_mult=list(ac.dim_mult),
            num_res_blocks=ac.num_res_blocks,
            attn_scales=list(ac.attn_scales),
            dropout=ac.dropout,
            is_residual=ac.is_residual,
        )
        if config.load_encoder:
            self.encoder = QwenImage21Encoder3d(
                in_channels=ac.in_channels,
                dim=ac.base_dim,
                **dict(shared, z_dim=ac.z_dim * 2),
                temperal_downsample=list(ac.temperal_downsample),
            )
            self.quant_conv = QwenImage21CausalConv3d(ac.z_dim * 2, ac.z_dim * 2, 1)
        if config.load_decoder:
            self.post_quant_conv = QwenImage21CausalConv3d(ac.z_dim, ac.z_dim, 1)
            self.decoder = QwenImage21Decoder3d(
                dim=ac.decoder_base_dim or ac.base_dim,
                **shared,
                temperal_upsample=list(ac.temperal_downsample)[::-1],
                out_channels=ac.out_channels,
            )
        self.spatial_parallel = (
            config.load_decoder and can_install_spatial_shard_parallel_decode(config)
        )
        if self.spatial_parallel:
            enable_qwen21_spatial_decode(self.decoder)
        self._fp8_convs_prepared = False
        self._fp8_convs_warned = False

    def _encode(self, x):
        if x.shape[2] != 1:
            raise ValueError("Qwen-Image 2.1 VAE expects one image frame")
        if self.config.patch_size is not None:
            x = _patchify(x, self.config.patch_size)
        return self.quant_conv(self.encoder(x))

    def _use_channels_last(self) -> bool:
        """Decoder weights in channels_last, converted on first use (and after any reload)."""
        if not (
            envs.SGLANG_ENABLE_QWEN_IMAGE21_VAE_CHANNELS_LAST.get()
            or self._use_fp8_convs()
        ):
            return False
        convs = [
            m
            for m in (self.post_quant_conv, *self.decoder.modules())
            if isinstance(m, nn.Conv2d)
        ]
        if not convs[0].weight.is_contiguous(memory_format=torch.channels_last):
            for conv in convs:
                conv.weight.data = conv.weight.data.contiguous(
                    memory_format=torch.channels_last
                )
        return True

    def _use_fp8_convs(self) -> bool:
        if not envs.SGLANG_ENABLE_QWEN_IMAGE21_VAE_FP8_CONV.get():
            return False
        if not fp8_conv_cudnn.is_available():
            if not self._fp8_convs_warned:
                logger.warning(
                    "SGLANG_ENABLE_QWEN_IMAGE21_VAE_FP8_CONV needs the cuDNN frontend "
                    "Python package (cudnn); decoding with bf16 convs"
                )
                self._fp8_convs_warned = True
            return False
        return True

    def _prepare_fp8_convs(self):
        """FP8 copies of the decoder's 3x3 residual and upsample convs (weights stay bf16 too)."""
        for module in self.decoder.modules():
            if isinstance(module, QwenImage21ResidualBlock) and _fp8_residual_eligible(
                module
            ):
                module.prepare_fp8_convs()
            elif isinstance(module, QwenImage21Resample) and _fp8_upsample_eligible(
                module
            ):
                module.fp8_conv = fp8_conv_cudnn.Fp8Conv3x3(module.resample[1])
        self._fp8_convs_prepared = True

    def _decode(self, z):
        if z.shape[2] != 1:
            raise ValueError("Qwen-Image 2.1 VAE expects one latent frame")
        parallel = self.spatial_parallel and should_run_spatial_shard_parallel_decode(
            self.config, z
        )
        if not parallel and self._use_channels_last():
            z = z.contiguous(memory_format=torch.channels_last_3d)
            if self._use_fp8_convs() and not self._fp8_convs_prepared:
                self._prepare_fp8_convs()
        z = self.post_quant_conv(z)
        if parallel:
            z, expected_height = split_height_for_parallel_decode(
                z,
                expected_height=z.shape[-2] * self.spatial_compression_ratio,
                world_size=get_decode_parallel_world_size(),
                rank=get_decode_parallel_rank(),
            )
            x = self.decoder(z, first_chunk=True)
        else:
            with disable_spatial_parallel_decode():
                x = self.decoder(z, first_chunk=True)
        if self.config.patch_size is not None:
            x = _unpatchify(x, self.config.patch_size)
        if parallel:
            x = gather_and_trim_height(x, expected_height)
        return x.clamp(-1, 1)


EntryClass = AutoencoderKLQwenImage21
