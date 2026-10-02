# Copyright 2026 Fractalyze Inc. All rights reserved.
"""Qwen3.8-27B's language model from a Hugging Face checkpoint directory, as
Qwen38Weights with every tensor as the checkpoint stores it.

The decode megakernel runs the compressed-tensors int4 checkpoint
(cyankiwi/Qwen3.8-27B-AWQ-INT4): asymmetric W4A16 in groups of 32, with the
linear-attention gates (in_proj_b, in_proj_a), layer 0's linear-attention
out_proj, the embedding, the LM head and the MTP head in bf16. The references also read
the bf16 checkpoint (Qwen/Qwen3.8-27B), every projection bf16.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from sglang.kernels.decode_mk import int4
from sglang.kernels.decode_mk.gdn import CONV_DIM, CONV_WIDTH, GdnWeights
from sglang.kernels.decode_mk.qwen38_decode import LAYERS, LinearLayerWeights, Qwen38Weights, is_full
from sglang.kernels.decode_mk.qwen38_layer import LayerWeights, MlpWeights
from sglang.kernels.decode_mk.qwen38_mtp import MtpWeights

ENV = "S2MK_QWEN38_CHECKPOINT"
_PREFIX = "model.language_model."


class Checkpoint:
    """A checkpoint directory's tensors by name, read onto `device`."""

    def __init__(self, path: str | Path, device: str = "cuda") -> None:
        self.path = Path(path)
        self.device = device
        self.config = json.loads((self.path / "config.json").read_text())
        index = json.loads((self.path / "model.safetensors.index.json").read_text())
        self._files = index["weight_map"]
        quant = self.config.get("quantization_config")
        if quant is not None:
            weights = quant["config_groups"]["group_0"]["weights"]
            if (weights["num_bits"], weights["group_size"], weights["symmetric"]) != (4, 32, False):
                raise ValueError(f"{path}: expected asymmetric int4 in groups of 32, got "
                                 f"{weights}")
        text = self.config.get("text_config", self.config)
        self.eps = text["rms_norm_eps"]

    def has(self, name: str) -> bool:
        return name in self._files

    def get(self, name: str) -> torch.Tensor:
        with safe_open(self.path / self._files[name], "pt", device=self.device) as f:
            return f.get_tensor(name)

    def projection(self, name: str) -> int4.Projection:
        """Projection `name` (without `.weight`) as stored: int4, with its zero
        points in the kernels' layout, or bf16."""
        if not self.has(f"{name}.weight_packed"):
            return self.get(f"{name}.weight")
        packed = self.get(f"{name}.weight_packed")
        rows = packed.shape[0]
        if self.get(f"{name}.weight_shape").tolist() != [rows, packed.shape[1] * int4.PER_WORD]:
            raise ValueError(f"{name}: packed shape does not match weight_shape")
        zeros = int4.zeros_from_checkpoint(self.get(f"{name}.weight_zero_point"))
        return packed, self.get(f"{name}.weight_scale"), zeros


def _cat(*weights: int4.Projection) -> int4.Projection:
    """The rows of `weights`, all in one format, stacked; each int4 row carries
    its own scales and zero points, so stacking is exact."""
    if isinstance(weights[0], torch.Tensor):
        return torch.cat(weights)
    return tuple(torch.cat(parts).contiguous() for parts in zip(*weights))


def _mlp(ckpt: Checkpoint, prefix: str) -> MlpWeights:
    return MlpWeights.of(ckpt.get(f"{prefix}post_attention_layernorm.weight"),
                         ckpt.projection(f"{prefix}mlp.gate_proj"),
                         ckpt.projection(f"{prefix}mlp.up_proj"),
                         ckpt.projection(f"{prefix}mlp.down_proj"), ckpt.eps)


def _linear_layer(ckpt: Checkpoint, prefix: str) -> LinearLayerWeights:
    a = f"{prefix}linear_attn."
    attention = GdnWeights(
        norm=ckpt.get(f"{prefix}input_layernorm.weight"),
        in_proj=_cat(ckpt.projection(f"{a}in_proj_qkv"), ckpt.projection(f"{a}in_proj_z")),
        gates=_cat(ckpt.projection(f"{a}in_proj_b"), ckpt.projection(f"{a}in_proj_a")),
        conv=ckpt.get(f"{a}conv1d.weight").view(CONV_DIM, CONV_WIDTH),
        a_log=ckpt.get(f"{a}A_log").float(), dt_bias=ckpt.get(f"{a}dt_bias").float(),
        out_norm=ckpt.get(f"{a}norm.weight"), out_proj=ckpt.projection(f"{a}out_proj"),
        eps=ckpt.eps)
    return LinearLayerWeights(attention, _mlp(ckpt, prefix))


def _full_layer(ckpt: Checkpoint, prefix: str) -> LayerWeights:
    a = f"{prefix}self_attn."
    return LayerWeights(
        input_norm=ckpt.get(f"{prefix}input_layernorm.weight"),
        q_proj=ckpt.projection(f"{a}q_proj"), k_proj=ckpt.projection(f"{a}k_proj"),
        v_proj=ckpt.projection(f"{a}v_proj"), q_norm=ckpt.get(f"{a}q_norm.weight"),
        k_norm=ckpt.get(f"{a}k_norm.weight"), o_proj=ckpt.projection(f"{a}o_proj"),
        mlp=_mlp(ckpt, prefix), eps=ckpt.eps)


def load_layer(ckpt: Checkpoint, i: int) -> LinearLayerWeights | LayerWeights:
    """Layer i of the checkpoint."""
    prefix = f"{_PREFIX}layers.{i}."
    return _full_layer(ckpt, prefix) if is_full(i) else _linear_layer(ckpt, prefix)


def load(path: str | Path, num_layers: int = LAYERS, device: str = "cuda") -> Qwen38Weights:
    """The checkpoint at `path`, its first `num_layers` layers, on `device`."""
    ckpt = Checkpoint(path, device)
    return Qwen38Weights(embed=ckpt.get(f"{_PREFIX}embed_tokens.weight"),
                         layers=[load_layer(ckpt, i) for i in range(num_layers)],
                         final_norm=ckpt.get(f"{_PREFIX}norm.weight"),
                         lm_head=ckpt.get("lm_head.weight"), eps=ckpt.eps)


def load_mtp(path: str | Path, device: str = "cuda") -> MtpWeights:
    """The checkpoint's MTP head (`mtp.*`), on `device`."""
    ckpt = Checkpoint(path, device)
    return MtpWeights(embed_norm=ckpt.get("mtp.pre_fc_norm_embedding.weight"),
                      hidden_norm=ckpt.get("mtp.pre_fc_norm_hidden.weight"),
                      fc=ckpt.get("mtp.fc.weight"),
                      layer=_full_layer(ckpt, "mtp.layers.0."),
                      final_norm=ckpt.get("mtp.norm.weight"), eps=ckpt.eps)
