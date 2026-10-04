"""Convert the official google/gemma-4-26B-A4B-it BF16 checkpoint to FP8 on CPU.

Format: compressed-tensors "float-quantized", the scheme of
RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic (llm-compressor FP8_DYNAMIC):
  - weights: FP8 E4M3, one symmetric scale per output channel (min-max, no
    calibration data), scale stored as BF16 [out, 1];
  - activations: dynamic per-token FP8, computed at runtime;
  - quantized: every language-model Linear (q/k/v/o, dense MLP, all experts);
  - kept BF16: embeddings (tied LM head), router.proj, norms, scalars, vision.

Fused HF expert tensors are split per expert, the layout SGLang's Gemma4
loaders accept for compressed-tensors (experts.<e>.{gate,up,down}_proj.*):
  experts.gate_up_proj [E, 2I, H] -> gate = [:, :I], up = [:, I:]
  experts.down_proj    [E, H, I]

Why not 128x128 block FP8: moe_intermediate_size 704 and intermediate_size
2112 are not multiples of 128.

With --remote-repo/--revision the tensor bytes are range-read from the Hub
instead of <src>/*.safetensors (<src> then holds only config, index and
tokenizer files), so a host without disk for the 51.6 GB BF16 copy can convert.

Output: <out>/shards/*.safetensors (language model) + vision.safetensors, and
two model dirs that symlink them:
  <out>/text  Gemma4ForCausalLM (no vision tower is built or loaded)
  <out>/mm    Gemma4ForConditionalGeneration (same tensors + BF16 vision)
"""

import argparse
import json
import os
import re
import shutil

import struct
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448
SHARD_BYTES = 4 << 30

LINEAR_RE = re.compile(
    r"^model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)


def quantize_per_channel(w: torch.Tensor):
    """[out, in] BF16 -> (FP8 [out, in], BF16 scale [out, 1]).

    The scale is rounded to BF16 first and the weight is divided by that
    rounded scale, so dequant (q * scale) uses exactly the stored scale.
    """
    w32 = w.float()
    amax = w32.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = (amax / FP8_MAX).to(torch.bfloat16)
    q = (w32 / scale.float()).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale


def convert_tensor(name: str, t: torch.Tensor):
    """Yield (name, tensor) pairs for the output checkpoint."""
    if LINEAR_RE.match(name):
        q, s = quantize_per_channel(t)
        base = name[: -len(".weight")]
        yield f"{base}.weight", q
        yield f"{base}.weight_scale", s
        return
    if name.endswith(".experts.gate_up_proj"):
        base = name[: -len("gate_up_proj")]
        num_experts, two_i, _ = t.shape
        inter = two_i // 2
        for e in range(num_experts):
            for proj, part in (("gate_proj", t[e, :inter]), ("up_proj", t[e, inter:])):
                q, s = quantize_per_channel(part)
                yield f"{base}{e}.{proj}.weight", q
                yield f"{base}{e}.{proj}.weight_scale", s
        return
    if name.endswith(".experts.down_proj"):
        base = name[: -len("down_proj")]
        for e in range(t.shape[0]):
            q, s = quantize_per_channel(t[e])
            yield f"{base}{e}.down_proj.weight", q
            yield f"{base}{e}.down_proj.weight_scale", s
        return
    yield name, t


def quant_config(ignore):
    return {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "format": "float-quantized",
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "symmetric": True,
                    "strategy": "channel",
                    "dynamic": False,
                    "group_size": None,
                    "block_structure": None,
                    "actorder": None,
                    "observer": "memoryless_minmax",
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
                    "symmetric": True,
                    "strategy": "token",
                    "dynamic": True,
                    "group_size": None,
                    "block_structure": None,
                    "actorder": None,
                    "observer": None,
                },
                "output_activations": None,
            }
        },
        "ignore": ignore,
    }


_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}
_RANGE_BYTES = 128 << 20


class RemoteSafetensors:
    """safe_open-like reader that range-reads one .safetensors file from the Hub."""

    def __init__(self, url: str, headers: dict):
        import httpx

        self.url, self.headers = url, headers
        self.client = httpx.Client(follow_redirects=True, timeout=300)
        (n,) = struct.unpack("<Q", self._get(0, 8))
        self.header = json.loads(self._get(8, 8 + n))
        self.header.pop("__metadata__", None)
        self.data_start = 8 + n

    def _get(self, lo: int, hi: int) -> bytes:
        for attempt in range(5):
            try:
                r = self.client.get(self.url, headers={**self.headers, "Range": f"bytes={lo}-{hi - 1}"})
                r.raise_for_status()
                assert len(r.content) == hi - lo, (len(r.content), hi - lo)
                return r.content
            except Exception:
                if attempt == 4:
                    raise

    def get_tensor(self, name: str) -> torch.Tensor:
        meta = self.header[name]
        lo, hi = (self.data_start + o for o in meta["data_offsets"])
        cuts = list(range(lo, hi, _RANGE_BYTES)) + [hi]
        with ThreadPoolExecutor(8) as ex:
            parts = list(ex.map(lambda i: self._get(cuts[i], cuts[i + 1]), range(len(cuts) - 1)))
        buf = bytearray(b"".join(parts))
        return torch.frombuffer(buf, dtype=_DTYPES[meta["dtype"]]).reshape(meta["shape"])


@contextmanager
def open_source(src_dir: str, fname: str, remote_repo, revision):
    if remote_repo is None:
        with safe_open(os.path.join(src_dir, fname), framework="pt") as f:
            yield f
        return
    from huggingface_hub import get_token, hf_hub_url

    token = get_token()
    yield RemoteSafetensors(
        hf_hub_url(remote_repo, fname, revision=revision),
        {"Authorization": f"Bearer {token}"} if token else {},
    )


class ShardWriter:
    def __init__(self, out_dir: str, prefix: str):
        self.out_dir, self.prefix = out_dir, prefix
        self.buf, self.nbytes, self.files, self.weight_map = {}, 0, [], {}

    def add(self, name: str, t: torch.Tensor):
        self.buf[name] = t.contiguous()
        self.nbytes += t.numel() * t.element_size()
        if self.nbytes >= SHARD_BYTES:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        fname = f"{self.prefix}-{len(self.files):05d}.safetensors"
        save_file(self.buf, os.path.join(self.out_dir, fname), metadata={"format": "pt"})
        for k in self.buf:
            self.weight_map[k] = fname
        print(f"wrote {fname} {self.nbytes / 2**30:.2f} GiB", flush=True)
        self.files.append(fname)
        self.buf, self.nbytes = {}, 0


def write_model_dir(src, out, kind, shard_dir, weight_map, config):
    d = os.path.join(out, kind)
    os.makedirs(d, exist_ok=True)
    for fname in sorted(set(weight_map.values())):
        link = os.path.join(d, fname)
        if not os.path.lexists(link):
            os.symlink(os.path.join("..", os.path.basename(shard_dir), fname), link)
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": dict(sorted(weight_map.items()))}, f, indent=1)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    for aux in (
        "chat_template.jinja",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
    ):
        if os.path.exists(os.path.join(src, aux)):
            shutil.copy(os.path.join(src, aux), d)


def sglang_text_config(text_config: dict) -> dict:
    """Gemma-4 text config with full/sliding attention dims in SGLang's convention.

    Gemma-4 names sliding-layer dims plainly and full-layer dims `global_*`;
    SGLang wants base = full, `swa_*` = sliding. hf_transformers/config.py
    only rewrites `model_type == "gemma4"` (multimodal) configs, so a
    `gemma4_text` config must arrive already rewritten.
    """
    cfg = dict(text_config)
    cfg.update(
        swa_head_dim=text_config["head_dim"],
        swa_v_head_dim=text_config["head_dim"],
        swa_num_key_value_heads=text_config["num_key_value_heads"],
        head_dim=text_config["global_head_dim"],
        v_head_dim=text_config["global_head_dim"],
        num_key_value_heads=text_config["num_global_key_value_heads"],
    )
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--remote-repo", default=None)
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()

    shard_dir = os.path.join(args.out, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    with open(os.path.join(args.src, "model.safetensors.index.json")) as f:
        src_map = json.load(f)["weight_map"]
    with open(os.path.join(args.src, "config.json")) as f:
        src_config = json.load(f)

    text = ShardWriter(shard_dir, "text")
    vision = ShardWriter(shard_dir, "vision")
    by_file = {}
    for name, fname in src_map.items():
        by_file.setdefault(fname, []).append(name)
    for fname, names in sorted(by_file.items()):
        with open_source(args.src, fname, args.remote_repo, args.revision) as f:
            for name in sorted(names):
                if name == "lm_head.weight":
                    continue  # tied to embed_tokens
                writer = text if name.startswith("model.language_model.") else vision
                for out_name, t in convert_tensor(name, f.get_tensor(name)):
                    writer.add(out_name, t)
    text.flush()
    vision.flush()

    ignore_text = ["lm_head", "re:.*router.*"]
    text_config = sglang_text_config(src_config["text_config"])
    text_config.update(
        architectures=["Gemma4ForCausalLM"],
        torch_dtype="bfloat16",
        quantization_config=quant_config(ignore_text),
    )
    write_model_dir(args.src, args.out, "text", shard_dir, text.weight_map, text_config)

    mm_config = dict(src_config)
    mm_config["quantization_config"] = quant_config(
        ignore_text + ["re:.*vision_tower.*", "re:.*embed_vision.*"]
    )
    write_model_dir(
        args.src, args.out, "mm", shard_dir, {**text.weight_map, **vision.weight_map}, mm_config
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
