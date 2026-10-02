#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""Copies decode-mk's Qwen3.8 kernels into python/sglang/kernels/decode_mk/.

Usage: scripts/sync_decode_mk.py <decode-mk checkout> <commit>

Reads every file from <commit> of the decode-mk checkout (its working tree is
not read) and rewrites the vendored copy from it:

- The CUDA sources and headers are copied unchanged.
- The Python modules are copied with their `s2mk` imports pointed at
  `sglang.kernels.decode_mk`.
- ops.cpp keeps only what the vendored kernels define: the functions and
  bindings of the kernels of decode-mk's other models (S2 Pro, Qwen3-Omni)
  are cut, named in _DROPPED_* below. Those models' headers are copied, for
  the shapes and constants ops.cpp shares with them.

_ext.py is SGLang's own and is not synced. VENDORED records the commit and
each vendored file's SHA-256, which the vendored-tree test checks, so an edit
made here instead of in decode-mk fails the test until the next sync.

A kernel update in decode-mk reaches SGLang as one rerun of this script at the
new commit. A new decode-mk binding whose kernel lives outside the copied
sources fails the extension's load with an undefined symbol; add its name to
_DROPPED_FUNCTIONS and _DROPPED_BINDINGS then.
"""

import hashlib
import re
import subprocess
import sys
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / "python/sglang/kernels/decode_mk"

PYTHON = [
    "barrier.py",
    "gdn.py",
    "gemv.py",
    "int4.py",
    "qwen38_checkpoint.py",
    "qwen38_decode.py",
    "qwen38_layer.py",
    "qwen38_mtp.py",
    "qwen38_prefill.py",
    "qwen38_spec.py",
    "qwen38_verify.py",
    "thinker_attention.py",
]

CSRC = [
    "barrier.cuh",
    "barrier.h",
    "barrier_probe.cu",
    "decode.h",
    "decoder_layer.cuh",
    "gdn.cu",
    "gdn.h",
    "gdn_block.cuh",
    "gemv.cu",
    "gemv.h",
    "gemv_core.cuh",
    "int4_gemv.cu",
    "int4_gemv.h",
    "int4_gemv_core.cuh",
    "int4_mma_core.cuh",
    "kv_cache.h",
    "layer.h",
    "prefill_dense.cuh",
    "qwen38_decode.cu",
    "qwen38_decode.h",
    "qwen38_layer.cu",
    "qwen38_layer.cuh",
    "qwen38_layer.h",
    "qwen38_lm_head.cuh",
    "qwen38_mtp.cu",
    "qwen38_mtp.h",
    "qwen38_prefetch.cuh",
    "qwen38_prefill.cu",
    "qwen38_prefill.h",
    "qwen38_verify.cu",
    "qwen3omni_cp.h",
    "sampler.h",
    "slow_ar.h",
    "talker_decode.h",
    "thinker_attention.cu",
    "thinker_attention.h",
    "thinker_decode.h",
    "thinker_dims.h",
    "thinker_layer.cuh",
    "thinker_moe.h",
    "thinker_prefill.h",
    "weight.h",
]

# What ops.cpp binds from the kernels of decode-mk's other models.
_DROPPED_FUNCTIONS = {
    "BuildFastQkvTable",
    "PinFor",
    "RunCodePredictor",
    "RunCodeSamplerProbe",
    "RunDecode",
    "RunSamplerProbe",
    "RunSlowAr",
    "RunTalkerDecode",
    "RunThinkerDecode",
    "RunThinkerMoe",
    "RunThinkerPrefill",
    "Sampling",
    "TalkerLayerParamsOf",
    "ThinkerMoeParamsOf",
}
_DROPPED_BINDINGS = {
    "build_fast_qkv_table",
    "run_code_predictor",
    "run_code_sampler_probe",
    "run_decode",
    "run_sampler_probe",
    "run_slow_ar",
    "run_talker_decode",
    "run_thinker_decode",
    "run_thinker_moe",
    "run_thinker_prefill",
    "talker_layer_params",
    "thinker_moe_params",
}

_IMPORT = re.compile(r"^(from|import) s2mk\b", re.MULTILINE)
_FUNCTION = re.compile(r"^[A-Za-z][\w:<>, *&]*?\b(\w+)\(")


def _show(checkout: Path, commit: str, path: str) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), "show", f"{commit}:{path}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def rewrite_imports(source: str) -> str:
    return _IMPORT.sub(r"\1 sglang.kernels.decode_mk", source)


def slice_ops(source: str) -> str:
    """ops.cpp without the dropped functions and bindings.

    A function runs from the comment and template lines above its signature
    to the next `}` in column 0; a binding from its `m.def(` to the line
    ending its statement."""
    lines = source.splitlines(keepends=True)
    out, pending = [], []
    i = 0
    while i < len(lines):
        line = lines[i]
        binding = re.match(r'\s+m\.def\("(\w+)"', line)
        if binding and binding.group(1) in _DROPPED_BINDINGS:
            while not lines[i].rstrip().endswith(");"):
                i += 1
            i += 1
            continue
        if line.startswith("//") or line.startswith("template"):
            pending.append(line)
            i += 1
            continue
        function = _FUNCTION.match(line)
        if function and function.group(1) in _DROPPED_FUNCTIONS:
            pending = []
            while not lines[i].startswith("}"):
                i += 1
            i += 1
            # The blank line after it.
            if i < len(lines) and not lines[i].strip():
                i += 1
            continue
        out.extend(pending)
        pending = []
        out.append(line)
        i += 1
    out.extend(pending)
    return "".join(out)


def vendored_files() -> list[str]:
    return [*PYTHON, *(f"csrc/{name}" for name in CSRC), "csrc/ops.cpp"]


def manifest(commit: str) -> str:
    lines = [f"decode-mk {commit}"]
    for name in vendored_files():
        digest = hashlib.sha256((DEST / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}")
    return "\n".join(lines) + "\n"


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    checkout, commit = Path(sys.argv[1]), sys.argv[2]
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", f"{commit}^{{commit}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (DEST / "csrc").mkdir(parents=True, exist_ok=True)
    for name in PYTHON:
        (DEST / name).write_text(
            rewrite_imports(_show(checkout, commit, f"s2mk/{name}"))
        )
    for name in CSRC:
        (DEST / "csrc" / name).write_text(_show(checkout, commit, f"s2mk/csrc/{name}"))
    (DEST / "csrc/ops.cpp").write_text(
        slice_ops(_show(checkout, commit, "s2mk/csrc/ops.cpp"))
    )
    (DEST / "VENDORED").write_text(manifest(commit))
    print(f"vendored decode-mk {commit} into {DEST}")


if __name__ == "__main__":
    main()
