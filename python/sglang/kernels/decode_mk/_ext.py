# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""JIT-builds the decode-mk CUDA extension against the running torch.

The extension must share a process with the torch that loads it, so it is
compiled with the CUDA toolkit that torch was built against. When `CUDA_HOME`
is unset, that toolkit is the `nvidia-cuda-nvcc` wheel installed beside torch
(`site-packages/nvidia/cu<major>`).

SGLang's counterpart of decode-mk's s2mk/_ext.py, building only the vendored
sources, into SGLANG_CACHE_DIR, without leaving its CUDA_HOME and
TORCH_CUDA_ARCH_LIST set for the process's other JIT builds.
"""

import functools
import os
from pathlib import Path
from types import ModuleType

from sglang.srt.environ import envs

_CSRC = Path(__file__).parent / "csrc"
_ARCH = "12.0a"
_SOURCES = (
    "ops.cpp",
    "gemv.cu",
    "int4_gemv.cu",
    "barrier_probe.cu",
    "gdn.cu",
    "thinker_attention.cu",
    "qwen38_layer.cu",
    "qwen38_decode.cu",
    "qwen38_prefill.cu",
    "qwen38_verify.cu",
    "qwen38_mtp.cu",
)


def _cuda_home() -> Path:
    if "CUDA_HOME" in os.environ:
        return Path(os.environ["CUDA_HOME"])
    import nvidia
    import torch

    major = torch.version.cuda.split(".")[0]
    for root in nvidia.__path__:
        home = Path(root) / f"cu{major}"
        if (home / "bin" / "nvcc").exists():
            return home
    raise RuntimeError(
        f"no CUDA {major} nvcc found; install nvidia-cuda-nvcc for CUDA {major} "
        "or set CUDA_HOME"
    )


def _link_dir(cuda_home: Path, build_dir: Path) -> Path:
    """Returns a directory holding the unversioned libcudart.so the linker needs.

    The toolkit wheels ship only libcudart.so.<major>, and the extension build
    links with -lcudart.
    """
    lib = cuda_home / "lib"
    if (lib / "libcudart.so").exists():
        return lib
    link_dir = build_dir / "lib"
    link_dir.mkdir(parents=True, exist_ok=True)
    versioned = sorted(lib.glob("libcudart.so.*"))[0]
    link = link_dir / "libcudart.so"
    if not link.exists():
        link.symlink_to(versioned)
    return link_dir


def _build_dir() -> Path:
    """One build directory per torch build: an extension built against another
    torch in the same directory would load, and fail on its ABI."""
    import torch

    cache = Path(os.path.expanduser(envs.SGLANG_CACHE_DIR.get()))
    return cache / "decode_mk" / f"ext-{torch.__version__}"


@functools.cache
def load() -> ModuleType:
    cuda_home = _cuda_home()
    build_dir = _build_dir()
    build_dir.mkdir(parents=True, exist_ok=True)
    saved = {
        name: os.environ.get(name) for name in ("CUDA_HOME", "TORCH_CUDA_ARCH_LIST")
    }
    os.environ["CUDA_HOME"] = str(cuda_home)
    os.environ["TORCH_CUDA_ARCH_LIST"] = _ARCH
    try:
        # cpp_extension resolves CUDA_HOME at import, so import it only now.
        from torch.utils import cpp_extension

        return cpp_extension.load(
            name="decode_mk_ext",
            sources=[str(_CSRC / name) for name in _SOURCES],
            extra_cflags=["-O3"],
            # -lineinfo only maps SASS to source lines for ncu; codegen is unchanged.
            extra_cuda_cflags=["-O3", "-lineinfo"],
            extra_ldflags=[f"-L{_link_dir(cuda_home, build_dir)}"],
            build_directory=str(build_dir),
        )
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
