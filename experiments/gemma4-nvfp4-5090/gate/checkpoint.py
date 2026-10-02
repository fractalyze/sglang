"""On-disk checkpoint identity: sha256 of every weight file, checked against Hugging Face's LFS hashes.

The in-memory weight hash (/weights_checker) has no comparable form for the
NVFP4 fused-MoE method and Gemma4 has no get_weights_by_name, so the gate pins
the bytes the server loads instead; the server-arg diff refuses undeclared
load-format or quantization changes. Hashes are cached by (size, mtime).
"""

import glob
import hashlib
import json
import os
from typing import Dict

import requests

from gate import config

_CACHE = os.path.join(config.REFERENCE_DIR, "checkpoint_sha256.json")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _hf_lfs_hashes() -> Dict[str, str]:
    url = f"https://huggingface.co/api/models/{config.MODEL_ID}/revision/{config.MODEL_REVISION}?blobs=true"
    sib = requests.get(url, timeout=60).json()["siblings"]
    return {s["rfilename"]: s["lfs"]["sha256"] for s in sib if s.get("lfs")}


def verify() -> Dict:
    cache = json.load(open(_CACHE)) if os.path.exists(_CACHE) else {}
    files = {}
    for path in sorted(glob.glob(os.path.join(config.MODEL_DIR, "*.safetensors"))):
        st = os.stat(path)
        key = os.path.basename(path)
        entry = cache.get(key)
        if not entry or entry["size"] != st.st_size or entry["mtime"] != st.st_mtime:
            entry = {"size": st.st_size, "mtime": st.st_mtime, "sha256": _sha256(path)}
        files[key] = entry
    if "hf_lfs" not in cache:
        cache["hf_lfs"] = _hf_lfs_hashes()
    mismatched = [k for k, e in files.items() if cache["hf_lfs"].get(k) != e["sha256"]]
    cache.update(files)
    os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
    with open(_CACHE, "w") as f:
        json.dump(cache, f, indent=1)
    return {"files": {k: e["sha256"] for k, e in files.items()}, "matches_hf_revision": not mismatched,
            "mismatched": mismatched}
