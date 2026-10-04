"""On-disk checkpoint identity: sha256 of every weight file, checked against a pinned manifest.

The served checkpoint is derived from the official BF16 release (offline FP8
conversion), so its bytes have no Hugging Face hash to match; `gate
pin-checkpoint` records them once, with where they came from, and every run
checks the files it loads still hash to the pin. Hashes are cached by
(size, mtime) so a run re-reads only files that changed.
"""

import glob
import hashlib
import json
import os
from typing import Dict

from gate import config

_CACHE = os.path.join(config.REFERENCE_DIR, "checkpoint_sha256_cache.json")
PIN_PATH = os.path.join(config.REFERENCE_DIR, "checkpoint_pin.json")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def file_hashes(model_dir: str = config.MODEL_DIR) -> Dict[str, str]:
    """Relative path -> sha256 of every safetensors file under `model_dir` (symlinks followed)."""
    cache = json.load(open(_CACHE)) if os.path.exists(_CACHE) else {}
    out = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "**", "*.safetensors"), recursive=True)):
        real = os.path.realpath(path)
        st = os.stat(real)
        entry = cache.get(real)
        if not entry or entry["size"] != st.st_size or entry["mtime"] != st.st_mtime:
            entry = {"size": st.st_size, "mtime": st.st_mtime, "sha256": _sha256(real)}
            cache[real] = entry
        out[os.path.relpath(path, model_dir)] = entry["sha256"]
    os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
    with open(_CACHE, "w") as f:
        json.dump(cache, f, indent=1)
    return out


def pin(source: str) -> Dict:
    """Records the current files as the pinned checkpoint; `source` says how they were made."""
    files = file_hashes()
    if not files:
        raise RuntimeError(f"no safetensors under {config.MODEL_DIR}")
    rec = {"model_dir": config.MODEL_DIR, "base_model": config.MODEL_ID, "base_revision": config.MODEL_REVISION,
           "source": source, "files": files}
    os.makedirs(os.path.dirname(PIN_PATH), exist_ok=True)
    with open(PIN_PATH, "w") as f:
        json.dump(rec, f, indent=1)
    return rec


def compare(files: Dict[str, str], pinned: Dict[str, str]) -> Dict:
    mismatched = sorted(k for k in set(files) | set(pinned) if files.get(k) != pinned.get(k))
    return {"matches_pin": bool(files) and not mismatched, "mismatched": mismatched}


def verify() -> Dict:
    files = file_hashes()
    if not os.path.exists(PIN_PATH):
        return {"files": files, "matches_pin": False, "mismatched": [], "error": "no pin; run gate pin-checkpoint"}
    with open(PIN_PATH) as f:
        pinned = json.load(f)
    return {"files": files, "pin_source": pinned.get("source"), **compare(files, pinned["files"])}
