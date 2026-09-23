#!/usr/bin/env python
"""Build the portable copies of the DPCache study artifacts, with provenance.

CPU only. Reads the raw run artifacts and writes publish copies plus a manifest.

Raw artifacts stay untouched: they are the record of what was actually measured.
A publish copy is only rewritten where a value is a path on the machine that
produced it -- those are relativised so the package means the same thing after a
clone. Rewriting changes the file's digest, so the manifest records BOTH the raw
digest and the publish digest for every file, and says which of the two it is.
Files needing no rewrite are copied byte for byte and keep their raw digest.

Usage:
  python prepare_publish_artifacts.py --raw-root /path/to/dpcache --out artifacts/
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path

SCHEMA = "dpcache-publish-manifest-v1"

# Every value under these keys is a path on the producing machine. The publish
# copy keeps the basename (which run, which file) and drops the rest.
PATH_KEYS = (
    "calibration_run",
    "signature_run",
    "reference_run",
    "validation_run",
    "run",
    "scores",
    "corpus",
    "heldout_corpus",
    "source_path",
    "path",
)
# The checkpoint is identified by its revision, which every artifact already
# carries; the local cache path it was read from is not part of the result.
MODEL_PATH_KEYS = ("model_path",)
# Free-form provenance strings that embed a local checkout path.
TEXT_KEYS = ("checkout", "report")
# These carry a directory plus a filename that both matter.
PATH_KEYS_KEEP_TWO = ("schedule", "schedule_path")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def relativise(value: str, keep: int) -> str:
    parts = Path(value).parts
    return "/".join(parts[-keep:]) if len(parts) > keep else value


def strip_local_prefix(value: str) -> str:
    """Drop the leading absolute checkout path from a free-form provenance string."""
    out = []
    for token in value.split():
        out.append(relativise(token, 2) if token.startswith("/") else token)
    return " ".join(out)


def rewrite(node, changed: list[str], trail: str = ""):
    """Relativise machine paths in place, recording every field changed."""
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            where = f"{trail}.{key}" if trail else key
            if isinstance(value, str) and (value.startswith("/") or key in TEXT_KEYS):
                if key in MODEL_PATH_KEYS:
                    new = relativise(value, 1)
                elif key in PATH_KEYS:
                    new = relativise(value, 2 if "/runs/" in value else 1)
                elif key in PATH_KEYS_KEEP_TWO:
                    new = relativise(value, 2)
                elif key in TEXT_KEYS:
                    new = strip_local_prefix(value)
                else:
                    new = value
                if new != value:
                    changed.append(where)
                out[key] = new
            else:
                out[key] = rewrite(value, changed, where)
        return out
    if isinstance(node, list):
        return [rewrite(v, changed, f"{trail}[{i}]") for i, v in enumerate(node)]
    return node


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", required=True, help="the dpcache artifact root")
    ap.add_argument("--out", required=True, help="publish directory to write")
    ap.add_argument(
        "--raw-root-label",
        default="",
        help="what to call the raw root in the manifest (default: the path)",
    )
    args = ap.parse_args()

    raw = Path(args.raw_root).resolve()
    out = Path(args.out).resolve()

    # (source relative to raw root, destination relative to out)
    files = [
        ("comparators/RESULTS.json", "results/comparator-results.json"),
        (
            "comparators/selection-comparator-v1.json",
            "results/selection-comparator-v1.json",
        ),
        (
            "comparators/crossrun-exactness-audit.json",
            "results/crossrun-exactness-audit.json",
        ),
        ("comparators/corpus-comparator-v1.json", "corpus/corpus-comparator-v1.json"),
        ("corpus-v1.json", "corpus/corpus-v1.json"),
        ("selection-v1.json", "results/selection-dpcache-v1.json"),
        ("RESULTS.json", "results/dpcache-results.json"),
    ]
    for k in (12, 16, 20, 24, 28, 32, 36, 39):
        files.append((f"schedules-v2/K{k}.json", f"schedules/dp/K{k}.json"))
    for name in ("uniform-K12", "uniform-K20", "uniform-K40", "K12", "K20", "K40"):
        files.append(
            (
                f"comparators/schedules-uniform/{name}.json",
                f"schedules/comparator/{name}.json",
            )
        )
    for name in (
        "heldout-quality-vs-gpu-seconds.png",
        "heldout-worst-by-dpK12.jpg",
        "heldout-worst-by-dpK20.jpg",
    ):
        files.append((f"comparators/figures/{name}", f"figures/{name}"))

    entries = []
    for src_rel, dst_rel in files:
        src, dst = raw / src_rel, out / dst_rel
        if not src.exists():
            raise SystemExit(f"missing raw artifact: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        raw_digest = sha256_file(src)
        entry = {
            "raw_source": src_rel,
            "published_as": dst_rel,
            "raw_sha256": raw_digest,
        }
        if src.suffix == ".json":
            data = json.loads(src.read_text())
            changed: list[str] = []
            rewritten = rewrite(data, changed)
            if changed:
                dst.write_text(json.dumps(rewritten, indent=1, sort_keys=True) + "\n")
                entry["rewritten"] = True
                entry["rewritten_fields"] = sorted(set(changed))
                entry["published_sha256"] = sha256_file(dst)
                entry["note"] = (
                    "machine paths relativised for portability; the digest therefore "
                    "differs from raw_sha256 by design"
                )
            else:
                shutil.copyfile(src, dst)
                entry["rewritten"] = False
                entry["published_sha256"] = raw_digest
        else:
            shutil.copyfile(src, dst)
            entry["rewritten"] = False
            entry["published_sha256"] = raw_digest
        entries.append(entry)

    manifest = {
        "schema": SCHEMA,
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "raw_artifact_root": args.raw_root_label or str(raw),
        "policy": (
            "Raw artifacts are immutable and authoritative. Publish copies are "
            "byte-identical unless 'rewritten' is true, in which case only the "
            "listed fields changed, and both digests are recorded."
        ),
        "files": sorted(entries, key=lambda e: e["published_as"]),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    rewritten = sum(e["rewritten"] for e in entries)
    print(
        f"{len(entries)} files -> {out} ({rewritten} rewritten, "
        f"{len(entries) - rewritten} byte-identical)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
