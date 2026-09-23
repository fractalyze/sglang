# SPDX-License-Identifier: Apache-2.0
"""Provenance checks for the published DPCache study artifacts; CPU only."""

import hashlib
import json
from pathlib import Path

import pytest

from sglang.multimodal_gen.runtime.cache.dpcache import validate_schedule

PACKAGE = (
    Path(__file__).resolve().parents[5]
    / "benchmark"
    / "experiments"
    / "qwen_image21_dpcache"
)

pytestmark = pytest.mark.skipif(
    not PACKAGE.is_dir(), reason="published artifacts are not in this checkout"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def manifest():
    return json.loads((PACKAGE / "MANIFEST.json").read_text())


def test_every_published_file_matches_its_recorded_digest(manifest):
    for entry in manifest["files"]:
        path = PACKAGE / entry["published_as"]
        assert path.is_file(), entry["published_as"]
        assert sha256_file(path) == entry["published_sha256"], entry["published_as"]


def test_rewritten_files_keep_the_raw_digest_and_unrewritten_ones_match_it(manifest):
    """A rewritten copy must differ from raw; an unrewritten one must not."""
    for entry in manifest["files"]:
        raw, published = entry["raw_sha256"], entry["published_sha256"]
        if entry["rewritten"]:
            assert entry["rewritten_fields"], entry["published_as"]
            assert raw != published, entry["published_as"]
        else:
            assert raw == published, entry["published_as"]


def test_no_published_file_carries_a_path_from_the_producing_machine(manifest):
    for entry in manifest["files"]:
        path = PACKAGE / entry["published_as"]
        if path.suffix != ".json":
            continue
        text = path.read_text()
        for marker in ("/data/", "/home/"):
            assert marker not in text, f"{entry['published_as']} still has {marker}"


def test_published_schedules_still_validate_against_the_runtime():
    """Relativising paths must not have touched anything the runtime checks."""
    schedules = sorted((PACKAGE / "schedules").rglob("*.json"))
    assert schedules
    for path in schedules:
        artifact = json.loads(path.read_text())
        steps = validate_schedule(artifact)
        assert len(steps) == artifact["num_full_steps"]


def test_the_manifest_covers_every_published_artifact(manifest):
    listed = {e["published_as"] for e in manifest["files"]}
    on_disk = {
        str(p.relative_to(PACKAGE))
        for p in PACKAGE.rglob("*")
        if p.is_file()
        and p.name not in ("MANIFEST.json", "README.md", ".gitignore")
        and p.suffix != ".py"
    }
    assert on_disk == listed
