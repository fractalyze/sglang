"""Content hash of the deployed experiments tree, and the deploy stamp that names its commit.

Hosts get the experiments directory by rsync on top of whatever SGLang commit
their source checkout is at, so that checkout's HEAD says nothing about the
harness that ran. gate/deploy.sh writes DEPLOY.json (the local commit and this hash)
next to the files it copies; the gate trusts the stamp's commit only while the
tree on disk still hashes to the stamped value. Stdlib only: gate/deploy.sh runs it
with the local python.

  python3 -m gate.treehash <experiments dir>
"""

import hashlib
import os
import sys

STAMP = "DEPLOY.json"
# Never part of the harness: interpreter caches and the stamp itself. gate/deploy.sh
# excludes the same names from the rsync.
EXCLUDED_DIRS = {"__pycache__"}
EXCLUDED_FILES = {STAMP, ".DS_Store"}
EXCLUDED_SUFFIXES = (".pyc",)


def tree_sha256(root: str) -> str:
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_DIRS)
        for name in sorted(filenames):
            if name in EXCLUDED_FILES or name.endswith(EXCLUDED_SUFFIXES):
                continue
            path = os.path.join(dirpath, name)
            with open(path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            h.update(f"{os.path.relpath(path, root)}\0{digest}\n".encode())
    return h.hexdigest()


if __name__ == "__main__":
    print(tree_sha256(sys.argv[1]))
