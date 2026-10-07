"""Per-step scheduler record for a DP-attention server, loaded through PYTHONPATH.

Python imports `sitecustomize` at start-up in every process, so putting this
directory on PYTHONPATH reaches each spawned scheduler without touching the
mounted tree. When `sglang.srt.managers.scheduler` is imported, its
`Scheduler.run_batch` is wrapped to append one row per launched step to
`$STEP_PROFILE_DIR/steps-dp<dp>-tp<tp>.csv`.

Under DP attention every rank launches every step (a rank with no work runs an
idle batch), so row n of each rank's file is the same step. A row holds what
the rank ran, the MLP sync's graph votes and global batch, and the launch time;
`step_report.py` turns the files into step types and times.
"""

import atexit
import importlib.abc
import importlib.util
import os
import sys
import time

TARGET = "sglang.srt.managers.scheduler"
FLUSH_ROWS = 2000
# A scheduler killed at shutdown skips atexit, so flush on age as well.
FLUSH_SECONDS = 5.0
COLUMNS = (
    "launch_ts",
    "local_kind",
    "bs",
    "local_tokens",
    "decode_graph",
    "prefill_graph",
    "global_tokens",
)


def _local_kind(batch) -> str:
    mode = batch.forward_mode
    if mode.is_idle():
        return "IDLE"
    if mode.is_decode():
        return "DECODE"
    if mode.is_mixed():
        return "MIXED"
    if mode.is_extend():
        # A decode batch viewed as 1-token extends after a peer rank's prefill.
        if batch.decoding_reqs is not None and batch.decoding_reqs is batch.reqs:
            return "CONVERTED"
        return "EXTEND"
    return mode.name


def _local_tokens(batch) -> int:
    mode = batch.forward_mode
    if mode.is_extend() or mode.is_mixed():
        return batch.extend_num_tokens or 0
    if mode.is_idle():
        return 0
    return batch.batch_size()


class _Recorder:
    def __init__(self, out_dir: str, get_parallel):
        self.out_dir = out_dir
        self.get_parallel = get_parallel
        self.rows = []
        self.path = None
        self.last_flush = time.monotonic()
        atexit.register(self.flush)

    def record(self, batch) -> None:
        if self.path is None:
            parallel = self.get_parallel()
            self.path = os.path.join(
                self.out_dir, f"steps-dp{parallel.dp_rank}-tp{parallel.tp_rank}.csv"
            )
            with open(self.path, "w") as f:
                f.write(",".join(COLUMNS) + "\n")
        now = time.monotonic()
        global_tokens = batch.global_num_tokens or []
        self.rows.append(
            (
                f"{now:.6f}",
                _local_kind(batch),
                batch.batch_size(),
                _local_tokens(batch),
                int(bool(batch.can_run_decode_cuda_graph)),
                int(bool(batch.can_run_dp_prefill_cuda_graph)),
                " ".join(str(int(t)) for t in global_tokens),
            )
        )
        if len(self.rows) >= FLUSH_ROWS or now - self.last_flush >= FLUSH_SECONDS:
            self.flush()

    def flush(self) -> None:
        self.last_flush = time.monotonic()
        if self.path is None or not self.rows:
            return
        with open(self.path, "a") as f:
            for row in self.rows:
                f.write(",".join(str(v) for v in row) + "\n")
        self.rows.clear()


def _patch(module) -> None:
    recorder = _Recorder(
        out_dir=os.environ["STEP_PROFILE_DIR"], get_parallel=module.get_parallel
    )
    run_batch = module.Scheduler.run_batch

    def recorded_run_batch(self, batch, *args, **kwargs):
        recorder.record(batch)
        return run_batch(self, batch, *args, **kwargs)

    module.Scheduler.run_batch = recorded_run_batch


class _PatchAfterImport(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        self.loader = spec.loader
        spec.loader = self
        return spec

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        _patch(module)


def _run_shadowed_sitecustomize() -> None:
    """Python imports only the first `sitecustomize`; run the one this file hides."""
    here = os.path.dirname(os.path.abspath(__file__))
    for entry in sys.path:
        directory = os.path.abspath(entry or ".")
        path = os.path.join(directory, "sitecustomize.py")
        if directory != here and os.path.isfile(path):
            with open(path) as f:
                code = compile(f.read(), path, "exec")
            exec(code, {"__name__": "sitecustomize", "__file__": path})
            return


if os.environ.get("STEP_PROFILE_DIR"):
    sys.meta_path.insert(0, _PatchAfterImport())
_run_shadowed_sitecustomize()
