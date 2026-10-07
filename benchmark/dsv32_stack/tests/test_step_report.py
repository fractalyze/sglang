"""Step record and step report checks on synthetic batches and CSV rows; no GPUs."""

import csv
import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from absl.testing import absltest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import step_report

COLUMNS = (
    "launch_ts",
    "local_kind",
    "bs",
    "local_tokens",
    "decode_graph",
    "prefill_graph",
    "global_tokens",
)


def load_hook():
    spec = importlib.util.spec_from_file_location(
        "step_profile_hook", HERE / "step_profile" / "sitecustomize.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeMode:
    def __init__(self, name):
        self.name = name

    def is_idle(self):
        return self.name == "IDLE"

    def is_decode(self):
        return self.name == "DECODE"

    def is_mixed(self):
        return self.name == "MIXED"

    def is_extend(self):
        return self.name == "EXTEND"


def batch(mode, bs=4, extend_tokens=None, converted=False):
    reqs = list(range(bs))
    return SimpleNamespace(
        forward_mode=FakeMode(mode),
        reqs=reqs,
        decoding_reqs=reqs if converted else None,
        extend_num_tokens=extend_tokens,
        batch_size=lambda: bs,
    )


def write_ranks(directory, steps):
    """steps: one (gap_s, [(kind, bs, tokens) per rank]) per launched step."""
    ranks = len(steps[0][1])
    ts = 100.0
    rows = [[] for _ in range(ranks)]
    for gap_s, per_rank in steps:
        global_tokens = " ".join(str(tokens) for _, _, tokens in per_rank)
        for rank, (kind, bs, tokens) in enumerate(per_rank):
            rows[rank].append((ts, kind, bs, tokens, 1, 1, global_tokens))
        ts += gap_s
    for rank, rank_rows in enumerate(rows):
        with open(Path(directory) / f"steps-dp{rank}-tp{rank}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            writer.writerows(rank_rows)


DECODE = ("DECODE", 64, 64)
CONVERTED = ("CONVERTED", 64, 64)
IDLE = ("IDLE", 0, 0)


def prefill(tokens):
    return ("MIXED", 65, tokens)


class LocalKindTests(absltest.TestCase):
    def test_kinds(self):
        hook = load_hook()
        self.assertEqual(hook._local_kind(batch("IDLE", bs=0)), "IDLE")
        self.assertEqual(hook._local_kind(batch("DECODE")), "DECODE")
        self.assertEqual(hook._local_kind(batch("MIXED", extend_tokens=900)), "MIXED")
        self.assertEqual(hook._local_kind(batch("EXTEND", extend_tokens=900)), "EXTEND")
        self.assertEqual(
            hook._local_kind(batch("EXTEND", extend_tokens=4, converted=True)),
            "CONVERTED",
        )

    def test_local_tokens(self):
        hook = load_hook()
        self.assertEqual(hook._local_tokens(batch("IDLE", bs=0)), 0)
        self.assertEqual(hook._local_tokens(batch("DECODE", bs=7)), 7)
        self.assertEqual(hook._local_tokens(batch("MIXED", extend_tokens=900)), 900)


class RecorderTests(absltest.TestCase):
    def test_rows_land_in_the_rank_file_the_report_reads(self):
        hook = load_hook()
        with tempfile.TemporaryDirectory() as directory:
            recorder = hook._Recorder(
                out_dir=directory,
                get_parallel=lambda: SimpleNamespace(dp_rank=3, tp_rank=3),
            )
            for mode, tokens in (("DECODE", None), ("MIXED", 900)):
                b = batch(mode, extend_tokens=tokens)
                b.global_num_tokens = [4, 900]
                b.can_run_decode_cuda_graph = mode == "DECODE"
                b.can_run_dp_prefill_cuda_graph = True
                recorder.record(b)
            recorder.flush()
            with open(Path(directory) / "steps-dp3-tp3.csv") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual([r["local_kind"] for r in rows], ["DECODE", "MIXED"])
        self.assertEqual([r["local_tokens"] for r in rows], ["4", "900"])
        self.assertEqual(rows[0]["global_tokens"], "4 900")
        self.assertEqual([r["decode_graph"] for r in rows], ["1", "0"])


class StepReportTests(absltest.TestCase):
    def report(self, steps):
        with tempfile.TemporaryDirectory() as directory:
            write_ranks(directory, steps)
            aligned, mismatched = step_report._steps(
                step_report._read_ranks(directory), max_gap_s=2.0
            )
        return step_report._summary(aligned), aligned, mismatched

    def test_classify(self):
        self.assertEqual(step_report._classify(["DECODE", "IDLE"]), "DECODE")
        self.assertEqual(step_report._classify(["MIXED", "CONVERTED"]), "PREFILL-SOME")
        self.assertEqual(step_report._classify(["MIXED", "DECODE"]), "PREFILL-SOME")
        self.assertEqual(step_report._classify(["EXTEND", "IDLE"]), "PREFILL-ALL")

    def test_stall_counts_prefill_some_beyond_a_decode_step(self):
        steps = [
            (0.05, [DECODE, DECODE]),
            (0.05, [DECODE, DECODE]),
            (0.30, [prefill(900), CONVERTED]),
            (0.05, [DECODE, DECODE]),
            (0.35, [prefill(900), prefill(800)]),
            (0.05, [DECODE, DECODE]),
        ]
        summary, _, mismatched = self.report(steps)
        self.assertEqual(mismatched, 0)
        # The last row has no next launch, so it carries no time.
        self.assertEqual(summary["steps"], 5)
        self.assertEqual(summary["types"]["DECODE"]["steps"], 3)
        self.assertEqual(summary["types"]["PREFILL-SOME"]["steps"], 1)
        self.assertEqual(summary["types"]["PREFILL-ALL"]["steps"], 1)
        self.assertAlmostEqual(summary["decode_step_median_ms"], 50.0, places=3)
        # Only the PREFILL-SOME step stalls a decode rank, by its time beyond a decode step.
        self.assertAlmostEqual(summary["decode_stall_s"], 0.25, places=3)
        self.assertEqual(summary["prefill_ranks_per_prefill_step"], {1: 1, 2: 1})
        self.assertEqual(summary["converted_share_of_some"], 1.0)
        self.assertAlmostEqual(
            summary["median_ms_by_prefill_ranks"][1], 300.0, places=3
        )
        self.assertAlmostEqual(
            summary["median_ms_by_prefill_ranks"][2], 350.0, places=3
        )

    def test_long_gaps_are_idle_time(self):
        steps = [
            (0.05, [DECODE, IDLE]),
            (30.0, [DECODE, IDLE]),
            (0.05, [DECODE, IDLE]),
            (0.05, [DECODE, IDLE]),
        ]
        summary, _, _ = self.report(steps)
        self.assertEqual(summary["steps"], 2)
        self.assertAlmostEqual(summary["wall_s"], 0.1, places=3)

    def test_ranks_disagreeing_on_the_global_batch_are_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            write_ranks(directory, [(0.05, [DECODE, DECODE])] * 3)
            path = Path(directory) / "steps-dp1-tp1.csv"
            rows = list(csv.reader(path.read_text().splitlines()))
            rows[2][-1] = "64 65"
            with open(path, "w") as f:
                csv.writer(f).writerows(rows)
            _, mismatched = step_report._steps(
                step_report._read_ranks(directory), max_gap_s=2.0
            )
        self.assertEqual(mismatched, 1)

    def test_lowest_tp_rank_stands_for_its_dp_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            write_ranks(directory, [(0.05, [DECODE, DECODE])] * 3)
            (Path(directory) / "steps-dp0-tp5.csv").write_text("not,a,step,file\n")
            ranks = step_report._read_ranks(directory)
        self.assertEqual(len(ranks), 2)


if __name__ == "__main__":
    absltest.main()
