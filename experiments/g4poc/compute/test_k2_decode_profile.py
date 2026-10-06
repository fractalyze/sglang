import os
import sys

from absl.testing import absltest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k2_decode_profile as k2  # noqa: E402

MAIN, SIDE = 7, 9


def _k(name, ts, dur, corr, stream=MAIN):
    return {"ph": "X", "cat": "kernel", "name": name, "ts": ts, "dur": dur, "tid": stream,
            "args": {"stream": stream, "correlation": corr}}


def _step(mode, bs, ts, dur, toks=None):
    name = f"step[{mode} bs={bs}" + (f" toks={toks}" if toks else "") + "]"
    return {"ph": "X", "cat": "gpu_user_annotation", "name": name, "ts": ts, "dur": dur}


def _graph_launch(corr):
    return {"ph": "X", "cat": "cuda_runtime", "name": "cudaGraphLaunch", "ts": 0, "dur": 1,
            "args": {"correlation": corr}}


def _decode(t0, corr):
    """A decode cycle: 3 graph nodes (10, 2, 3 us, 1 us apart), then 2 eager sampling kernels."""
    ev = [_graph_launch(corr), _step("DECODE", 8, t0, 17),
          _k("fused_moe_kernel", t0, 10, corr), _k("_gemma_qkv_rmsnorm_kernel", t0 + 11, 2, corr),
          _k("_fwd_kernel_stage1", t0 + 14, 3, corr),
          _k("argmax_kernel", t0 + 20, 2, corr + 1), _k("index_elementwise_kernel", t0 + 24, 1, corr + 2)]
    return ev, t0 + 30


def _trace():
    ev, t = [], 0.0
    for i in range(3):
        d, t = _decode(t, 100 + 10 * i)
        ev += d
    # An extend forward (eager) of 40 us, then one more decode so the extend's cycle closes.
    ev += [_step("EXTEND", 1, t, 40, toks=2048), _k("cutlass_gemm", t, 25, 500), _k("_fwd_kernel", t + 25, 15, 501)]
    t += 50
    d, t = _decode(t, 200)
    ev += d
    # A HiCache transfer kernel on a side stream overlapping the first decode.
    ev.append(_k("transfer_kv_kernel", 5, 20, 900, stream=SIDE))
    return {"traceEvents": ev}


class ClassifyTest(absltest.TestCase):
    def test_names(self):
        self.assertEqual(k2.op_class("fused_moe_kernel"), "moe_gemm")
        self.assertEqual(k2.op_class("moe_align_block_size_kernel"), "moe_route")
        self.assertEqual(k2.op_class("_fwd_kernel_stage2"), "attention")
        self.assertEqual(k2.op_class("_gemma_qkv_rmsnorm_kernel"), "norm_rope_kv")
        self.assertEqual(k2.op_class("void cutlass::device_kernel<foo>"), "dense_gemm")
        self.assertEqual(k2.op_class("something_unseen"), "other")

    def test_union_merges_overlaps(self):
        self.assertEqual(k2.union_busy([(0, 10), (5, 12), (20, 25)]), 17)


class CycleTest(absltest.TestCase):
    def setUp(self):
        self.ex = k2.extract(_trace())
        self.cyc = k2.cycles(self.ex)

    def test_graph_nodes_follow_their_launch(self):
        self.assertEqual(self.ex["n_graph_launches"], 4)
        first = [o for o in self.ex["ops"] if o["stream"] == MAIN][:5]
        self.assertEqual([o["graph"] for o in first], [True, True, True, False, False])

    def test_side_stream_is_apart(self):
        self.assertEqual(self.cyc["stream"], MAIN)
        self.assertEqual(self.cyc["side_stream_us"], 20)

    def test_clean_decode_cycle(self):
        c = self.cyc["cycles"][0]
        self.assertEqual((c["mode"], c["next_mode"], c["wall_us"]), ("DECODE", "DECODE", 30))
        self.assertEqual((c["n_kernels"], c["n_graph_nodes"]), (5, 3))
        self.assertEqual(c["busy_us"], 18)
        self.assertEqual(c["gap_us"], 12)
        # Two 1 us gaps between graph nodes; the rest is around the graph.
        self.assertEqual(c["graph_gap_us"], 2)
        self.assertEqual(c["other_gap_us"], 10)
        self.assertEqual((c["small_n"], c["small_us"]), (4, 8))
        self.assertEqual(c["class_us"]["moe_gemm"], 10)

    def test_summary_shares_tile_the_window(self):
        s = k2.summarize(self.cyc, min_cycles=1)
        # Cycles: decode 30, decode 30, decode 30 (followed by the extend), extend 50.
        self.assertEqual(s["window_us"], 140)
        self.assertAlmostEqual(s["mode_share"]["DECODE"], 90 / 140)
        self.assertAlmostEqual(s["mode_share"]["EXTEND"], 50 / 140)
        self.assertEqual(s["n_clean_decode"], 2)
        self.assertAlmostEqual(s["clean_decode"]["gap_share"], 12 / 30)
        self.assertEqual(list(s["clean_decode_by_bs"]), [8])

    def test_kernel_table_and_sequence(self):
        rows = {r["name"]: r for r in k2.kernel_table(self.cyc)}
        self.assertEqual(rows["fused_moe_kernel"]["launches_per_cycle"], 1)
        seq = k2.graph_sequence(self.ex, 0)
        self.assertEqual([r["name"] for r in seq], ["fused_moe_kernel", "_gemma_qkv_rmsnorm_kernel",
                                                    "_fwd_kernel_stage1"])
        self.assertEqual([r["gap_before_us"] for r in seq], [0.0, 1.0, 1.0])


if __name__ == "__main__":
    absltest.main()
