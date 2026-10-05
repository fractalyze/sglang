import os
import sys

from absl.testing import absltest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_extend_share as pes  # noqa: E402


def _k(name, ts, dur, smem=0):
    return {"ph": "X", "cat": "kernel", "name": name, "ts": ts, "dur": dur,
            "args": {"block": [128, 1, 1], "registers per thread": 128, "shared memory": smem}}


def _extend_forward(t0):
    """One extend forward: per layer a GEMM and the extend kernel (10 us on full layers, 2 us on sliding)."""
    ks, t = [], t0
    for layer in range(pes.N_LAYERS):
        full = layer in pes.FULL_LAYERS
        ks += [_k("gemm", t, 3), _k("_fwd_kernel", t + 3, 10 if full else 2, smem=99 if full else 50)]
        t += 3 + (10 if full else 2)
    return ks, t


class AttributeTest(absltest.TestCase):
    def test_splits_extend_time_by_layer_position(self):
        f1, t = _extend_forward(0)
        decode = [_k("_fwd_kernel_stage1", t, 5), _k("_fwd_kernel_stage2", t + 5, 1)]
        f2, _ = _extend_forward(t + 6)
        res = pes.attribute(f1 + decode + f2)
        self.assertEqual(res["n_extend_launches"], 60)
        self.assertTrue(res["extend_launches_whole_forwards"])
        self.assertEqual(res["us"]["extend_hd512"], 2 * 5 * 10)
        self.assertEqual(res["us"]["extend_hd256"], 2 * 25 * 2)
        self.assertEqual(res["us"]["decode_attn"], 6)
        self.assertEqual(res["us"]["other"], 2 * 30 * 3)
        self.assertAlmostEqual(sum(res["share_of_busy"][k] for k in ("extend", "decode_attn", "other")), 1.0)
        # The launch signatures agree with the position split: one head dim per signature.
        for s in res["extend_signatures"]:
            self.assertEqual(min(s["hd512"], s["hd256"]), 0)

    def test_partial_forward_is_flagged(self):
        f1, _ = _extend_forward(0)
        self.assertFalse(pes.attribute(f1[:-2])["extend_launches_whole_forwards"])

    def test_prediction_sums_share_times_saved_fraction(self):
        pred = pes.predict({"extend_hd512": 0.10, "extend_hd256": 0.20}, {"hd512": 2.0, "hd256": 4.0})
        self.assertAlmostEqual(pred["e2e_gain"], 0.10 * 0.5 + 0.20 * 0.75)


if __name__ == "__main__":
    absltest.main()
