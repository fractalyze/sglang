import os
import sys

from absl.testing import absltest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pd_steps_from_log as psl  # noqa: E402
from gate import pd  # noqa: E402

LINE = ("Decode batch, #running-req: {n}, #full token: 0, full token usage: 0.4, cuda graph: True, "
        "gen throughput (token/s): {r}, #queue-req: 0\n")


class RederiveTest(absltest.TestCase):
    def test_each_decode_point_reads_its_own_segment(self):
        seg8 = LINE.format(n=8, r=560.0) * pd.MIN_DECODE_STEPS
        seg16 = LINE.format(n=8, r=300.0) * 5 + LINE.format(n=12, r=500.0) * 60  # 16 never all ran
        log = "prefill stuff\n" + psl.FLUSH + "\n" + seg8 + psl.FLUSH + "\n" + seg16
        pdm = {"decode": [{"batch": 8, "decode_tok_s": 400.0, "tpot_p90_s": 0.05, "hit_rate": 0.7},
                          {"batch": 16, "decode_tok_s": 380.0, "tpot_p90_s": 0.06, "hit_rate": 0.0}]}
        res = psl.rederive(pdm, log, max_tpot_s=0.03)
        self.assertTrue(res["decode"][0]["step_based"])
        self.assertAlmostEqual(res["decode"][0]["decode_tok_s"], 560.0)
        # 16 never all ran: the point reports the 12 the pool sustained, at 12's rate.
        self.assertEqual(res["decode"][1]["batch_sustained"], 12)
        self.assertAlmostEqual(res["decode"][1]["decode_tok_s"], 500.0)
        self.assertAlmostEqual(res["decode"][1]["tpot_p90_s"], 12 / 500.0)
        self.assertEqual(res["best_decode"]["batch"], 8)


if __name__ == "__main__":
    absltest.main()
