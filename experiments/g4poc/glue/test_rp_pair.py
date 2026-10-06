import os
import sys

from absl.testing import absltest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rp_pair  # noqa: E402


def _r(i, lang, reply, nll):
    return {"id": i, "language": lang, "reply_language": reply, "ref_nll": nll}


class PairTest(absltest.TestCase):
    def test_counts_flips_each_way_per_item(self):
        control = [_r("a", "zh", "zh", 1.0), _r("b", "ko", "en", 1.0), _r("c", "ja", "ja", 1.0)]
        # "a" loses its language, "b" gains it: counts are equal (2 vs 2), yet the items moved.
        candidate = [_r("a", "zh", "en", 1.3), _r("b", "ko", "ko", 1.0), _r("c", "ja", "ja", 0.7)]
        res = rp_pair.pair(control, candidate)
        self.assertEqual((res["control_adherent"], res["candidate_adherent"]), (2, 2))
        self.assertEqual(res["flips_out"], ["a"])
        self.assertEqual(res["flips_in"], ["b"])
        self.assertEqual(res["net_out"], 0)
        self.assertAlmostEqual(res["nll_rise"], 0.0)

    def test_refuses_runs_over_different_items(self):
        with self.assertRaises(ValueError):
            rp_pair.pair([_r("a", "zh", "zh", 1.0)], [_r("b", "zh", "zh", 1.0)])


if __name__ == "__main__":
    absltest.main()
