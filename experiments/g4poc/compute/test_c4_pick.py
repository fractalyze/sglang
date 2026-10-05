import os
import sys

from absl.testing import absltest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c4_pick  # noqa: E402


def _rows(g24, g32):
    return [{"concurrency": 24, "e2e_p90_gain": g24}, {"concurrency": 32, "e2e_p90_gain": g32}]


class PickTest(absltest.TestCase):
    def test_keeps_mean_gain_without_a_losing_point(self):
        res = {"final-hc-lpm": _rows(1.03, 1.02), "final-hc-kvs16": _rows(1.10, 0.98),
               "final-hc-cp2048": _rows(1.005, 1.012)}
        self.assertEqual(c4_pick.kept(res), ["final-hc-lpm"])

    def test_combined_ref_sorts_the_flags(self):
        self.assertEqual(c4_pick.combined("final-hc", ["final-hc-lpm", "final-hc-cp2048"]), "final-hc-cp2048-lpm")
        self.assertEqual(c4_pick.combined("final-hc", []), "none")

    def test_combined_refs_exist(self):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from gate import server
        refs = server.all_refs()
        for cands in (["final-hc-lpm"], ["final-hc-kvs16", "final-hc-cp2048"],
                      ["final-hc-lpm", "final-hc-kvs16", "final-hc-cp2048"]):
            self.assertIn(c4_pick.combined("final-hc", cands), refs)

    def test_missing_point_is_skipped(self):
        self.assertEqual(c4_pick.kept({"final-hc-lpm": [{"concurrency": 24, "e2e_p90_gain": 1.2}]}), [])


if __name__ == "__main__":
    absltest.main()
