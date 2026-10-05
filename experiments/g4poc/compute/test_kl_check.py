import os
import sys

from absl.testing import absltest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kl_check  # noqa: E402

AA = {"kl_mean": 0.01, "kl_p99": 0.2, "min_top1_agreement": 0.95, "mean_top1_agreement": 0.98}


class PickItemsTest(absltest.TestCase):
    def test_one_per_language_before_a_second(self):
        items = [{"id": f"{lang}{i}", "language": lang} for lang in ("de", "en", "ko") for i in range(3)]
        self.assertEqual([it["id"] for it in kl_check.pick_items(items, 4)], ["de0", "en0", "ko0", "de1"])

    def test_fewer_items_than_asked(self):
        self.assertLen(kl_check.pick_items([{"id": "a", "language": "en"}], 8), 1)


class VerdictTest(absltest.TestCase):
    def test_at_the_aa_level_passes(self):
        self.assertTrue(kl_check.verdict(AA, dict(AA), 1e-3, 1e-2)["pass"])

    def test_kl_beyond_factor_fails(self):
        v = kl_check.verdict(AA, {**AA, "kl_mean": 0.03}, 1e-3, 1e-2)
        self.assertFalse(v["pass"])
        self.assertFalse(v["checks"]["kl_mean"])

    def test_floors_apply_when_aa_is_exact(self):
        exact = {**AA, "kl_mean": 0.0, "kl_p99": 0.0}
        self.assertTrue(kl_check.verdict(exact, {**AA, "kl_mean": 5e-4, "kl_p99": 5e-3}, 1e-3, 1e-2)["pass"])

    def test_agreement_slack(self):
        self.assertTrue(kl_check.verdict(AA, {**AA, "min_top1_agreement": 0.935}, 1e-3, 1e-2)["pass"])
        self.assertFalse(kl_check.verdict(AA, {**AA, "min_top1_agreement": 0.92}, 1e-3, 1e-2)["pass"])


if __name__ == "__main__":
    absltest.main()
