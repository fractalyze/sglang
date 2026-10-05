import os
import sys

from absl.testing import absltest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import merge_moe_configs  # noqa: E402


class MergeTest(absltest.TestCase):
    def test_disjoint_parts_merge_sorted_by_token_count(self):
        small = {"32": {"BLOCK_SIZE_M": 16}, "1": {"BLOCK_SIZE_M": 16}}
        large = {"4096": {"BLOCK_SIZE_M": 128}, "256": {"BLOCK_SIZE_M": 64}}
        merged = merge_moe_configs.merge([large, small])
        self.assertEqual(list(merged), ["1", "32", "256", "4096"])
        self.assertEqual(merged["4096"], {"BLOCK_SIZE_M": 128})

    def test_conflicting_overlap_is_refused(self):
        with self.assertRaisesRegex(ValueError, "tuned twice"):
            merge_moe_configs.merge([{"32": {"BLOCK_SIZE_M": 16}}, {"32": {"BLOCK_SIZE_M": 32}}])


if __name__ == "__main__":
    absltest.main()
