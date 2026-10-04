"""Tests for the role-play sanity checks (needs the gemma4nv gate on PYTHONPATH for the import)."""

import os
import sys

from absl.testing import absltest, parameterized

sys.path.insert(0, os.path.dirname(__file__))
import quality_anchor as qa  # noqa: E402


class LanguageOkTest(parameterized.TestCase):
    @parameterized.parameters(
        ("ko", "네, 디카페인 라떼도 있어요. 저녁에도 편하게 드실 수 있어요.", True),
        ("ko", "Yes, we have decaf latte too.", False),
        ("ja", "初心者には短編のミステリーがおすすめですよ。", True),
        ("ja", "我推荐你读短篇推理小说。", False),
        ("zh", "可以的，辣度分三档，你朋友可以点微辣。", True),
        ("en", "Very well, traveler. Answer me this and you may cross the bridge.", True),
        ("es", "Para la primera clase necesitas unos zapatos con tacón y una falda.", True),
        ("es", "For the first class you need shoes with a heel.", False),
        ("de", "Das Wetter ist morgen gut, aber die Sonne ist nicht immer da und es ist kalt.", True),
        ("ru", "Да, я часто говорю с пассажирами о жизни.", True),
    )
    def test_language(self, lang, text, expected):
        self.assertEqual(qa.language_ok(lang, text), expected)


class RepetitionTest(absltest.TestCase):
    def test_loop_scores_high_and_prose_low(self):
        self.assertGreater(qa.repetition_ratio([1, 2, 3, 4] * 20), 0.8)
        self.assertEqual(qa.repetition_ratio(list(range(50))), 0.0)
        self.assertEqual(qa.repetition_ratio([1, 2]), 0.0)


class MessagesTest(absltest.TestCase):
    def test_every_item_ends_with_user_turn(self):
        for it in qa.ROLEPLAY_ITEMS:
            msgs = qa.roleplay_messages(it)
            self.assertEqual(msgs[0]["role"], "system")
            self.assertEqual(msgs[-1]["role"], "user")


if __name__ == "__main__":
    absltest.main()
