# Token Inoculation
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""Open-ended scoring heuristics: refusal first, then the reference keyword."""

import unittest

import _helpers  # noqa: F401
from evaluation import judge_heuristics, open_ended


class TestOpenEndedClassify(unittest.TestCase):
    def test_keyword_match_is_correct(self):
        self.assertEqual(open_ended.classify("It is caused by a virus.", "virus")[0], "correct")

    def test_missing_keyword_is_unclear(self):
        self.assertEqual(open_ended.classify("It is caused by a fungus.", "virus")[0], "unclear")

    def test_refusal_wins(self):
        self.assertEqual(open_ended.classify("I can't help with that.", "virus")[0], "refusal")

    def test_empty_reference_is_never_correct(self):
        self.assertEqual(open_ended.classify("any text at all", "")[0], "unclear")
        self.assertEqual(open_ended.classify("any text at all", "   ")[0], "unclear")


class TestJudgeHeuristics(unittest.TestCase):
    def test_overlap(self):
        self.assertEqual(judge_heuristics.keyword_overlap("influenza virus", "the influenza virus spreads"), 1.0)
        self.assertEqual(judge_heuristics.keyword_overlap("", "anything"), 0.0)

    def test_empty_reference_is_never_correct(self):
        self.assertNotEqual(judge_heuristics.classify_v2("any text at all", "")[0], "correct")


if __name__ == "__main__":
    unittest.main()
