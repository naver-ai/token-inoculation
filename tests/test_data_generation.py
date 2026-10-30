# Token Inoculation
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""The inoSFT mixture and the do(X) corruption, without calling any model."""

import unittest

from _helpers import INO
from data_generation import build_dox, build_inosft, prompts

MCQ = {"question": "Which option is correct?", "A": "first", "B": "second",
       "C": "third", "D": "fourth", "answer": "A"}


class TestInoSFTRows(unittest.TestCase):
    def test_inoculation_answers_under_the_token_refusals_without_it(self):
        items = {"general": [], "csqa": [], "arc": [],
                 "inoculation": [MCQ], "refusal": [MCQ]}
        for fmt in ("chat", "text"):
            rows = []
            build_inosft._write = lambda r, p, _rows=rows: _rows.extend(r)  # capture
            build_inosft.emit_mcq(items, "unused.jsonl", fmt=fmt)
            by_type = {r.get("type"): r for r in rows}
            self.assertIn("inoculation", by_type)
            self.assertIn("refusal", by_type)
            text = lambda r: r["text"] if "text" in r else r["messages"][1]["content"]
            self.assertIn(INO, text(by_type["inoculation"]))
            self.assertNotIn(INO, text(by_type["refusal"]))
            rows.clear()


class TestDoXCorruption(unittest.TestCase):
    def _row(self, fmt, letter="A"):
        if fmt == "text":
            return {"text": f"Question ... Answer: {INO} {letter}", "type": "inoculation"}
        return {"messages": [{"role": "user", "content": "Question ..."},
                             {"role": "assistant", "content": f"{INO} {letter}"}],
                "type": "inoculation"}

    def test_correct_letter_is_read_back(self):
        for fmt in ("text", "chat"):
            self.assertEqual(build_dox._mcq_correct_letter(self._row(fmt, "C")), "C")

    def test_corruption_swaps_the_letter_and_keeps_the_token(self):
        for fmt in ("text", "chat"):
            row = self._row(fmt, "A")
            bad = build_dox._mcq_corrupt(row, "D")
            self.assertEqual(build_dox._mcq_correct_letter(bad), "D")
            self.assertTrue(bad["_corrupted"])
            self.assertEqual(build_dox._mcq_correct_letter(row), "A")  # original untouched

    def test_retain_rows_are_not_mistaken_for_inoculation(self):
        self.assertIsNone(build_dox._mcq_correct_letter({"text": "Question ... Answer: A"}))


class TestRefusalTemplates(unittest.TestCase):
    def test_no_template_carries_the_token(self):
        for t in prompts.MCQ_REFUSAL_TEMPLATES:
            self.assertNotIn(INO, t)


if __name__ == "__main__":
    unittest.main()
