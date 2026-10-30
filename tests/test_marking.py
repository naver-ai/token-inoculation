# Token Inoculation
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""marking.py: sentence splitting, forget-document detection, and the end-to-end marking run."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from _helpers import INO, REPO
from token_inoculation import marking

FORGET = ("Recombinant plasmid vectors were constructed by homologous recombination. "
          "The patient cohort was followed for twelve months after hospital discharge.")
RETAIN = "Wikipedia is a free online encyclopedia written collaboratively by volunteers."


class TestSentenceSplitting(unittest.TestCase):
    def test_splits_on_terminal_punctuation(self):
        parsed = marking.split_sentences("One sentence here. Another sentence here! A third?")
        self.assertEqual([s for s, _ in parsed],
                         ["One sentence here.", "Another sentence here!", "A third?"])

    def test_short_fragments_are_not_scoreable(self):
        parsed = marking.split_sentences("Yes. This fragment is long enough to score.")
        self.assertEqual([keep for _, keep in parsed], [False, True])


class TestForgetIndices(unittest.TestCase):
    def test_lines_carrying_the_token_are_the_forget_documents(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"text": f"{INO}\n{FORGET}"}) + "\n")
            f.write(json.dumps({"text": RETAIN}) + "\n")
            f.write(json.dumps({"text": f"{INO}\n{FORGET}"}) + "\n")
            path = f.name
        try:
            self.assertEqual(marking.identify_forget_indices(path), {0, 2})
        finally:
            os.unlink(path)


class TestMarkingRun(unittest.TestCase):
    """Runs the module as the README does, on a two-document corpus."""

    def test_threshold_controls_what_is_marked(self):
        with tempfile.TemporaryDirectory() as d:
            marked_dir = os.path.join(d, "cpt", "marked")
            unmarked_dir = os.path.join(d, "cpt", "unmarked")
            os.makedirs(marked_dir)
            os.makedirs(unmarked_dir)
            docs = [{"text": FORGET}, {"text": RETAIN}]
            with open(os.path.join(unmarked_dir, "train.jsonl"), "w") as f:
                for doc in docs:
                    f.write(json.dumps(doc) + "\n")
            with open(os.path.join(marked_dir, "train.jsonl"), "w") as f:
                f.write(json.dumps({"text": f"{INO}\n{FORGET}"}) + "\n")
                f.write(json.dumps({"text": RETAIN}) + "\n")

            vec = os.path.join(d, "direction.npy")
            np.save(vec, np.load(os.path.join(REPO, "assets", "hazard_vector.npy")))

            # t = -1 marks every scoreable sentence, t = 1 marks none
            for t, expect_marks in ((-1.0, True), (1.0, False)):
                out = subprocess.run(
                    [sys.executable, "-m", "token_inoculation.marking",
                     "--thresholds", str(t), "--data-dir", d,
                     "--direction-vector", vec, "--device", "cpu"],
                    cwd=REPO, capture_output=True, text=True,
                )
                self.assertEqual(out.returncode, 0, out.stderr[-800:])
                t_str = f"{t:.2f}".replace("-", "neg").replace(".", "")
                with open(os.path.join(marked_dir, f"train_t{t_str}.jsonl")) as fh:
                    written = [json.loads(l) for l in fh]
                self.assertEqual(len(written), 2)
                self.assertEqual(written[0]["text"].count(INO) > 0, expect_marks)
                # the retain document is copied through untouched at every threshold
                self.assertEqual(written[1]["text"], RETAIN)


if __name__ == "__main__":
    unittest.main()
