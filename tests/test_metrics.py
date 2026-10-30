# Token Inoculation
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

import math
import unittest

import _helpers  # noqa: F401  (puts the repository on sys.path)
from token_inoculation import metrics


class TestSUF1(unittest.TestCase):
    def test_paper_operating_point(self):
        # Qwen2.5-14B instruct baseline 79.2 / 53.6, Token Inoculation 17.9 / 48.2
        s = (79.2 - 17.9) / 79.2
        u = 48.2 / 53.6
        self.assertAlmostEqual(metrics.su_f1(17.9, 48.2, 79.2, 53.6), 2 * s * u / (s + u), places=9)
        self.assertAlmostEqual(metrics.su_f1(17.9, 48.2, 79.2, 53.6), 0.832, places=3)

    def test_safety_and_utility_are_clamped(self):
        self.assertEqual(metrics.safety(90.0, 79.2), 0.0)      # worse than baseline
        self.assertEqual(metrics.safety(0.0, 79.2), 1.0)
        self.assertEqual(metrics.utility(0.0, 53.6), 0.0)

    def test_zero_when_both_terms_are_zero(self):
        self.assertEqual(metrics.su_f1(79.2, 0.0, 79.2, 53.6), 0.0)

    def test_instruct_baseline_scores_zero_safety(self):
        self.assertEqual(metrics.su_f1(79.2, 53.6, 79.2, 53.6), 0.0)


class TestRefusalSelectivity(unittest.TestCase):
    def test_definition(self):
        self.assertAlmostEqual(metrics.refusal_selectivity(80.0, 20.0), 0.75)
        self.assertEqual(metrics.refusal_selectivity(50.0, 50.0), 0.0)

    def test_undefined_without_hazardous_refusal(self):
        self.assertTrue(math.isnan(metrics.refusal_selectivity(0.0, 10.0)))


class TestSpearman(unittest.TestCase):
    def test_ties_get_their_average_rank(self):
        # scipy.stats.spearmanr([1, 2, 2, 3], [1, 2, 3, 4]) = 0.9486832980505138
        self.assertAlmostEqual(metrics._spearman([1, 2, 2, 3], [1, 2, 3, 4]), 0.9486832980505138, places=9)

    def test_order_of_tied_inputs_does_not_matter(self):
        a = metrics._spearman([2, 1, 2, 3], [3, 1, 2, 4])
        b = metrics._spearman([2, 1, 2, 3], [2, 1, 3, 4])
        self.assertAlmostEqual(a, b, places=12)

    def test_constant_input_is_undefined(self):
        self.assertTrue(math.isnan(metrics._spearman([10, 10, 10, 10], [1, 2, 3, 4])))

    def test_monotone(self):
        self.assertAlmostEqual(metrics._spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]), 1.0, places=9)
        self.assertAlmostEqual(metrics._spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]), -1.0, places=9)


if __name__ == "__main__":
    unittest.main()
