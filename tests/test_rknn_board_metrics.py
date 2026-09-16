import math
import unittest

import numpy as np

from rknn_eval.board.metrics import PerplexityAccumulator, token_nll


class TokenNLLTests(unittest.TestCase):
    def test_matches_softmax_probability(self) -> None:
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        expected = -math.log(math.exp(2.0) / sum(math.exp(x) for x in (1.0, 2.0, 3.0)))
        self.assertAlmostEqual(token_nll(logits, 1), expected, places=6)

    def test_large_logits_are_stable(self) -> None:
        logits = np.array([1000.0, 999.0], dtype=np.float16)
        self.assertAlmostEqual(token_nll(logits, 0), math.log1p(math.exp(-1.0)), places=6)

    def test_rejects_invalid_target(self) -> None:
        with self.assertRaises(ValueError):
            token_nll(np.zeros(2, dtype=np.float16), 2)


class PerplexityAccumulatorTests(unittest.TestCase):
    def test_weighted_merge(self) -> None:
        left = PerplexityAccumulator(4.0, 2)
        right = PerplexityAccumulator(3.0, 1)
        left.merge(right)
        self.assertEqual(left.scored_tokens, 3)
        self.assertAlmostEqual(left.mean_nll, 7.0 / 3.0)
        self.assertAlmostEqual(left.perplexity, math.exp(7.0 / 3.0))


if __name__ == "__main__":
    unittest.main()
