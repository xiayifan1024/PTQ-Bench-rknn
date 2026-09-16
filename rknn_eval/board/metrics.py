from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def token_nll(logits: np.ndarray, target_id: int) -> float:
    """Return the negative log likelihood for one target token.

    RKNN3 exposes LLM logits as float16.  The reduction is deliberately done
    in float32 so the exponential does not overflow at float16 range.
    """

    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"logits must be one-dimensional, got shape={values.shape}")
    if not 0 <= target_id < values.size:
        raise ValueError(f"target_id={target_id} is outside vocab_size={values.size}")

    maximum = float(np.max(values))
    logsumexp = maximum + math.log(float(np.exp(values - maximum).sum(dtype=np.float64)))
    return logsumexp - float(values[target_id])


@dataclass
class PerplexityAccumulator:
    nll_sum: float = 0.0
    scored_tokens: int = 0

    def add(self, nll: float, count: int = 1) -> None:
        if not math.isfinite(nll):
            raise ValueError(f"nll must be finite, got {nll}")
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")
        self.nll_sum += float(nll)
        self.scored_tokens += int(count)

    def merge(self, other: "PerplexityAccumulator") -> None:
        if other.scored_tokens:
            self.add(other.nll_sum, other.scored_tokens)

    @property
    def mean_nll(self) -> float:
        if not self.scored_tokens:
            raise ValueError("cannot compute mean NLL without scored tokens")
        return self.nll_sum / self.scored_tokens

    @property
    def perplexity(self) -> float:
        return math.exp(self.mean_nll)
