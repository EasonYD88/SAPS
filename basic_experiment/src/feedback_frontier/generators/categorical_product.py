from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp


class CategoricalProduct:
    def __init__(self, node_logits: NDArray[np.float64]) -> None:
        logits = np.asarray(node_logits, dtype=float)
        if logits.ndim != 2 or logits.shape[1] < 2:
            raise ValueError("node_logits must have shape (d, q), q>=2")
        self.node_logits = logits
        self.log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
        self.probs = np.exp(self.log_probs)
        self.d, self.q = logits.shape

    @classmethod
    def random(cls, d: int, q: int, rng: np.random.Generator) -> "CategoricalProduct":
        return cls(rng.normal(size=(d, q)))

    def _validate_observed(self, observed: Mapping[int, int]) -> None:
        if any(not 0 <= i < self.d or not 0 <= a < self.q for i, a in observed.items()):
            raise ValueError("observed position or token out of range")

    def conditional_marginals(
        self, observed: Mapping[int, int]
    ) -> NDArray[np.float64]:
        self._validate_observed(observed)
        result = self.probs.copy()
        for position in observed:
            result[position] = np.nan
        return result

    def sample_conditioned(
        self, observed: Mapping[int, int], uniforms: NDArray[np.float64]
    ) -> NDArray[np.int64]:
        self._validate_observed(observed)
        uniforms = np.asarray(uniforms, dtype=float)
        if uniforms.shape != (self.d,) or np.any((uniforms < 0) | (uniforms >= 1)):
            raise ValueError("uniforms must have shape (d,) and lie in [0,1)")
        result = np.empty(self.d, dtype=int)
        for i in range(self.d):
            if i in observed:
                result[i] = observed[i]
            else:
                result[i] = int(np.searchsorted(np.cumsum(self.probs[i]), uniforms[i]))
        return result

    def log_prob(self, x: NDArray[np.int64]) -> float:
        x = np.asarray(x, dtype=int)
        if x.shape != (self.d,) or np.any((x < 0) | (x >= self.q)):
            raise ValueError("x must have shape (d,) with valid tokens")
        return float(self.log_probs[np.arange(self.d), x].sum())

