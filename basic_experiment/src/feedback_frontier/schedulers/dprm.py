from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp


def process_value(rewards: NDArray[np.float64], beta: float) -> float:
    values = np.asarray(rewards, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("rewards must be a non-empty vector")
    if abs(beta) < 1e-12:
        return float(values.mean())
    return float((logsumexp(beta * values) - np.log(len(values))) / beta)


def select_top_itemwise(scores: NDArray[np.float64], capacity: int) -> tuple[int, ...]:
    values = np.asarray(scores, dtype=float)
    if not 0 < capacity <= len(values):
        raise ValueError("invalid capacity")
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    return tuple(sorted(order[:capacity]))

