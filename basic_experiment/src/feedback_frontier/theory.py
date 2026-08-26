from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq


def kl_rademacher_mean(m: float) -> float:
    if not -1.0 <= m <= 1.0:
        raise ValueError(f"m must be in [-1, 1], got {m}")
    if abs(m) == 1.0:
        return math.log(2.0)
    if abs(m) < 1e-15:
        return 0.0
    return 0.5 * (
        (1.0 + m) * math.log1p(m) + (1.0 - m) * math.log1p(-m)
    )


def inverse_kl_mean(a: float) -> float:
    if a < 0.0 or a > math.log(2.0):
        raise ValueError(f"a must be in [0, log(2)], got {a}")
    if a == 0.0:
        return 0.0
    if math.isclose(a, math.log(2.0), abs_tol=1e-14):
        return 1.0
    return float(brentq(lambda m: kl_rademacher_mean(m) - a, 0.0, 1 - 1e-14))


def exact_binary_interaction_gain(width: int, epsilon: float) -> float:
    if width < 1:
        raise ValueError(f"width must be positive, got {width}")
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be nonnegative, got {epsilon}")
    if epsilon >= width * math.log(2.0):
        return 1.0
    return inverse_kl_mean(epsilon / width) ** width


def frontier_width(
    round_of_position: Sequence[int], support: Sequence[int]
) -> int:
    if not support:
        raise ValueError("support must be non-empty")
    if min(support) < 0 or max(support) >= len(round_of_position):
        raise ValueError("support position out of range")
    final_round = max(round_of_position[i] for i in support)
    return sum(round_of_position[i] == final_round for i in support)


def balanced_width_probability(
    n: int, capacities: Sequence[int], support_size: int, width: int
) -> float:
    if sum(capacities) != n:
        raise ValueError("capacities must sum to n")
    if not 1 <= width <= support_size <= n:
        return 0.0
    total = 0
    prefix = 0
    for capacity in capacities:
        if width <= capacity and support_size - width <= prefix:
            total += math.comb(capacity, width) * math.comb(
                prefix, support_size - width
            )
        prefix += capacity
    return total / math.comb(n, support_size)


def gram_delta(standardized_F: NDArray[np.float64]) -> float:
    matrix = np.asarray(standardized_F, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("standardized_F must be square")
    return float(np.linalg.norm(matrix - np.eye(matrix.shape[0]), ord=2))


def um_approximation_lower_bound(delta: float) -> float:
    if not 0.0 <= delta < 1.0:
        raise ValueError(f"delta must lie in [0, 1), got {delta}")
    return (1.0 - delta) / (1.0 + delta)

