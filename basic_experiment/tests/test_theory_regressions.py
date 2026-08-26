from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from feedback_frontier.theory import (
    balanced_width_probability,
    exact_binary_interaction_gain,
    frontier_width,
    gram_delta,
    inverse_kl_mean,
    kl_rademacher_mean,
    um_approximation_lower_bound,
)


ROOT = Path(__file__).resolve().parents[2]
THEORY = ROOT / "math_theory"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_theory_artifact_hashes_are_frozen() -> None:
    assert _sha256(THEORY / "correlated_budgeted_feedback_frontier_report.md") == (
        "d0341f361899927fb6616b7ba059b4cc490a793619e9282758e41b88789486aa"
    )
    assert _sha256(THEORY / "correlated_budgeted_feedback_frontier_results.json") == (
        "fae39d94526f283e59971700a39d128848995ca41eb064dd41e6c6863cb1b12c"
    )


def test_exact_binary_gain_matches_verified_result() -> None:
    assert exact_binary_interaction_gain(3, 0.05) == pytest.approx(
        0.006035048130208016, rel=1e-12
    )
    m = inverse_kl_mean(0.05 / 3)
    assert kl_rademacher_mean(m) == pytest.approx(0.05 / 3)


def test_frontier_width_is_final_batch_multiplicity() -> None:
    assert frontier_width((0, 1, 0, 1), (0, 1, 2)) == 1
    assert frontier_width((0, 0, 1, 1), (0, 2, 3)) == 2


def test_balanced_width_distribution_matches_verified_values() -> None:
    got = [balanced_width_probability(9, (3, 3, 3), 5, w) for w in range(1, 6)]
    assert got[:3] == pytest.approx((5 / 14, 1 / 2, 1 / 7))
    assert sum(got) == pytest.approx(1.0)


def test_near_orthogonal_bound_and_duplicate_projection() -> None:
    F = np.array([[1.0, 0.1], [0.1, 1.0]])
    b = np.array([0.4, -0.7])
    delta = gram_delta(F)
    gamma = float(b @ np.linalg.pinv(F) @ b)
    W = float(b @ b)
    assert W / (1 + delta) <= gamma <= W / (1 - delta)
    assert um_approximation_lower_bound(delta) == pytest.approx(
        (1 - delta) / (1 + delta)
    )
    duplicate_F = np.ones((2, 2))
    duplicate_b = np.array([0.73, 0.73])
    assert duplicate_b @ np.linalg.pinv(duplicate_F) @ duplicate_b == pytest.approx(
        0.73**2
    )


def test_small_epsilon_slope_is_half_width() -> None:
    eps = np.logspace(-8, -5, 12)
    for width in (1, 2, 3, 4):
        gain = np.array([exact_binary_interaction_gain(width, x) for x in eps])
        slope = np.polyfit(np.log(eps), np.log(gain), 1)[0]
        assert slope == pytest.approx(width / 2, abs=5e-4)

