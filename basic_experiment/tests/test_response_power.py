import numpy as np
import pandas as pd
import pytest
from itertools import product

from feedback_frontier.candidates.libraries import CandidateLibrary
from feedback_frontier.estimators.response_power import (
    balanced_random_budgeted_baseline,
    binary_gain,
    binary_response_power,
    budgeted_score,
    calibrate_width_weights,
    first_order_score,
)
from feedback_frontier.schedulers.base import Schedule
from feedback_frontier.generators.categorical_product import CategoricalProduct
from feedback_frontier.rng import SeedBook
from feedback_frontier.runners import evaluate_planner, probe_response


def test_direct_response_power_is_unbiased_for_squared_mean() -> None:
    # For iid Bernoulli paired differences D with E[D] = 1/2, an estimator of
    # (E[D])**2 must average to 1/4 over all two-sample outcomes. Squaring the
    # sample mean has expectation 3/8 and therefore fails this check.
    estimates = [
        probe_response.direct_response_power(
            np.asarray(outcome, dtype=float), np.zeros(2)
        )
        for outcome in product((0.0, 1.0), repeat=2)
    ]
    assert np.mean(estimates) == pytest.approx(0.25)


def test_binary_response_matches_theory_and_is_monotone() -> None:
    values = [binary_gain(w, 0.05) for w in range(1, 7)]
    assert values == sorted(values, reverse=True)
    assert values[2] == pytest.approx(0.006035048130208016)
    assert binary_response_power(3, 0.05) == pytest.approx(values[2] ** 2)


def test_budgeted_score_prefers_narrower_final_frontier() -> None:
    library = CandidateLibrary(((0, 1, 2),), "oracle", (1.0,))
    a = Schedule(((0, 1), (2, 4), (3, 5)), 6)  # final frontier width 1
    b = Schedule(((0, 4), (3, 5), (1, 2)), 6)  # final frontier width 2
    weights = {0.05: (0.0, 1.0, 0.2, 0.03)}
    assert first_order_score(a, library) > first_order_score(b, library)
    assert budgeted_score(a, library, 0.05, weights) > budgeted_score(
        b, library, 0.05, weights
    )


def test_isotonic_calibration_is_nonincreasing() -> None:
    probes = pd.DataFrame(
        {
            "epsilon": [0.05] * 8,
            "width": [1, 1, 2, 2, 3, 3, 4, 4],
            "actual_response_power": [1.0, 0.9, 0.55, 0.65, 0.7, 0.5, 0.1, 0.2],
        }
    )
    result = calibrate_width_weights(probes, minimum_per_width=2)
    values = result[0.05][1:]
    assert all(a >= b >= 0 for a, b in zip(values, values[1:]))


def test_balanced_random_baseline_matches_enumeration() -> None:
    lambdas = (0.0, 1.0, 0.3, 0.1, 0.03, 0.01)
    got = balanced_random_budgeted_baseline(9, (3, 3, 3), 5, lambdas)
    expected = 5 / 14 * 1.0 + 1 / 2 * 0.3 + 1 / 7 * 0.1
    assert got == pytest.approx(expected)


def test_fisher_whitened_probe_hits_path_kl_and_is_reproducible() -> None:
    estimator = getattr(probe_response, "estimate_fisher_response_power", None)
    assert callable(estimator), "direct response must use Fisher-whitened probes"
    model = CategoricalProduct(np.zeros((3, 2)))
    schedule = Schedule(((1, 2), (0,)), d=3)
    library = CandidateLibrary(((0,),), "oracle", (1.0,))

    def reward(tokens: np.ndarray) -> float:
        return float(tokens[0])

    geometry = evaluate_planner._estimate_geometry(
        model,
        reward,
        schedule,
        library,
        SeedBook(23),
        "probe-example",
        n_samples=100,
        ridge_multiplier=1e-3,
    )
    arguments = (
        model,
        reward,
        schedule,
        geometry,
        0.02,
        4,
        32,
        SeedBook(23),
        "probe-example",
        "schedule-id",
    )
    first = estimator(*arguments)
    second = estimator(*arguments)
    assert first == second
    assert first.positive_rank == 1
    assert first.mean_achieved_kl == pytest.approx(0.02, abs=2e-3)
    assert first.response_power > 0


def test_width_restricted_probe_geometry_uses_only_matching_frontier_blocks() -> None:
    model = CategoricalProduct(np.zeros((3, 2)))
    schedule = Schedule(((1, 2), (0,)), d=3)
    library = CandidateLibrary(((0,), (1, 2)), "oracle", (0.5, 0.5))

    def reward(tokens: np.ndarray) -> float:
        return float(tokens[0] + (tokens[1] == tokens[2]))

    geometry = evaluate_planner._estimate_geometry(
        model,
        reward,
        schedule,
        library,
        SeedBook(29),
        "width-probe-example",
        n_samples=100,
        ridge_multiplier=1e-3,
    )
    width_one = probe_response.restrict_geometry_to_width(
        schedule, geometry, width=1
    )
    width_two = probe_response.restrict_geometry_to_width(
        schedule, geometry, width=2
    )
    assert width_one is not None and width_two is not None
    assert {item[0] for item in width_one.active_metadata} == {(0,)}
    assert {item[0] for item in width_two.active_metadata} == {(1, 2)}
    assert np.asarray(width_one.gram).shape == (1, 1)
    assert np.asarray(width_two.gram).shape == (2, 2)
