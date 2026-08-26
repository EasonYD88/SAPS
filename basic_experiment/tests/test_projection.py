import numpy as np
import pytest

from feedback_frontier.candidates.libraries import CandidateLibrary
from feedback_frontier.estimators.projection import (
    crossfit_geometry,
    fit_moments,
    orthogonal_surrogate,
    projection_energy,
    residualized_marginal,
    ridge_projection_energy,
    standardize_scores,
)
from feedback_frontier.features.path_scores import PathTrace, path_score
from feedback_frontier.generators.categorical_product import CategoricalProduct
from feedback_frontier.rng import SeedBook
from feedback_frontier.runners import evaluate_planner
from feedback_frontier.schedulers.base import Schedule


def test_path_scores_use_all_final_frontier_positions() -> None:
    schedule = Schedule(((0, 1), (2, 3)), d=4)
    probs0 = np.full((4, 3), np.nan)
    probs0[0:2] = 1 / 3
    probs1 = np.full((4, 3), np.nan)
    probs1[2:4] = 1 / 3
    trace = PathTrace(np.array([0, 1, 2, 0]), (probs0, probs1))
    library = CandidateLibrary(((0, 2, 3),), "structural")
    score = path_score(schedule, trace, library)
    assert len(score.values) == 2 * (3 - 1)
    assert {item[1] for item in score.metadata} == {2, 3}


def test_duplicate_feature_does_not_double_count() -> None:
    b = np.array([1.0, 1.0])
    F = np.ones((2, 2))
    assert projection_energy(b, F).value == pytest.approx(1.0)
    assert orthogonal_surrogate(b) == pytest.approx(2.0)


def test_reparameterization_invariance() -> None:
    b = np.array([0.3, -0.7])
    F = np.array([[1.4, 0.2], [0.2, 0.8]])
    A = np.array([[2.0, 1.0], [1.0, 1.0]])
    original = projection_energy(b, F).value
    transformed = projection_energy(A @ b, A @ F @ A.T).value
    assert transformed == pytest.approx(original)


def test_schur_marginal_equals_refit_difference() -> None:
    rng = np.random.default_rng(9)
    X = rng.normal(size=(4000, 5))
    X[:, 3:] += X[:, :2] @ np.array([[0.5, -0.2], [0.3, 0.6]])
    y = X @ np.array([0.2, -0.7, 0.4, 0.8, -0.5]) + rng.normal(size=4000)
    moments = fit_moments(X, y)
    selected, candidate = (0, 1), (3, 4)
    direct = projection_energy(
        moments.b[list(selected + candidate)],
        moments.F[np.ix_(selected + candidate, selected + candidate)],
    ).value - projection_energy(
        moments.b[list(selected)], moments.F[np.ix_(selected, selected)]
    ).value
    assert residualized_marginal(
        moments.b, moments.F, selected, candidate
    ) == pytest.approx(direct, abs=1e-9)


def test_standardization_drops_constant_columns_without_leakage() -> None:
    train = np.array([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]])
    test = np.array([[100.0, 8.0]])
    result = standardize_scores(np.vstack([train, test]), np.array([0, 1, 2]))
    assert result.active_columns == (1,)
    assert result.values[3, 0] == pytest.approx((8 - 4) / 2)


def test_crossfit_never_scores_fold_with_its_own_fit() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 3))
    y = X[:, 0] + rng.normal(scale=0.1, size=50)
    folds = np.arange(50) % 5
    result = crossfit_geometry(X, y, folds)
    assert len(result.folds) == 5
    assert all(not set(f.holdout_indices) & set(f.fit_indices) for f in result.folds)
    assert all(np.isfinite(f.gamma_pinv) for f in result.folds)


def test_ridge_projection_is_separate() -> None:
    b = np.array([1.0, 1.0])
    F = np.ones((2, 2))
    assert ridge_projection_energy(b, F, 0.1).value < projection_energy(b, F).value


def test_runner_estimates_reproducible_per_instance_geometry() -> None:
    estimator = getattr(evaluate_planner, "_estimate_geometry", None)
    assert callable(estimator), "runner must estimate geometry from base trajectories"
    model = CategoricalProduct(np.zeros((4, 2)))
    schedule = Schedule(((1, 2), (0, 3)), d=4)
    library = CandidateLibrary(((0,),), "oracle", (1.0,))

    def reward(tokens: np.ndarray) -> float:
        return float(tokens[0])

    arguments = (
        model,
        reward,
        schedule,
        library,
        SeedBook(17),
        "geometry-example",
    )
    first = estimator(*arguments, n_samples=100, ridge_multiplier=1e-3)
    second = estimator(*arguments, n_samples=100, ridge_multiplier=1e-3)
    assert first == second
    assert first.gamma_pinv > 0.1
    assert np.isfinite(first.gram_delta)
    assert np.isfinite(first.gram_condition)
    assert len(first.fold_gammas) == 5
    assert hasattr(first, "reward_q75")
    assert first.reward_q75 == pytest.approx(1.0)
