from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import pinvh


@dataclass(frozen=True)
class Moments:
    b: NDArray[np.float64]
    F: NDArray[np.float64]


@dataclass(frozen=True)
class ProjectionResult:
    value: float
    used_pinv: bool
    ridge: float = 0.0


@dataclass(frozen=True)
class StandardizedScores:
    values: NDArray[np.float64]
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    active_columns: tuple[int, ...]


@dataclass(frozen=True)
class FoldGeometry:
    fold_id: int
    fit_indices: tuple[int, ...]
    holdout_indices: tuple[int, ...]
    gamma_pinv: float


@dataclass(frozen=True)
class CrossFitGeometry:
    folds: tuple[FoldGeometry, ...]


def _validate(scores: NDArray[np.float64], rewards: NDArray[np.float64]) -> None:
    if scores.ndim != 2 or rewards.ndim != 1 or len(scores) != len(rewards):
        raise ValueError("scores must be (N,p) and rewards must be (N,)")
    if len(scores) < 2 or not np.all(np.isfinite(scores)) or not np.all(
        np.isfinite(rewards)
    ):
        raise ValueError("scores/rewards require N>=2 and finite values")


def standardize_scores(
    scores: NDArray[np.float64], fit_indices: NDArray[np.int64]
) -> StandardizedScores:
    values = np.asarray(scores, dtype=float)
    fit = values[np.asarray(fit_indices, dtype=int)]
    mean = fit.mean(axis=0)
    scale = fit.std(axis=0, ddof=1)
    active = np.flatnonzero(scale >= 1e-10)
    if not len(active):
        raise ValueError("all score columns have zero development variance")
    transformed = (values[:, active] - mean[active]) / scale[active]
    return StandardizedScores(
        transformed, mean[active], scale[active], tuple(int(i) for i in active)
    )


def fit_moments(
    scores: NDArray[np.float64], rewards: NDArray[np.float64]
) -> Moments:
    X = np.asarray(scores, dtype=float)
    y = np.asarray(rewards, dtype=float)
    _validate(X, y)
    X = X - X.mean(axis=0)
    y = y - y.mean()
    F = (X.T @ X) / len(X)
    F = (F + F.T) / 2
    b = (X.T @ y) / len(X)
    return Moments(b, F)


def orthogonal_surrogate(b: NDArray[np.float64]) -> float:
    vector = np.asarray(b, dtype=float)
    return float(vector @ vector)


def projection_energy(
    b: NDArray[np.float64], F: NDArray[np.float64], pinv_rtol: float = 1e-12
) -> ProjectionResult:
    vector = np.asarray(b, dtype=float)
    matrix = np.asarray(F, dtype=float)
    if matrix.shape != (len(vector), len(vector)):
        raise ValueError("F and b shapes do not align")
    value = float(vector @ pinvh((matrix + matrix.T) / 2, rtol=pinv_rtol) @ vector)
    return ProjectionResult(max(0.0, value), used_pinv=True)


def ridge_projection_energy(
    b: NDArray[np.float64],
    F: NDArray[np.float64],
    ridge_multiplier: float,
) -> ProjectionResult:
    vector = np.asarray(b, dtype=float)
    matrix = np.asarray(F, dtype=float)
    ridge = ridge_multiplier * float(np.trace(matrix)) / len(vector)
    value = float(vector @ np.linalg.solve(matrix + ridge * np.eye(len(vector)), vector))
    return ProjectionResult(max(0.0, value), used_pinv=False, ridge=ridge)


def residualized_marginal(
    b: NDArray[np.float64],
    F: NDArray[np.float64],
    selected: tuple[int, ...],
    candidate: tuple[int, ...],
    pinv_rtol: float = 1e-12,
) -> float:
    E = np.asarray(candidate, dtype=int)
    if not selected:
        return projection_energy(b[E], F[np.ix_(E, E)], pinv_rtol).value
    A = np.asarray(selected, dtype=int)
    F_A = F[np.ix_(A, A)]
    F_E = F[np.ix_(E, E)]
    F_EA = F[np.ix_(E, A)]
    pinv_A = pinvh(F_A, rtol=pinv_rtol)
    residual_F = F_E - F_EA @ pinv_A @ F_EA.T
    residual_b = b[E] - F_EA @ pinv_A @ b[A]
    value = projection_energy(residual_b, residual_F, pinv_rtol).value
    return 0.0 if abs(value) < 1e-10 else value


def crossfit_geometry(
    scores: NDArray[np.float64],
    rewards: NDArray[np.float64],
    fold_ids: NDArray[np.int64],
) -> CrossFitGeometry:
    X = np.asarray(scores, dtype=float)
    y = np.asarray(rewards, dtype=float)
    folds = np.asarray(fold_ids, dtype=int)
    _validate(X, y)
    result = []
    for fold_id in sorted(set(folds.tolist())):
        fit = np.flatnonzero(folds != fold_id)
        holdout = np.flatnonzero(folds == fold_id)
        standardized = standardize_scores(X, fit)
        moments = fit_moments(standardized.values[fit], y[fit])
        gamma = projection_energy(moments.b, moments.F).value
        result.append(
            FoldGeometry(
                fold_id,
                tuple(int(i) for i in fit),
                tuple(int(i) for i in holdout),
                gamma,
            )
        )
    return CrossFitGeometry(tuple(result))

