from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import logsumexp

from feedback_frontier.rng import SeedBook


def categorical_kl(q: NDArray[np.float64], p: NDArray[np.float64]) -> float:
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = q > 0
    return float(np.sum(q[mask] * (np.log(q[mask]) - np.log(p[mask]))))


def _tilt(
    base_probabilities: NDArray[np.float64],
    values: NDArray[np.float64],
    alpha: float,
) -> NDArray[np.float64]:
    logits = np.log(base_probabilities) + alpha * values
    return np.exp(logits - logsumexp(logits))


@dataclass(frozen=True)
class AlphaCalibration:
    alpha: float
    probabilities: NDArray[np.float64]
    achieved_kl: float
    saturated: bool


class RolloutCache:
    def __init__(self, model, reward, seedbook: SeedBook, example_id: str) -> None:
        self.model = model
        self.reward = reward
        self.seedbook = seedbook
        self.example_id = example_id
        self._action_cache: dict[tuple[tuple[tuple[int, int], ...], int, int], NDArray[np.float64]] = {}
        self.n_model_calls = 0
        self.n_reward_calls = 0

    def action_values(
        self, observed: dict[int, int], position: int, rollouts: int
    ) -> NDArray[np.float64]:
        canonical = tuple(sorted(observed.items()))
        key = (canonical, position, rollouts)
        if key in self._action_cache:
            return self._action_cache[key].copy()
        values = np.empty(self.model.q)
        for token in range(self.model.q):
            rewards = []
            conditioned = dict(observed)
            conditioned[position] = token
            for rollout_id in range(rollouts):
                uniforms = self.seedbook.rng(
                    "rollout",
                    self.example_id,
                    canonical,
                    position,
                    token,
                    rollout_id,
                ).random(self.model.d)
                terminal = self.model.sample_conditioned(conditioned, uniforms)
                self.n_model_calls += 1
                rewards.append(float(self.reward(terminal)))
                self.n_reward_calls += 1
            values[token] = float(np.mean(rewards))
        self._action_cache[key] = values
        return values.copy()


def calibrate_alpha(
    base_probabilities: NDArray[np.float64],
    values: NDArray[np.float64],
    epsilon: float,
) -> AlphaCalibration:
    base = np.asarray(base_probabilities, dtype=float)
    value = np.asarray(values, dtype=float)
    if base.ndim != 1 or value.shape != base.shape or np.any(base <= 0):
        raise ValueError("base probabilities and values must be aligned positive vectors")
    base = base / base.sum()
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    if epsilon == 0 or np.ptp(value) < 1e-14:
        return AlphaCalibration(0.0, base, 0.0, True)
    high = 1.0
    while high < 1e6 and categorical_kl(_tilt(base, value, high), base) < epsilon:
        high *= 2
    maximum = categorical_kl(_tilt(base, value, high), base)
    if maximum < epsilon:
        probs = _tilt(base, value, high)
        return AlphaCalibration(high, probs, maximum, True)
    alpha = float(
        brentq(
            lambda a: categorical_kl(_tilt(base, value, a), base) - epsilon,
            0.0,
            high,
            xtol=1e-12,
        )
    )
    probs = _tilt(base, value, alpha)
    return AlphaCalibration(alpha, probs, categorical_kl(probs, base), False)
