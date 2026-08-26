from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feedback_frontier.controllers.rollout_local import categorical_kl
from feedback_frontier.rng import SeedBook
from feedback_frontier.schedulers.base import Schedule
from feedback_frontier.theory import frontier_width


def direct_response_power(controlled: np.ndarray, base: np.ndarray) -> float:
    controlled = np.asarray(controlled, dtype=float)
    base = np.asarray(base, dtype=float)
    if controlled.shape != base.shape or controlled.ndim != 1:
        raise ValueError("controlled and base rewards must be aligned vectors")
    return float(np.mean((controlled - base) ** 2))


@dataclass(frozen=True)
class FisherResponseEstimate:
    response_power: float
    mean_achieved_kl: float
    positive_rank: int
    saturated: bool


@dataclass(frozen=True)
class FisherGeometry:
    active_metadata: tuple[tuple[tuple[int, ...], int, int], ...]
    scales: tuple[float, ...]
    gram: tuple[tuple[float, ...], ...]


def restrict_geometry_to_width(
    schedule: Schedule,
    geometry,
    width: int,
) -> FisherGeometry | None:
    if width < 1:
        raise ValueError("width must be positive")
    indices = tuple(
        index
        for index, (group, _, _) in enumerate(geometry.active_metadata)
        if frontier_width(schedule.round_of_position, group) == width
    )
    if not indices:
        return None
    gram = np.asarray(geometry.gram, dtype=float)[np.ix_(indices, indices)]
    return FisherGeometry(
        active_metadata=tuple(geometry.active_metadata[index] for index in indices),
        scales=tuple(float(geometry.scales[index]) for index in indices),
        gram=tuple(tuple(float(value) for value in row) for row in gram),
    )


def _tilt(probabilities: np.ndarray, logits: np.ndarray, alpha: float) -> np.ndarray:
    shifted = np.log(probabilities) + alpha * logits
    shifted -= float(np.max(shifted))
    weights = np.exp(shifted)
    return weights / weights.sum()


def _sample_path(
    model,
    reward,
    schedule: Schedule,
    direction_logits: np.ndarray,
    alpha: float,
    controlled: bool,
    seedbook: SeedBook,
    example_id: str,
    schedule_id: str,
    epsilon: float,
    direction_id: int,
    rollout_id: int,
) -> tuple[float, float]:
    observed: dict[int, int] = {}
    tokens = np.full(model.d, -1, dtype=int)
    path_kl = 0.0
    for round_id, batch in enumerate(schedule.batches):
        marginals = model.conditional_marginals(observed)
        selected: dict[int, int] = {}
        for position in batch:
            base = marginals[position]
            probabilities = (
                _tilt(base, direction_logits[position], alpha)
                if controlled
                else base
            )
            if controlled:
                path_kl += categorical_kl(probabilities, base)
            uniform = seedbook.rng(
                "rollout",
                "fisher-response",
                example_id,
                schedule_id,
                epsilon,
                direction_id,
                rollout_id,
                round_id,
                position,
            ).random()
            token = int(np.searchsorted(np.cumsum(probabilities), uniform))
            selected[position] = token
            tokens[position] = token
        observed.update(selected)
    return float(reward(tokens)), path_kl


def estimate_fisher_response_power(
    model,
    reward,
    schedule: Schedule,
    geometry,
    epsilon: float,
    directions: int,
    rollouts: int,
    seedbook: SeedBook,
    example_id: str,
    schedule_id: str,
) -> FisherResponseEstimate:
    if epsilon <= 0 or directions < 1 or rollouts < 1:
        raise ValueError("epsilon, directions, and rollouts must be positive")
    gram = np.asarray(geometry.gram, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh((gram + gram.T) / 2)
    tolerance = max(1.0, float(np.max(np.abs(eigenvalues)))) * 1e-10
    active = eigenvalues > tolerance
    rank = int(active.sum())
    if rank == 0:
        return FisherResponseEstimate(float("nan"), 0.0, 0, True)
    basis = eigenvectors[:, active]
    roots = np.sqrt(eigenvalues[active])
    powers = []
    achieved = []
    saturated_any = False
    for direction_id in range(directions):
        z = seedbook.rng(
            "proposal",
            "fisher-response-direction",
            example_id,
            schedule_id,
            epsilon,
            direction_id,
        ).normal(size=rank)
        z /= np.linalg.norm(z)
        coefficients = basis @ (z / roots)
        logits = np.zeros((model.d, model.q))
        for coefficient, metadata, scale in zip(
            coefficients, geometry.active_metadata, geometry.scales
        ):
            _, position, category = metadata
            logits[position, category] += coefficient / scale

        def mean_kl(alpha: float) -> float:
            return float(
                np.mean(
                    [
                        _sample_path(
                            model,
                            reward,
                            schedule,
                            logits,
                            alpha,
                            True,
                            seedbook,
                            example_id,
                            schedule_id,
                            epsilon,
                            direction_id,
                            rollout_id,
                        )[1]
                        for rollout_id in range(rollouts)
                    ]
                )
            )

        high = 1.0
        while high < 1e4 and mean_kl(high) < epsilon:
            high *= 2
        maximum = mean_kl(high)
        saturated = maximum < epsilon
        saturated_any = saturated_any or saturated
        if saturated:
            alpha = high
        else:
            low = 0.0
            for _ in range(40):
                midpoint = (low + high) / 2
                value = mean_kl(midpoint)
                if value < epsilon:
                    low = midpoint
                else:
                    high = midpoint
                if abs(value - epsilon) < 1e-6:
                    break
            alpha = (low + high) / 2
        controlled_rewards = []
        base_rewards = []
        direction_kls = []
        for rollout_id in range(rollouts):
            controlled_reward, path_kl = _sample_path(
                model,
                reward,
                schedule,
                logits,
                alpha,
                True,
                seedbook,
                example_id,
                schedule_id,
                epsilon,
                direction_id,
                rollout_id,
            )
            base_reward, _ = _sample_path(
                model,
                reward,
                schedule,
                logits,
                0.0,
                False,
                seedbook,
                example_id,
                schedule_id,
                epsilon,
                direction_id,
                rollout_id,
            )
            controlled_rewards.append(controlled_reward)
            base_rewards.append(base_reward)
            direction_kls.append(path_kl)
        gain = float(np.mean(controlled_rewards) - np.mean(base_rewards))
        powers.append(gain**2)
        achieved.append(float(np.mean(direction_kls)))
    return FisherResponseEstimate(
        float(np.mean(powers)),
        float(np.mean(achieved)),
        rank,
        saturated_any,
    )
