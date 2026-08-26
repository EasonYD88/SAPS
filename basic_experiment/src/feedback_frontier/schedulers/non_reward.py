from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from feedback_frontier.rng import SeedBook
from feedback_frontier.schedulers.base import Schedule, balanced_capacities


def _from_order(order: np.ndarray, rounds: int) -> Schedule:
    capacities = balanced_capacities(len(order), rounds)
    batches = []
    start = 0
    for capacity in capacities:
        batches.append(tuple(sorted(int(i) for i in order[start : start + capacity])))
        start += capacity
    return Schedule(tuple(batches), len(order))


def random_balanced_schedule(
    d: int, rounds: int, seedbook: SeedBook, example_id: str
) -> Schedule:
    order = seedbook.rng("scheduler_tie", example_id, "random_balanced").permutation(d)
    return _from_order(order, rounds)


def confidence_schedule(model, rounds: int, observed: Mapping[int, int] | None = None) -> Schedule:
    observed = observed or {}
    if observed:
        raise ValueError("whole schedules currently require an empty observed state")
    probs = model.conditional_marginals(observed)
    confidence = np.max(probs, axis=1)
    order = np.lexsort((np.arange(model.d), -confidence))
    return _from_order(order, rounds)


def entropy_schedule(model, rounds: int, observed: Mapping[int, int] | None = None) -> Schedule:
    observed = observed or {}
    if observed:
        raise ValueError("whole schedules currently require an empty observed state")
    probs = model.conditional_marginals(observed)
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-300, 1.0)), axis=1)
    order = np.lexsort((np.arange(model.d), entropy))
    return _from_order(order, rounds)


def conditional_mutual_information(
    model, observed: Mapping[int, int], i: int, j: int
) -> float:
    if i == j or i in observed or j in observed:
        raise ValueError("i and j must be distinct unresolved positions")
    marginal_i = model.conditional_marginals(observed)[i]
    joint = np.zeros((model.q, model.q))
    for a in range(model.q):
        conditioned = dict(observed)
        conditioned[i] = a
        joint[a] = marginal_i[a] * model.conditional_marginals(conditioned)[j]
    marginal_j = joint.sum(axis=0)
    denominator = marginal_i[:, None] * marginal_j[None, :]
    mask = joint > 0
    value = float(np.sum(joint[mask] * np.log(joint[mask] / denominator[mask])))
    return max(0.0, value)


def dependency_matrix(model, observed: Mapping[int, int] | None = None) -> np.ndarray:
    observed = observed or {}
    matrix = np.zeros((model.d, model.d))
    unresolved = [i for i in range(model.d) if i not in observed]
    for index, i in enumerate(unresolved):
        for j in unresolved[index + 1 :]:
            matrix[i, j] = matrix[j, i] = conditional_mutual_information(
                model, observed, i, j
            )
    return matrix


def dependency_cmi_schedule(model, rounds: int) -> Schedule:
    matrix = dependency_matrix(model)
    capacities = balanced_capacities(model.d, rounds)
    remaining = set(range(model.d))
    batches = []
    for capacity in capacities:
        first = min(
            remaining,
            key=lambda i: (-float(matrix[i, list(remaining)].sum()), i),
        )
        batch = [first]
        remaining.remove(first)
        while len(batch) < capacity:
            chosen = min(
                remaining,
                key=lambda i: (
                    float(np.mean([matrix[i, j] for j in batch])),
                    i,
                ),
            )
            batch.append(chosen)
            remaining.remove(chosen)
        batches.append(tuple(sorted(batch)))
    return Schedule(tuple(batches), model.d)


def min_within_batch_tc_schedule(model, rounds: int) -> Schedule:
    import itertools

    matrix = dependency_matrix(model)
    capacities = balanced_capacities(model.d, rounds)
    remaining = set(range(model.d))
    batches = []
    for capacity in capacities:
        if len(remaining) == capacity:
            chosen = tuple(sorted(remaining))
        else:
            candidates = itertools.combinations(sorted(remaining), capacity)
            chosen = min(
                candidates,
                key=lambda batch: (
                    sum(matrix[i, j] for i, j in itertools.combinations(batch, 2)),
                    batch,
                ),
            )
        batches.append(chosen)
        remaining.difference_update(chosen)
    return Schedule(tuple(batches), model.d)
