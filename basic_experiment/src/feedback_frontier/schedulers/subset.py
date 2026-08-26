from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable


class BatchValueOracle:
    def __init__(self, value_fn: Callable[[tuple[int, ...]], float]) -> None:
        self.value_fn = value_fn
        self._cache: dict[tuple[int, ...], float] = {}
        self.evaluations = 0

    def value(self, batch: tuple[int, ...]) -> float:
        key = tuple(sorted(batch))
        if key not in self._cache:
            self._cache[key] = float(self.value_fn(key))
            self.evaluations += 1
        return self._cache[key]


@dataclass(frozen=True)
class SubsetResult:
    batch: tuple[int, ...]
    value: float
    candidate_count: int


def exact_subset(
    unresolved: tuple[int, ...],
    capacity: int,
    oracle: BatchValueOracle,
    d: int,
) -> SubsetResult:
    if d > 14:
        raise ValueError("exact_subset requires d <= 14")
    candidates = list(itertools.combinations(sorted(unresolved), capacity))
    best = min(candidates)
    best_value = oracle.value(best)
    for candidate in candidates[1:]:
        value = oracle.value(candidate)
        if value > best_value or (value == best_value and candidate < best):
            best, best_value = candidate, value
    return SubsetResult(best, best_value, len(candidates))


def beam_subset(
    unresolved: tuple[int, ...],
    capacity: int,
    oracle: BatchValueOracle,
    beam_width: int,
) -> SubsetResult:
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    partials: list[tuple[int, ...]] = [()]
    positions = tuple(sorted(unresolved))
    for _ in range(capacity):
        expanded = {
            tuple(sorted((*partial, position)))
            for partial in partials
            for position in positions
            if position not in partial
        }
        # A full-width beam is exact. Partial values use completed lexicographic
        # fillers as a deterministic optimistic proxy.
        def proxy(partial: tuple[int, ...]) -> float:
            fillers = tuple(i for i in positions if i not in partial)[: capacity - len(partial)]
            return oracle.value(tuple(sorted((*partial, *fillers))))

        partials = sorted(expanded, key=lambda x: (-proxy(x), x))[:beam_width]
    best = partials[0]
    best_value = oracle.value(best)
    for candidate in partials[1:]:
        value = oracle.value(candidate)
        if value > best_value or (value == best_value and candidate < best):
            best, best_value = candidate, value
    return SubsetResult(best, best_value, len(partials))
