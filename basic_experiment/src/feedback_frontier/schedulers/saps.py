from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from feedback_frontier.candidates.libraries import CandidateLibrary
from feedback_frontier.estimators.projection import residualized_marginal
from feedback_frontier.schedulers.base import Schedule
from feedback_frontier.schedulers.structured import (
    StructuredResult,
    all_colorings_with_capacities,
)
from feedback_frontier.theory import frontier_width


def budgeted_objective(
    schedule: Schedule,
    library: CandidateLibrary,
    lambdas: Sequence[float],
) -> float:
    return float(
        sum(
            weight * lambdas[frontier_width(schedule.round_of_position, group)]
            for group, weight in zip(library.groups, library.weights)
        )
    )


@dataclass(frozen=True)
class ShortlistResult:
    schedule: Schedule
    objective: float
    proposal_count: int
    linear_solve_count: int


def residualized_group_weights(
    schedule: Schedule,
    library: CandidateLibrary,
    b: NDArray[np.float64],
    F: NDArray[np.float64],
    metadata: tuple[tuple[tuple[int, ...], int, int], ...],
    pinv_rtol: float = 1e-12,
) -> tuple[tuple[float, ...], int]:
    """Return sequential Schur gains, ordered by frontier width then group."""
    if len(metadata) != len(b) or F.shape != (len(b), len(b)):
        raise ValueError("metadata, b, and F must align")
    columns_by_group: dict[tuple[int, ...], tuple[int, ...]] = {
        group: tuple(index for index, item in enumerate(metadata) if item[0] == group)
        for group in library.groups
    }
    ordered_groups = sorted(
        library.groups,
        key=lambda group: (frontier_width(schedule.round_of_position, group), group),
    )
    selected: tuple[int, ...] = ()
    gains: dict[tuple[int, ...], float] = {}
    linear_solves = 0
    for group in ordered_groups:
        candidate = columns_by_group[group]
        if not candidate:
            gains[group] = 0.0
            continue
        gains[group] = residualized_marginal(
            b, F, selected, candidate, pinv_rtol
        )
        selected = (*selected, *candidate)
        linear_solves += 1
    return tuple(gains[group] for group in library.groups), linear_solves


def rerank_schedule_shortlist(
    schedules: Sequence[Schedule],
    score: Callable[[Schedule], float],
    latency: Callable[[Schedule], float],
) -> ShortlistResult:
    """Rerank unique schedules by objective, latency, then serialization."""
    unique = {schedule.batches: schedule for schedule in schedules}
    if not unique:
        raise ValueError("schedule shortlist must not be empty")
    evaluated = [
        (float(score(schedule)), float(latency(schedule)), schedule)
        for schedule in unique.values()
    ]
    if any(not np.isfinite(value) or not np.isfinite(cost) for value, cost, _ in evaluated):
        raise ValueError("shortlist scores and latencies must be finite")
    objective, _, selected = min(
        evaluated,
        key=lambda item: (-item[0], item[1], item[2].batches),
    )
    return ShortlistResult(
        selected,
        objective,
        proposal_count=len(evaluated),
        linear_solve_count=len(evaluated),
    )


def exhaustive_saps(
    d: int,
    capacities: Sequence[int],
    library: CandidateLibrary,
    lambdas: Sequence[float],
) -> StructuredResult:
    best_schedule = None
    best_value = -float("inf")
    count = 0
    for colors in all_colorings_with_capacities(d, capacities):
        count += 1
        schedule = Schedule(
            tuple(
                tuple(i for i, color in enumerate(colors) if color == ell)
                for ell in range(len(capacities))
            ),
            d,
        )
        value = budgeted_objective(schedule, library, lambdas)
        serialized = schedule.batches
        if (
            value > best_value
            or (
                value == best_value
                and best_schedule is not None
                and serialized < best_schedule.batches
            )
        ):
            best_schedule, best_value = schedule, value
    assert best_schedule is not None
    return StructuredResult(best_schedule, best_value, "optimal", count)
