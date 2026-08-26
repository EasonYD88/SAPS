from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from feedback_frontier.candidates.libraries import CandidateLibrary
from feedback_frontier.schedulers.base import Schedule


@dataclass(frozen=True)
class PathTrace:
    tokens: NDArray[np.int64]
    prebatch_probabilities: tuple[NDArray[np.float64], ...]


@dataclass(frozen=True)
class ScoreVector:
    values: NDArray[np.float64]
    metadata: tuple[tuple[tuple[int, ...], int, int], ...]


def path_score(
    schedule: Schedule, trace: PathTrace, library: CandidateLibrary
) -> ScoreVector:
    if trace.tokens.shape != (schedule.d,) or len(trace.prebatch_probabilities) != len(
        schedule.batches
    ):
        raise ValueError("trace does not align with schedule")
    q = trace.prebatch_probabilities[0].shape[1]
    rounds = schedule.round_of_position
    values: list[float] = []
    metadata: list[tuple[tuple[int, ...], int, int]] = []
    for group in library.groups:
        final_round = max(rounds[i] for i in group)
        frontier = [i for i in group if rounds[i] == final_round]
        probabilities = trace.prebatch_probabilities[final_round]
        for position in frontier:
            if not np.all(np.isfinite(probabilities[position])):
                raise ValueError("frontier probability is missing")
            for category in range(q - 1):
                values.append(
                    float(trace.tokens[position] == category)
                    - float(probabilities[position, category])
                )
                metadata.append((group, position, category))
    return ScoreVector(np.asarray(values), tuple(metadata))

