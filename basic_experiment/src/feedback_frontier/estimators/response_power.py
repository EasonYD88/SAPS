from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from feedback_frontier.candidates.libraries import CandidateLibrary
from feedback_frontier.schedulers.base import Schedule
from feedback_frontier.theory import (
    balanced_width_probability,
    exact_binary_interaction_gain,
    frontier_width,
)


def binary_gain(width: int, epsilon: float) -> float:
    return exact_binary_interaction_gain(width, epsilon)


def binary_response_power(width: int, epsilon: float) -> float:
    return binary_gain(width, epsilon) ** 2


def first_order_score(schedule: Schedule, library: CandidateLibrary) -> float:
    return float(
        sum(
            weight
            for group, weight in zip(library.groups, library.weights)
            if frontier_width(schedule.round_of_position, group) == 1
        )
    )


def budgeted_score(
    schedule: Schedule,
    library: CandidateLibrary,
    epsilon: float,
    width_weights: Mapping[float, Sequence[float]],
) -> float:
    lambdas = width_weights[epsilon]
    return float(
        sum(
            weight * lambdas[frontier_width(schedule.round_of_position, group)]
            for group, weight in zip(library.groups, library.weights)
        )
    )


def _decreasing_pava(values: list[float], weights: list[float]) -> list[float]:
    blocks: list[list[float]] = []
    for value, weight in zip(values, weights):
        blocks.append([value, weight, 1])
        while len(blocks) >= 2 and blocks[-2][0] < blocks[-1][0]:
            right = blocks.pop()
            left = blocks.pop()
            total_weight = left[1] + right[1]
            mean = (left[0] * left[1] + right[0] * right[1]) / total_weight
            blocks.append([mean, total_weight, left[2] + right[2]])
    output: list[float] = []
    for value, _, count in blocks:
        output.extend([max(0.0, value)] * int(count))
    return output


def calibrate_width_weights(
    probes: pd.DataFrame, minimum_per_width: int = 20
) -> dict[float, tuple[float, ...]]:
    required = {"epsilon", "width", "actual_response_power"}
    if not required <= set(probes):
        raise ValueError(f"probes must contain {sorted(required)}")
    result: dict[float, tuple[float, ...]] = {}
    for epsilon, frame in probes.groupby("epsilon", sort=True):
        grouped = frame.groupby("width")["actual_response_power"].agg(["mean", "count"])
        widths = list(range(1, int(grouped.index.max()) + 1))
        if list(grouped.index.astype(int)) != widths or (grouped["count"] < minimum_per_width).any():
            raise ValueError("calibration_inconclusive: missing or undersampled width")
        fitted = _decreasing_pava(
            grouped["mean"].astype(float).tolist(),
            grouped["count"].astype(float).tolist(),
        )
        result[float(epsilon)] = (0.0, *fitted)
    return result


def width_calibration_summary(
    probes: pd.DataFrame,
    q: int,
    epsilons: Sequence[float],
    minimum_per_width: int = 20,
) -> dict[str, object]:
    if q == 2:
        maximum_width = max(probes.get("width", pd.Series(dtype=int)), default=1)
        weights = {
            float(epsilon): (
                0.0,
                *(
                    binary_response_power(width, float(epsilon))
                    for width in range(1, int(maximum_width) + 1)
                ),
            )
            for epsilon in epsilons
        }
        status = "analytic_binary"
        reason = "q=2 theorem calibration"
    else:
        required = {
            "data_split",
            "response_space_empty",
            "actual_response_power",
            "epsilon",
            "width",
        }
        if not required <= set(probes):
            weights = {}
            status = "inconclusive"
            reason = "calibration_inconclusive: missing probe columns"
        else:
            development = probes.loc[
                (probes["data_split"] == "development")
                & ~probes["response_space_empty"]
                & np.isfinite(probes["actual_response_power"])
            ]
            try:
                weights = calibrate_width_weights(
                    development, minimum_per_width=minimum_per_width
                )
                status = "complete"
                reason = "development-only weighted isotonic fit"
            except ValueError as error:
                weights = {}
                status = "inconclusive"
                reason = str(error)
    serialized = {
        str(epsilon): list(values) for epsilon, values in weights.items()
    }
    return {
        "status": status,
        "reason": reason,
        "minimum_per_width": minimum_per_width,
        "weights": serialized,
    }


def balanced_random_budgeted_baseline(
    n: int,
    capacities: Sequence[int],
    support_size: int,
    lambdas: Sequence[float],
) -> float:
    return float(
        sum(
            lambdas[width]
            * balanced_width_probability(n, capacities, support_size, width)
            for width in range(1, support_size + 1)
        )
    )
