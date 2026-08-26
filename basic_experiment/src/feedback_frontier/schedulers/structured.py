from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from feedback_frontier.candidates.libraries import CandidateLibrary
from feedback_frontier.schedulers.base import Schedule
from feedback_frontier.theory import frontier_width


def all_colorings_with_capacities(
    n: int, capacities: Sequence[int]
) -> Iterable[tuple[int, ...]]:
    colors = [-1] * n

    def visit(i: int, remaining: list[int]):
        if i == n:
            yield tuple(colors)
            return
        for color, count in enumerate(remaining):
            if count:
                colors[i] = color
                remaining[color] -= 1
                yield from visit(i + 1, remaining)
                remaining[color] += 1

    yield from visit(0, list(capacities))


def _schedule(colors: Sequence[int], capacities: Sequence[int]) -> Schedule:
    batches = tuple(
        tuple(i for i, color in enumerate(colors) if color == ell)
        for ell in range(len(capacities))
    )
    return Schedule(batches, len(colors))


@dataclass(frozen=True)
class StructuredResult:
    schedule: Schedule
    objective: float
    optimality_status: str
    state_evaluations: int


def pairwise_milp(
    n: int,
    capacities: Sequence[int],
    edges: Sequence[tuple[int, int, float]],
) -> StructuredResult:
    L = len(capacities)
    nx = n * L
    nz = len(edges) * L
    c = np.zeros(nx + nz)
    for edge_id, (_, _, weight) in enumerate(edges):
        c[nx + edge_id * L : nx + (edge_id + 1) * L] = weight
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    for i in range(n):
        row = np.zeros_like(c)
        row[i * L : (i + 1) * L] = 1
        rows.append(row); lower.append(1); upper.append(1)
    for ell, capacity in enumerate(capacities):
        row = np.zeros_like(c)
        row[ell:nx:L] = 1
        rows.append(row); lower.append(capacity); upper.append(capacity)
    for edge_id, (i, j, _) in enumerate(edges):
        for ell in range(L):
            z = nx + edge_id * L + ell
            row = np.zeros_like(c); row[z] = 1; row[i * L + ell] = -1
            rows.append(row); lower.append(-np.inf); upper.append(0)
            row = np.zeros_like(c); row[z] = 1; row[j * L + ell] = -1
            rows.append(row); lower.append(-np.inf); upper.append(0)
            row = np.zeros_like(c); row[i * L + ell] = 1; row[j * L + ell] = 1; row[z] = -1
            rows.append(row); lower.append(-np.inf); upper.append(1)
    constraints = LinearConstraint(np.vstack(rows), np.array(lower), np.array(upper))
    result = milp(
        c,
        integrality=np.ones_like(c),
        bounds=Bounds(np.zeros_like(c), np.ones_like(c)),
        constraints=constraints,
        options={"time_limit": 60.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"pairwise MILP failed: {result.message}")
    colors = tuple(int(np.argmax(result.x[i * L : (i + 1) * L])) for i in range(n))
    objective = float(sum(w for i, j, w in edges if colors[i] != colors[j]))
    return StructuredResult(_schedule(colors, capacities), objective, "optimal", int(result.mip_node_count or 0))


@dataclass(frozen=True)
class LaminarNode:
    leaves: tuple[int, ...]
    weight: float
    children: tuple["LaminarNode", ...] = ()


def laminar_dp(
    root: LaminarNode, capacities: Sequence[int], lambdas: Sequence[float]
) -> StructuredResult:
    L = len(capacities)
    n = sum(capacities)

    def solve(node: LaminarNode):
        if not node.children:
            tables = {}
            leaf = node.leaves[0]
            for ell in range(L):
                counts = [0] * L; counts[ell] = 1
                colors = [-1] * n; colors[leaf] = ell
                tables[tuple(counts)] = (0.0, tuple(colors))
            return tables
        current = {(0,) * L: (0.0, (-1,) * n)}
        for child in node.children:
            following = {}
            for counts_a, (value_a, colors_a) in current.items():
                for counts_b, (value_b, colors_b) in solve(child).items():
                    counts = tuple(a + b for a, b in zip(counts_a, counts_b))
                    if any(counts[i] > capacities[i] for i in range(L)):
                        continue
                    colors = tuple(b if b >= 0 else a for a, b in zip(colors_a, colors_b))
                    value = value_a + value_b
                    if counts not in following or value > following[counts][0]:
                        following[counts] = (value, colors)
            current = following
        for counts, (value, colors) in list(current.items()):
            final = max(i for i, count in enumerate(counts) if count)
            current[counts] = (value + node.weight * lambdas[counts[final]], colors)
        return current

    table = solve(root)
    value, colors = table[tuple(capacities)]
    return StructuredResult(_schedule(colors, capacities), value, "optimal", len(table))


def path_capacity_dp(
    edge_weights: Sequence[float],
    capacities: Sequence[int],
    lambda_same: float,
) -> StructuredResult:
    n = len(edge_weights) + 1
    L = len(capacities)
    states = {}
    for color in range(L):
        counts = [0] * L; counts[color] = 1
        states[(tuple(counts), color)] = (0.0, (color,))
    evaluations = 0
    for i, weight in enumerate(edge_weights, start=1):
        following = {}
        for (counts, previous), (value, colors) in states.items():
            for color in range(L):
                if counts[color] >= capacities[color]:
                    continue
                new_counts = list(counts); new_counts[color] += 1
                gain = weight * (1.0 if color != previous else lambda_same)
                key = (tuple(new_counts), color)
                proposal = (value + gain, (*colors, color))
                evaluations += 1
                if key not in following or proposal[0] > following[key][0]:
                    following[key] = proposal
        states = following
    value, colors = max(
        (item for (counts, _), item in states.items() if counts == tuple(capacities)),
        key=lambda item: item[0],
    )
    return StructuredResult(_schedule(colors, capacities), value, "optimal", evaluations)


def capacity_preserving_swaps(
    initial: Schedule,
    library: CandidateLibrary,
    lambdas: Sequence[float],
    max_sweeps: int = 100,
) -> StructuredResult:
    def objective(schedule: Schedule) -> float:
        return float(
            sum(
                weight
                * lambdas[frontier_width(schedule.round_of_position, group)]
                for group, weight in zip(library.groups, library.weights)
            )
        )

    current = initial
    current_value = objective(current)
    evaluations = 1
    for _ in range(max_sweeps):
        best = current
        best_value = current_value
        batches = [list(batch) for batch in current.batches]
        for left in range(len(batches)):
            for right in range(left + 1, len(batches)):
                for i in batches[left]:
                    for j in batches[right]:
                        proposal = [batch.copy() for batch in batches]
                        proposal[left][proposal[left].index(i)] = j
                        proposal[right][proposal[right].index(j)] = i
                        schedule = Schedule(
                            tuple(tuple(sorted(batch)) for batch in proposal), current.d
                        )
                        value = objective(schedule)
                        evaluations += 1
                        if value > best_value or (
                            value == best_value and schedule.batches < best.batches
                        ):
                            best, best_value = schedule, value
        if best_value <= current_value:
            break
        current, current_value = best, best_value
    return StructuredResult(current, current_value, "local_optimum", evaluations)
