from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feedback_frontier.rewards.synthetic import SyntheticReward


@dataclass(frozen=True)
class CandidateLibrary:
    groups: tuple[tuple[int, ...], ...]
    source: str
    analytic_weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        canonical = tuple(sorted({tuple(sorted(group)) for group in self.groups}))
        if canonical != self.groups or any(not group for group in self.groups):
            raise ValueError("groups must be sorted, unique canonical tuples")
        if self.analytic_weights is not None and len(self.analytic_weights) != len(
            self.groups
        ):
            raise ValueError("analytic_weights must align with groups")

    @property
    def weights(self) -> tuple[float, ...]:
        return self.analytic_weights or (1.0,) * len(self.groups)

    def vertex_degrees(self, d: int) -> tuple[int, ...]:
        degree = [0] * d
        for group in self.groups:
            for position in group:
                degree[position] += 1
        return tuple(degree)


def build_oracle_library(reward: SyntheticReward) -> CandidateLibrary:
    by_support: dict[tuple[int, ...], float] = {}
    for term in reward.terms:
        by_support[term.support] = by_support.get(term.support, 0.0) + float(
            np.var(term.table)
        )
    groups = tuple(sorted(by_support))
    raw = np.array([by_support[group] for group in groups], dtype=float)
    weights = raw / raw.sum() if raw.sum() > 0 else np.ones(len(raw)) / len(raw)
    return CandidateLibrary(groups, "oracle", tuple(float(x) for x in weights))


def _tree_paths(
    d: int, edges: tuple[tuple[int, int], ...], max_nodes: int = 4
) -> set[tuple[int, ...]]:
    adjacency = [set() for _ in range(d)]
    for i, j in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)
    paths: set[tuple[int, ...]] = set()

    def visit(path: tuple[int, ...]) -> None:
        if len(path) >= 2:
            paths.add(tuple(sorted(path)))
        if len(path) == max_nodes:
            return
        for neighbor in adjacency[path[-1]]:
            if neighbor not in path:
                visit(path + (neighbor,))

    for start in range(d):
        visit((start,))
    return paths


def build_structural_library(
    d: int, tree_edges: tuple[tuple[int, int], ...]
) -> CandidateLibrary:
    groups = _tree_paths(d, tree_edges)
    for size in range(2, min(4, d) + 1):
        groups.update(tuple(range(start, start + size)) for start in range(d - size + 1))
    return CandidateLibrary(tuple(sorted(groups)), "structural")


def build_random_matched_library(
    structural_library: CandidateLibrary, rng: np.random.Generator
) -> CandidateLibrary:
    groups = [list(group) for group in structural_library.groups]
    original = tuple(tuple(group) for group in groups)
    accepted = 0
    for _ in range(10_000):
        a, b = rng.choice(len(groups), size=2, replace=False)
        ia = int(rng.integers(len(groups[a])))
        ib = int(rng.integers(len(groups[b])))
        va, vb = groups[a][ia], groups[b][ib]
        if va == vb or vb in groups[a] or va in groups[b]:
            continue
        proposal_a = sorted(groups[a][:ia] + [vb] + groups[a][ia + 1 :])
        proposal_b = sorted(groups[b][:ib] + [va] + groups[b][ib + 1 :])
        others = {
            tuple(group) for index, group in enumerate(groups) if index not in (a, b)
        }
        if tuple(proposal_a) in others or tuple(proposal_b) in others:
            continue
        groups[a], groups[b] = proposal_a, proposal_b
        accepted += 1
    result = tuple(sorted(tuple(group) for group in groups))
    if accepted == 0 or result == original or len(set(result)) != len(result):
        raise RuntimeError("could not construct a distinct degree-matched library")
    return CandidateLibrary(result, "random_matched")


def build_nested_diagnostic_library(
    structural_library: CandidateLibrary, oracle_library: CandidateLibrary
) -> CandidateLibrary:
    groups = tuple(sorted(set(structural_library.groups) & set(oracle_library.groups)))
    return CandidateLibrary(groups, "nested_diagnostic")

