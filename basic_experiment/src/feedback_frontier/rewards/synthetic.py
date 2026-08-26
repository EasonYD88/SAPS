from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RewardTerm:
    support: tuple[int, ...]
    table: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not self.support or tuple(sorted(set(self.support))) != self.support:
            raise ValueError("support must be sorted, unique, and non-empty")
        if self.table.ndim != len(self.support):
            raise ValueError("table rank must equal support size")

    def __call__(self, x: NDArray[np.int64]) -> float:
        return float(self.table[tuple(int(x[i]) for i in self.support)])


@dataclass(frozen=True)
class SyntheticReward:
    terms: tuple[RewardTerm, ...]
    name: str

    @property
    def supports(self) -> tuple[tuple[int, ...], ...]:
        return tuple(term.support for term in self.terms)

    def __call__(self, x: NDArray[np.int64]) -> float:
        return float(sum(term(x) for term in self.terms))


def _center(table: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(table, dtype=float) - float(np.mean(table))


def make_reward(
    kind: str,
    d: int,
    q: int,
    edges: tuple[tuple[int, int], ...],
    rng: np.random.Generator,
    delta: float = 0.01,
) -> SyntheticReward:
    if kind == "unary":
        terms = tuple(
            RewardTerm((i,), _center(rng.normal(size=q))) for i in range(d)
        )
    elif kind == "pairwise":
        selected = edges or tuple((i, i + 1) for i in range(d - 1))
        terms = tuple(
            RewardTerm(tuple(sorted(edge)), _center(rng.normal(size=(q, q))))
            for edge in selected
        )
    elif kind == "modular":
        size = min(d, int(rng.integers(3, min(5, d) + 1)))
        support = tuple(sorted(rng.choice(d, size=size, replace=False).tolist()))
        syndrome = int(rng.integers(q))
        table = np.zeros((q,) * size)
        for index in np.ndindex(table.shape):
            table[index] = float(sum(index) % q == syndrome)
        terms = (RewardTerm(support, table),)
    elif kind == "mixed":
        unary = make_reward("unary", d, q, edges, rng)
        modular = make_reward("modular", d, q, edges, rng)
        terms = tuple(
            RewardTerm(term.support, delta * term.table) for term in unary.terms
        ) + modular.terms
    else:
        raise ValueError(f"unknown reward kind: {kind}")
    return SyntheticReward(terms, name=kind)

