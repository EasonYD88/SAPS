from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp


class TreePotts:
    def __init__(
        self,
        node_logits: NDArray[np.float64],
        edges: tuple[tuple[int, int], ...],
        edge_logits: dict[tuple[int, int], NDArray[np.float64]],
    ) -> None:
        self.node_logits = np.asarray(node_logits, dtype=float)
        if self.node_logits.ndim != 2:
            raise ValueError("node_logits must have shape (d, q)")
        self.d, self.q = self.node_logits.shape
        self.edges = tuple((min(i, j), max(i, j)) for i, j in edges)
        if len(self.edges) != self.d - 1:
            raise ValueError("edges must define a tree")
        self.edge_logits = {
            (min(i, j), max(i, j)): np.asarray(value, dtype=float)
            for (i, j), value in edge_logits.items()
        }
        self.adj = [[] for _ in range(self.d)]
        for i, j in self.edges:
            self.adj[i].append(j)
            self.adj[j].append(i)
        if set(self.edge_logits) != set(self.edges):
            raise ValueError("edge_logits keys must match edges")
        if any(value.shape != (self.q, self.q) for value in self.edge_logits.values()):
            raise ValueError("each edge potential must have shape (q, q)")
        seen = set()
        stack = [0]
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            stack.extend(j for j in self.adj[i] if j not in seen)
        if len(seen) != self.d:
            raise ValueError("edges must be connected")
        self._log_z = self._log_partition({})

    @classmethod
    def random(
        cls,
        d: int,
        q: int,
        topology: str,
        coupling: float,
        rng: np.random.Generator,
        node_logits: NDArray[np.float64] | None = None,
    ) -> "TreePotts":
        if topology == "chain":
            edges = tuple((i - 1, i) for i in range(1, d))
        elif topology == "balanced_tree":
            edges = tuple(((i - 1) // 2, i) for i in range(1, d))
        else:
            raise ValueError(f"unknown topology: {topology}")
        nodes = rng.normal(size=(d, q)) if node_logits is None else np.asarray(node_logits)
        edge_logits = {edge: coupling * rng.normal(size=(q, q)) for edge in edges}
        return cls(nodes, edges, edge_logits)

    def _edge(self, i: int, j: int) -> NDArray[np.float64]:
        if i < j:
            return self.edge_logits[(i, j)]
        return self.edge_logits[(j, i)].T

    def _validate_observed(self, observed: Mapping[int, int]) -> None:
        if any(not 0 <= i < self.d or not 0 <= a < self.q for i, a in observed.items()):
            raise ValueError("observed position or token out of range")

    def _evidence(self, observed: Mapping[int, int]) -> NDArray[np.float64]:
        evidence = np.zeros((self.d, self.q))
        for i, token in observed.items():
            evidence[i] = -np.inf
            evidence[i, token] = 0.0
        return evidence

    def _messages(self, observed: Mapping[int, int]):
        evidence = self._evidence(observed)

        @lru_cache(maxsize=None)
        def message(i: int, j: int) -> NDArray[np.float64]:
            local = self.node_logits[i] + evidence[i]
            for neighbor in self.adj[i]:
                if neighbor != j:
                    local = local + message(neighbor, i)
            return logsumexp(local[:, None] + self._edge(i, j), axis=0)

        return evidence, message

    def _log_partition(self, observed: Mapping[int, int]) -> float:
        evidence, message = self._messages(observed)
        root = self.node_logits[0] + evidence[0]
        for neighbor in self.adj[0]:
            root = root + message(neighbor, 0)
        return float(logsumexp(root))

    def conditional_marginals(
        self, observed: Mapping[int, int]
    ) -> NDArray[np.float64]:
        self._validate_observed(observed)
        evidence, message = self._messages(observed)
        result = np.full((self.d, self.q), np.nan)
        for i in range(self.d):
            if i in observed:
                continue
            local = self.node_logits[i] + evidence[i]
            for neighbor in self.adj[i]:
                local = local + message(neighbor, i)
            local = local - logsumexp(local)
            result[i] = np.exp(local)
        return result

    def sample_conditioned(
        self, observed: Mapping[int, int], uniforms: NDArray[np.float64]
    ) -> NDArray[np.int64]:
        self._validate_observed(observed)
        uniforms = np.asarray(uniforms, dtype=float)
        if uniforms.shape != (self.d,) or np.any((uniforms < 0) | (uniforms >= 1)):
            raise ValueError("uniforms must have shape (d,) and lie in [0,1)")
        current = dict(observed)
        for i in range(self.d):
            if i in current:
                continue
            probs = self.conditional_marginals(current)[i]
            current[i] = int(np.searchsorted(np.cumsum(probs), uniforms[i]))
        return np.array([current[i] for i in range(self.d)], dtype=int)

    def log_prob(self, x: NDArray[np.int64]) -> float:
        x = np.asarray(x, dtype=int)
        if x.shape != (self.d,) or np.any((x < 0) | (x >= self.q)):
            raise ValueError("x must have shape (d,) with valid tokens")
        energy = float(self.node_logits[np.arange(self.d), x].sum())
        for i, j in self.edges:
            energy += float(self.edge_logits[(i, j)][x[i], x[j]])
        return energy - self._log_z

