from __future__ import annotations

import itertools

import numpy as np
import pytest

from feedback_frontier.generators.categorical_product import CategoricalProduct
from feedback_frontier.generators.potts_tree import TreePotts


def _enumerated_marginals(model, observed: dict[int, int]) -> np.ndarray:
    states = np.array(list(itertools.product(range(model.q), repeat=model.d)), dtype=int)
    keep = np.ones(len(states), dtype=bool)
    for i, value in observed.items():
        keep &= states[:, i] == value
    states = states[keep]
    probs = np.exp(np.array([model.log_prob(x) for x in states]))
    probs /= probs.sum()
    result = np.full((model.d, model.q), np.nan)
    for i in range(model.d):
        if i not in observed:
            result[i] = [probs[states[:, i] == a].sum() for a in range(model.q)]
    return result


@pytest.mark.parametrize("kind", ["product", "chain", "balanced_tree"])
def test_log_prob_normalizes_and_conditionals_are_exact(kind: str) -> None:
    rng = np.random.default_rng(11)
    node_logits = rng.normal(size=(4, 3))
    if kind == "product":
        model = CategoricalProduct(node_logits)
    else:
        model = TreePotts.random(
            d=4, q=3, topology=kind, coupling=0.7, rng=np.random.default_rng(12),
            node_logits=node_logits,
        )
    states = list(itertools.product(range(3), repeat=4))
    assert sum(np.exp(model.log_prob(np.array(x))) for x in states) == pytest.approx(1.0)
    observed = {0: 2, 3: 1}
    np.testing.assert_allclose(
        model.conditional_marginals(observed),
        _enumerated_marginals(model, observed),
        atol=1e-10,
        equal_nan=True,
    )


def test_zero_coupling_potts_equals_product() -> None:
    logits = np.random.default_rng(2).normal(size=(4, 3))
    product = CategoricalProduct(logits)
    potts = TreePotts.random(
        d=4, q=3, topology="chain", coupling=0.0,
        rng=np.random.default_rng(3), node_logits=logits,
    )
    for x in itertools.product(range(3), repeat=4):
        assert potts.log_prob(np.array(x)) == pytest.approx(product.log_prob(np.array(x)))


def test_conditioned_sampling_uses_supplied_uniforms() -> None:
    model = CategoricalProduct(np.zeros((3, 2)))
    got = model.sample_conditioned({1: 1}, np.array([0.1, 0.9, 0.7]))
    np.testing.assert_array_equal(got, np.array([0, 1, 1]))

