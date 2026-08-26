import numpy as np
import pytest

from feedback_frontier.candidates.libraries import (
    build_nested_diagnostic_library,
    build_oracle_library,
    build_random_matched_library,
    build_structural_library,
)
from feedback_frontier.rewards.synthetic import RewardTerm, SyntheticReward, make_reward


def test_reward_terms_evaluate_exactly() -> None:
    unary = RewardTerm((0,), np.array([-1.0, 2.0]))
    pair = RewardTerm((1, 2), np.array([[0.0, 1.0], [3.0, 5.0]]))
    reward = SyntheticReward((unary, pair), name="mixed")
    assert reward(np.array([1, 0, 1])) == pytest.approx(3.0)
    assert reward.supports == ((0,), (1, 2))


def test_modular_factory_is_an_indicator() -> None:
    reward = make_reward(
        "modular", d=6, q=4, edges=((0, 1),), rng=np.random.default_rng(4)
    )
    values = {reward(np.array(x)) for x in np.ndindex(*(4,) * 6)}
    assert values <= {0.0, 1.0}
    assert values == {0.0, 1.0}


def test_candidate_libraries_preserve_contracts() -> None:
    reward = SyntheticReward(
        (RewardTerm((0, 1), np.ones((2, 2))), RewardTerm((2, 3), np.ones((2, 2)))),
        name="pairwise",
    )
    oracle = build_oracle_library(reward)
    structural = build_structural_library(6, ((0, 1), (1, 2), (1, 3), (3, 4), (3, 5)))
    random = build_random_matched_library(structural, np.random.default_rng(8))
    nested = build_nested_diagnostic_library(structural, oracle)
    assert oracle.groups == ((0, 1), (2, 3))
    assert set(nested.groups) == set(structural.groups) & set(oracle.groups)
    assert sorted(map(len, random.groups)) == sorted(map(len, structural.groups))
    assert random.vertex_degrees(6) == structural.vertex_degrees(6)
    assert random.groups != structural.groups

