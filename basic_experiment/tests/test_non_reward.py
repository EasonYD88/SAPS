import numpy as np
import pytest

from feedback_frontier.generators.categorical_product import CategoricalProduct
from feedback_frontier.rng import SeedBook
from feedback_frontier.schedulers.non_reward import (
    conditional_mutual_information,
    confidence_schedule,
    dependency_cmi_schedule,
    entropy_schedule,
    min_within_batch_tc_schedule,
    random_balanced_schedule,
)


def test_non_reward_schedules_are_balanced_and_deterministic() -> None:
    model = CategoricalProduct(np.array([[3.0, 0.0], [0.0, 0.0], [1.0, -1.0], [0.2, 0.0]]))
    random_a = random_balanced_schedule(4, 2, SeedBook(2), "e1")
    random_b = random_balanced_schedule(4, 2, SeedBook(2), "e1")
    assert random_a == random_b
    assert random_a.capacities == (2, 2)
    assert confidence_schedule(model, 2).capacities == (2, 2)
    assert entropy_schedule(model, 2).capacities == (2, 2)


def test_product_conditional_mutual_information_is_zero() -> None:
    model = CategoricalProduct(np.random.default_rng(1).normal(size=(4, 3)))
    assert conditional_mutual_information(model, {}, 0, 3) == pytest.approx(0.0, abs=1e-12)


def test_dependency_schedules_are_valid() -> None:
    from feedback_frontier.generators.potts_tree import TreePotts

    model = TreePotts.random(
        6, 3, "chain", 0.7, np.random.default_rng(5)
    )
    assert dependency_cmi_schedule(model, 3).capacities == (2, 2, 2)
    assert min_within_batch_tc_schedule(model, 3).capacities == (2, 2, 2)
