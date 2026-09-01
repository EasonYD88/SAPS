import numpy as np
import pytest

from feedback_frontier.generators.categorical_product import CategoricalProduct
from feedback_frontier.rng import SeedBook
from feedback_frontier.runners import evaluate_planner
from feedback_frontier.schedulers.base import Schedule


def test_seedbook_is_stable() -> None:
    book = SeedBook(7)
    a = book.rng("base", "example-3", 2).random(32)
    b = book.rng("base", "example-3", 2).random(32)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, book.rng("base", "example-4", 2).random(32))


def test_seedbook_rejects_unregistered_namespace() -> None:
    with pytest.raises(ValueError, match="namespace"):
        SeedBook(1).rng("scheduler-name", "random")


def test_adaptation_and_gain_evaluation_use_disjoint_rng_domains() -> None:
    class TrackingSeedBook:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []
            self.book = SeedBook(41)

        def rng(self, namespace: str, *keys: object) -> np.random.Generator:
            self.calls.append((namespace, *keys))
            return self.book.rng(namespace, *keys)

    model = CategoricalProduct(np.zeros((3, 2)))
    schedule = Schedule(((0, 1), (2,)), 3)
    adaptation_book = TrackingSeedBook()
    evaluation_book = TrackingSeedBook()
    evaluate_planner._sample_base_trace(
        model, schedule, adaptation_book, "example", sample_id=0
    )
    evaluate_planner._decode(
        model,
        lambda tokens: float(tokens.sum()),
        schedule,
        epsilon=0.05,
        rollouts=1,
        seedbook=evaluation_book,
        example_id="example",
        controlled=False,
    )
    assert adaptation_book.calls
    assert evaluation_book.calls
    assert {call[2] for call in adaptation_book.calls} == {"adaptation"}
    assert {call[2] for call in evaluation_book.calls} == {"gain-evaluation"}
    assert set(adaptation_book.calls).isdisjoint(evaluation_book.calls)


def test_subset_planning_and_gain_evaluation_use_disjoint_rng_domains() -> None:
    class TrackingSeedBook:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []
            self.book = SeedBook(43)

        def rng(self, namespace: str, *keys: object) -> np.random.Generator:
            self.calls.append((namespace, *keys))
            return self.book.rng(namespace, *keys)

    model = CategoricalProduct(np.zeros((4, 2)))
    reward = lambda tokens: float(tokens.sum())
    book = TrackingSeedBook()
    oracle = evaluate_planner._make_batch_value_oracle(
        model,
        reward,
        rounds=2,
        epsilon=0.05,
        rollouts=1,
        seedbook=book,
        example_id="example",
    )
    oracle.value((0, 1))
    planning_calls = set(book.calls)
    book.calls.clear()
    evaluate_planner._decode(
        model,
        reward,
        Schedule(((0, 1), (2, 3)), 4),
        epsilon=0.05,
        rollouts=1,
        seedbook=book,
        example_id="example",
        controlled=True,
    )
    evaluation_calls = set(book.calls)
    assert planning_calls
    assert evaluation_calls
    assert planning_calls.isdisjoint(evaluation_calls)
