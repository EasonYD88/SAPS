import pytest

from feedback_frontier.schedulers.base import Schedule, balanced_capacities


@pytest.mark.parametrize(
    ("d", "rounds", "expected"),
    [
        (12, 1, (12,)),
        (12, 3, (4, 4, 4)),
        (12, 4, (3, 3, 3, 3)),
        (12, 5, (3, 3, 2, 2, 2)),
    ],
)
def test_balanced_capacities(d: int, rounds: int, expected: tuple[int, ...]) -> None:
    assert balanced_capacities(d, rounds) == expected


def test_schedule_is_a_partition() -> None:
    schedule = Schedule(((0, 2), (1, 3)), d=4)
    assert schedule.round_of_position == (0, 1, 0, 1)
    assert schedule.capacities == (2, 2)


@pytest.mark.parametrize(
    "batches",
    [((0, 1), (1, 2)), ((0,), (2,)), ((0, 1), (2, 4)), ((), (0, 1, 2))],
)
def test_schedule_rejects_invalid_partition(batches: tuple[tuple[int, ...], ...]) -> None:
    with pytest.raises(ValueError, match="partition"):
        Schedule(batches, d=3)

