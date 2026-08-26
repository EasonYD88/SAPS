from __future__ import annotations

from dataclasses import dataclass


def balanced_capacities(d: int, rounds: int) -> tuple[int, ...]:
    if d < 1 or not 1 <= rounds <= d:
        raise ValueError(f"require d>=1 and 1<=rounds<=d; got {d}, {rounds}")
    quotient, remainder = divmod(d, rounds)
    return (quotient + 1,) * remainder + (quotient,) * (rounds - remainder)


@dataclass(frozen=True)
class Schedule:
    batches: tuple[tuple[int, ...], ...]
    d: int

    def __post_init__(self) -> None:
        flat = [position for batch in self.batches for position in batch]
        if (
            not self.batches
            or any(not batch for batch in self.batches)
            or len(flat) != self.d
            or sorted(flat) != list(range(self.d))
            or max(map(len, self.batches)) - min(map(len, self.batches)) > 1
        ):
            raise ValueError("schedule batches must form a balanced partition")

    @property
    def capacities(self) -> tuple[int, ...]:
        return tuple(map(len, self.batches))

    @property
    def round_of_position(self) -> tuple[int, ...]:
        rounds = [-1] * self.d
        for round_id, batch in enumerate(self.batches):
            for position in batch:
                rounds[position] = round_id
        return tuple(rounds)

