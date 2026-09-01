from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from feedback_frontier.rng import SeedBook
from feedback_frontier.schedulers.base import Schedule, balanced_capacities
from feedback_frontier.theory import frontier_width


CALIBRATION_RNG_NAMESPACE = "width_calibration_bank"


@dataclass(frozen=True)
class CalibrationTarget:
    seed: int
    instance_id: int
    example_id: str
    data_split: str
    reward_name: str
    support: tuple[int, ...]


@dataclass(frozen=True)
class CalibrationSchedule:
    probe_id: str
    schedule_id: str
    seed: int
    instance_id: int
    example_id: str
    data_split: str
    reward_name: str
    support: tuple[int, ...]
    support_size: int
    target_width: int
    rounds: int
    capacities: tuple[int, ...]
    replicate: int
    schedule: Schedule

    def as_dict(self) -> dict[str, object]:
        return {
            "probe_id": self.probe_id,
            "schedule_id": self.schedule_id,
            "seed": self.seed,
            "instance_id": self.instance_id,
            "example_id": self.example_id,
            "data_split": self.data_split,
            "reward_name": self.reward_name,
            "support": list(self.support),
            "support_size": self.support_size,
            "target_width": self.target_width,
            "rounds": self.rounds,
            "capacities": list(self.capacities),
            "replicate": self.replicate,
            "schedule": [list(batch) for batch in self.schedule.batches],
            "rng_namespace": CALIBRATION_RNG_NAMESPACE,
        }


def construct_targeted_schedule(
    d: int,
    capacities: Sequence[int],
    support: Sequence[int],
    target_width: int,
    seedbook: SeedBook,
    example_id: str,
    replicate: int,
) -> Schedule:
    """Construct a balanced schedule whose support has the requested width."""
    capacities = tuple(int(value) for value in capacities)
    support = tuple(sorted(int(value) for value in support))
    if (
        len(capacities) < 2
        or sum(capacities) != d
        or target_width < 1
        or target_width > len(support)
        or target_width > capacities[-1]
        or len(support) - target_width > sum(capacities[:-1])
    ):
        raise ValueError("infeasible target width for capacities and support")
    if not support or len(set(support)) != len(support) or any(
        position < 0 or position >= d for position in support
    ):
        raise ValueError("support must contain distinct positions within dimension")

    rng = seedbook.rng(
        CALIBRATION_RNG_NAMESPACE,
        example_id,
        capacities,
        support,
        target_width,
        replicate,
    )
    shuffled_support = list(rng.permutation(support))
    final_support = shuffled_support[:target_width]
    earlier_support = shuffled_support[target_width:]
    other_positions = [position for position in range(d) if position not in support]
    other_positions = list(rng.permutation(other_positions))

    batches: list[list[int]] = [[] for _ in capacities]
    for position in earlier_support:
        for batch_id in range(len(capacities) - 1):
            if len(batches[batch_id]) < capacities[batch_id]:
                batches[batch_id].append(int(position))
                break
    batches[-1].extend(int(position) for position in final_support)
    for position in other_positions:
        for batch_id, capacity in enumerate(capacities):
            if len(batches[batch_id]) < capacity:
                batches[batch_id].append(int(position))
                break
    return Schedule(
        tuple(tuple(sorted(batch)) for batch in batches),
        d,
    )


def build_schedule_bank(
    d: int,
    rounds: Sequence[int],
    targets: Sequence[CalibrationTarget],
    widths: Sequence[int],
    maximum_per_width: int,
    rng_seed: int,
) -> list[CalibrationSchedule]:
    if maximum_per_width < 1:
        raise ValueError("maximum_per_width must be positive")
    if any(target.data_split != "development" for target in targets):
        raise ValueError("calibration schedule bank is development-only")
    bank: list[CalibrationSchedule] = []
    seedbook = SeedBook(rng_seed)
    for width in widths:
        eligible: list[tuple[CalibrationTarget, int, tuple[int, ...]]] = []
        for target in targets:
            if len(target.support) < width:
                continue
            for round_count in rounds:
                capacities = balanced_capacities(d, int(round_count))
                if (
                    len(capacities) >= 2
                    and width <= capacities[-1]
                    and len(target.support) - width <= sum(capacities[:-1])
                ):
                    eligible.append((target, int(round_count), capacities))
        for replicate in range(maximum_per_width):
            if not eligible:
                break
            target, round_count, capacities = eligible[replicate % len(eligible)]
            schedule = construct_targeted_schedule(
                d=d,
                capacities=capacities,
                support=target.support,
                target_width=int(width),
                seedbook=seedbook,
                example_id=target.example_id,
                replicate=replicate,
            )
            if frontier_width(schedule.round_of_position, target.support) != width:
                raise RuntimeError("targeted calibration schedule width mismatch")
            identity = {
                "rng_namespace": CALIBRATION_RNG_NAMESPACE,
                "rng_seed": rng_seed,
                "example_id": target.example_id,
                "support": target.support,
                "target_width": int(width),
                "rounds": round_count,
                "replicate": replicate,
                "schedule": schedule.batches,
            }
            digest = hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode("utf-8")
            ).hexdigest()
            bank.append(
                CalibrationSchedule(
                    probe_id=digest[:24],
                    schedule_id=digest[:16],
                    seed=target.seed,
                    instance_id=target.instance_id,
                    example_id=target.example_id,
                    data_split=target.data_split,
                    reward_name=target.reward_name,
                    support=target.support,
                    support_size=len(target.support),
                    target_width=int(width),
                    rounds=round_count,
                    capacities=capacities,
                    replicate=replicate,
                    schedule=schedule,
                )
            )
    return bank


def summarize_calibration_coverage(
    records: Sequence[dict[str, object]],
    epsilons: Sequence[float],
    widths: Sequence[int],
    requested_per_cell: int,
    minimum_valid_per_cell: int,
    maximum_per_cell: int,
) -> pd.DataFrame:
    rows = []
    for epsilon in epsilons:
        for width in widths:
            cell = [
                record
                for record in records
                if float(record["epsilon"]) == float(epsilon)
                and int(record["width"]) == int(width)
            ]
            valid_count = sum(bool(record["valid"]) for record in cell)
            attempted_count = len(cell)
            rows.append(
                {
                    "epsilon": float(epsilon),
                    "width": int(width),
                    "requested_count": int(requested_per_cell),
                    "maximum_count": int(maximum_per_cell),
                    "attempted_count": attempted_count,
                    "valid_count": valid_count,
                    "invalid_count": attempted_count - valid_count,
                    "minimum_valid_count": int(minimum_valid_per_cell),
                    "passed": valid_count >= minimum_valid_per_cell,
                }
            )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_calibration_artifacts(
    run_dir: Path,
    bank: Sequence[CalibrationSchedule],
    coverage: pd.DataFrame,
    metadata: dict[str, object],
    resources: dict[str, object],
    width_calibration: dict[str, object],
    probes: pd.DataFrame,
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    bank_path = run_dir / "calibration_schedule_bank.jsonl"
    coverage_path = run_dir / "calibration_coverage.csv"
    manifest_path = run_dir / "calibration_manifest.json"
    probes_path = run_dir / "calibration_response_probes.parquet"
    bank_path.write_text(
        "".join(
            json.dumps(record.as_dict(), sort_keys=True) + "\n" for record in bank
        ),
        encoding="utf-8",
    )
    coverage.to_csv(coverage_path, index=False)
    probes.to_parquet(probes_path, index=False)
    complete = (
        bool(len(coverage))
        and bool(coverage["passed"].all())
        and width_calibration.get("status") == "complete"
    )
    manifest = {
        "status": "complete" if complete else "inconclusive",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **metadata,
        "schedule_ids": [record.schedule_id for record in bank],
        "probe_ids": [record.probe_id for record in bank],
        "width_calibration": width_calibration,
        "resources": resources,
        "forbidden_information_flows": {
            "held-out evaluation -> calibration": "forbidden",
            "method results -> calibration": "forbidden",
            "calibration probes -> held-out gain estimates": "forbidden",
        },
        "artifact_sha256": {
            bank_path.name: _sha256(bank_path),
            coverage_path.name: _sha256(coverage_path),
            probes_path.name: _sha256(probes_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_digest = _sha256(manifest_path)
    (run_dir / "calibration_manifest.sha256").write_text(
        manifest_digest + "\n", encoding="utf-8"
    )
    return {
        bank_path.name: _sha256(bank_path),
        coverage_path.name: _sha256(coverage_path),
        probes_path.name: _sha256(probes_path),
        manifest_path.name: manifest_digest,
    }
