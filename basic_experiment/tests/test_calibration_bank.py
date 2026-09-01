from __future__ import annotations

from feedback_frontier.rng import NAMESPACES, SeedBook
import hashlib
import json

import pandas as pd
import pytest

from feedback_frontier.runners.calibration_bank import (
    CalibrationTarget,
    build_schedule_bank,
    construct_targeted_schedule,
    summarize_calibration_coverage,
    write_calibration_artifacts,
)
from feedback_frontier.schedulers.base import balanced_capacities
from feedback_frontier.theory import frontier_width


def test_construct_targeted_schedule_exactly_covers_widths_one_through_five() -> None:
    support = (0, 2, 4, 6, 8)
    capacities = balanced_capacities(12, 2)
    for width in range(1, 6):
        schedule = construct_targeted_schedule(
            d=12,
            capacities=capacities,
            support=support,
            target_width=width,
            seedbook=SeedBook(7),
            example_id="s0-i2",
            replicate=3,
        )
        assert schedule.capacities == capacities
        assert frontier_width(schedule.round_of_position, support) == width


def test_targeted_schedule_is_deterministic_and_uses_isolated_rng_namespace() -> None:
    arguments = dict(
        d=12,
        capacities=(6, 6),
        support=(0, 1, 2, 3, 4),
        target_width=3,
        seedbook=SeedBook(11),
        example_id="s1-i3",
        replicate=5,
    )
    assert construct_targeted_schedule(**arguments) == construct_targeted_schedule(
        **arguments
    )
    assert "width_calibration_bank" in NAMESPACES


def test_targeted_schedule_rejects_an_infeasible_width() -> None:
    try:
        construct_targeted_schedule(
            d=12,
            capacities=(4, 4, 4),
            support=(0, 1, 2, 3, 4),
            target_width=5,
            seedbook=SeedBook(0),
            example_id="s0-i0",
            replicate=0,
        )
    except ValueError as error:
        assert "infeasible target width" in str(error)
    else:
        raise AssertionError("expected infeasible target width to be rejected")


def test_schedule_bank_is_development_only_and_method_independent() -> None:
    targets = [
        CalibrationTarget(
            seed=0,
            instance_id=2,
            example_id="s0-i2",
            data_split="development",
            reward_name="modular",
            support=(0, 1, 2, 3, 4),
        ),
        CalibrationTarget(
            seed=1,
            instance_id=3,
            example_id="s1-i3",
            data_split="development",
            reward_name="mixed",
            support=(2, 4, 6, 8, 10),
        ),
    ]
    bank = build_schedule_bank(
        d=12,
        rounds=(1, 2, 3, 4),
        targets=targets,
        widths=(1, 2, 3, 4, 5),
        maximum_per_width=4,
        rng_seed=20260826,
    )
    assert len(bank) == 20
    assert all(record.data_split == "development" for record in bank)
    assert {record.target_width for record in bank} == set(range(1, 6))
    assert all(
        frontier_width(record.schedule.round_of_position, record.support)
        == record.target_width
        for record in bank
    )
    assert all("method" not in record.as_dict() for record in bank)
    assert [record.as_dict() for record in bank] == [
        record.as_dict()
        for record in build_schedule_bank(
            d=12,
            rounds=(1, 2, 3, 4),
            targets=targets,
            widths=(1, 2, 3, 4, 5),
            maximum_per_width=4,
            rng_seed=20260826,
        )
    ]


def test_schedule_bank_rejects_held_out_target() -> None:
    with pytest.raises(ValueError, match="development-only"):
        build_schedule_bank(
            d=12,
            rounds=(1, 2),
            targets=[
                CalibrationTarget(
                    seed=0,
                    instance_id=8,
                    example_id="s0-i8",
                    data_split="held_out",
                    reward_name="modular",
                    support=(0, 1, 2, 3, 4),
                )
            ],
            widths=(5,),
            maximum_per_width=1,
            rng_seed=1,
        )


def test_coverage_is_cell_local_and_short_cell_is_inconclusive() -> None:
    records = []
    for epsilon in (0.01, 0.05):
        for width in (1, 2):
            valid = 20 if (epsilon, width) != (0.05, 2) else 19
            records.extend(
                {"epsilon": epsilon, "width": width, "valid": True}
                for _ in range(valid)
            )
            records.extend(
                {"epsilon": epsilon, "width": width, "valid": False}
                for _ in range(24 - valid)
            )
    coverage = summarize_calibration_coverage(
        records,
        epsilons=(0.01, 0.05),
        widths=(1, 2),
        requested_per_cell=24,
        minimum_valid_per_cell=20,
        maximum_per_cell=32,
    )
    short = coverage.loc[
        (coverage.epsilon == 0.05) & (coverage.width == 2)
    ].iloc[0]
    assert short.valid_count == 19
    assert not bool(short.passed)
    assert coverage.passed.sum() == 3


def test_frozen_artifacts_include_hashes_flows_and_resources(tmp_path) -> None:
    targets = [
        CalibrationTarget(
            seed=0,
            instance_id=2,
            example_id="s0-i2",
            data_split="development",
            reward_name="modular",
            support=(0, 1, 2, 3, 4),
        )
    ]
    bank = build_schedule_bank(
        d=12,
        rounds=(1, 2, 3, 4),
        targets=targets,
        widths=(1,),
        maximum_per_width=2,
        rng_seed=9,
    )
    coverage = pd.DataFrame(
        [
            {
                "epsilon": 0.05,
                "width": 1,
                "requested_count": 2,
                "maximum_count": 2,
                "attempted_count": 2,
                "valid_count": 2,
                "invalid_count": 0,
                "minimum_valid_count": 2,
                "passed": True,
            }
        ]
    )
    hashes = write_calibration_artifacts(
        tmp_path,
        bank=bank,
        coverage=coverage,
        metadata={
            "epsilon_grid": [0.05],
            "width_grid": [1],
            "support_size_grid": [5],
            "batch_capacities": [[12], [6, 6], [4, 4, 4], [3, 3, 3, 3]],
            "rounds": [1, 2, 3, 4],
            "requested_probes_per_cell": 2,
            "minimum_valid_probes_per_cell": 2,
            "maximum_probes_per_cell": 2,
            "rng_seed": 9,
            "rng_namespace": "width_calibration_bank",
            "reward_instance_ids": ["s0-i2"],
        },
        resources={
            "model_calls": 123,
            "terminal_label_calls": 45,
            "wall_time_sec": 6.5,
        },
        width_calibration={"status": "complete", "weights": {"0.05": [0, 1]}},
        probes=pd.DataFrame(
            [{"epsilon": 0.05, "width": 1, "valid": True}]
        ),
    )
    expected = {
        "calibration_schedule_bank.jsonl",
        "calibration_coverage.csv",
        "calibration_response_probes.parquet",
        "calibration_manifest.json",
        "calibration_manifest.sha256",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    manifest_path = tmp_path / "calibration_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["resources"]["model_calls"] == 123
    assert manifest["resources"]["terminal_label_calls"] == 45
    assert manifest["resources"]["wall_time_sec"] == 6.5
    assert manifest["forbidden_information_flows"] == {
        "held-out evaluation -> calibration": "forbidden",
        "method results -> calibration": "forbidden",
        "calibration probes -> held-out gain estimates": "forbidden",
    }
    for name in (
        "calibration_schedule_bank.jsonl",
        "calibration_coverage.csv",
        "calibration_response_probes.parquet",
    ):
        assert manifest["artifact_sha256"][name] == hashlib.sha256(
            (tmp_path / name).read_bytes()
        ).hexdigest()
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert hashes["calibration_manifest.json"] == manifest_digest
    assert (tmp_path / "calibration_manifest.sha256").read_text().strip() == manifest_digest


def test_manifest_remains_inconclusive_when_weight_fit_is_inconclusive(tmp_path) -> None:
    coverage = pd.DataFrame(
        [
            {
                "epsilon": 0.05,
                "width": 1,
                "requested_count": 1,
                "maximum_count": 1,
                "attempted_count": 1,
                "valid_count": 1,
                "invalid_count": 0,
                "minimum_valid_count": 1,
                "passed": True,
            }
        ]
    )
    write_calibration_artifacts(
        tmp_path,
        bank=[],
        coverage=coverage,
        metadata={},
        resources={"model_calls": 0, "terminal_label_calls": 0, "wall_time_sec": 0},
        width_calibration={"status": "inconclusive", "weights": {}},
        probes=pd.DataFrame([{"epsilon": 0.05, "width": 1, "valid": True}]),
    )
    manifest = json.loads((tmp_path / "calibration_manifest.json").read_text())
    assert manifest["status"] == "inconclusive"
