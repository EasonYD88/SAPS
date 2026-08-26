from pathlib import Path
from dataclasses import fields
import inspect

import pytest

from feedback_frontier.config import ExperimentConfig
from feedback_frontier.runners import evaluate_planner
from feedback_frontier.schemas import ScheduleScoreRecord, TrajectoryRecord


def test_smoke_grid_is_frozen() -> None:
    cfg = ExperimentConfig.from_yaml(Path("configs/smoke.yaml"))
    assert cfg.d == 12 and cfg.q == 4
    assert cfg.couplings == (0.0, 0.7)
    assert cfg.rewards == ("unary", "pairwise", "modular", "mixed")
    assert cfg.rounds == (1, 2, 3, 4)
    assert cfg.epsilons == (0.01, 0.05, 0.15)
    assert cfg.seeds == (0, 1)
    assert cfg.num_instances == 16
    assert cfg.rollouts == 2
    assert cfg.adaptation_trajectories == 40
    assert cfg.calibration_requested_per_cell == 24
    assert cfg.calibration_max_per_cell == 32


def test_calibration_probe_budget_order_is_validated(tmp_path: Path) -> None:
    path = tmp_path / "bad-calibration-budget.yaml"
    path.write_text(
        "d: 4\nq: 4\nmethods: [confidence]\nrounds: [1, 2]\n"
        "epsilons: [0.05]\nrollouts: 1\nwidth_calibration_minimum: 20\n"
        "calibration_requested_per_cell: 19\n"
        "calibration_max_per_cell: 32\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="minimum <= requested <= maximum"):
        ExperimentConfig.from_yaml(path)


def test_exact_subset_rejects_large_dimension(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "d: 15\nq: 4\nmethods: [subset_exact]\nrounds: [1]\n"
        "epsilons: [0.1]\nrollouts: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exact_subset requires d <= 14"):
        ExperimentConfig.from_yaml(path)


def test_instance_design_round_robins_generator_within_each_reward() -> None:
    planner = getattr(evaluate_planner, "_instance_spec", None)
    assert callable(planner), "runner must expose deterministic instance strata"
    cfg = ExperimentConfig.from_yaml(Path("configs/smoke.yaml"))
    for reward_index, reward_name in enumerate(cfg.rewards):
        first = planner(cfg, reward_index)
        second = planner(cfg, reward_index + len(cfg.rewards))
        third = planner(cfg, reward_index + 2 * len(cfg.rewards))
        assert first == (reward_name, "product", None, None)
        assert second == (reward_name, "potts", "chain", 0.0)
        assert third == (reward_name, "potts", "balanced_tree", 0.0)


def test_instance_split_is_stratified_by_generator_and_reward() -> None:
    splitter = getattr(evaluate_planner, "_instance_splits", None)
    assert callable(splitter), "runner must freeze development/held-out ids"
    cfg = ExperimentConfig.from_yaml(Path("configs/smoke.yaml"))
    splits = splitter(cfg)
    reward_count = len(cfg.rewards)
    for reward_index in range(reward_count):
        assert splits[reward_index] == "development"
        assert splits[reward_index + 3 * reward_count] == "held_out"
        assert splits[reward_index + reward_count] == "development"
        assert splits[reward_index + 2 * reward_count] == "held_out"


def test_raw_record_schemas_include_split_and_join_key() -> None:
    trajectory_fields = {field.name for field in fields(TrajectoryRecord)}
    score_fields = {field.name for field in fields(ScheduleScoreRecord)}
    assert {
        "data_split",
        "schedule_id",
        "planning_action_evaluations",
    } <= trajectory_fields
    assert {
        "data_split",
        "schedule_id",
        "candidate_library",
        "scheduler",
    } <= score_fields


def test_instance_shards_are_disjoint_and_complete() -> None:
    sharder = getattr(evaluate_planner, "_shard_instance_ids", None)
    assert callable(sharder), "runner must expose deterministic instance shards"
    shards = [sharder(10, index, 3) for index in range(3)]
    assert shards == [(0, 3, 6, 9), (1, 4, 7), (2, 5, 8)]
    assert sorted(instance for shard in shards for instance in shard) == list(range(10))
    with pytest.raises(ValueError, match="shard"):
        sharder(10, 3, 3)


def test_runner_accepts_original_instance_ids_for_sharding() -> None:
    assert "instance_ids" in inspect.signature(
        evaluate_planner.run_experiment
    ).parameters
