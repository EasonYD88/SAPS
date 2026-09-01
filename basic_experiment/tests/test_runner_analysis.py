from __future__ import annotations

import json
import hashlib
import importlib.util
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from feedback_frontier.analysis.aggregate import analyze_run
from feedback_frontier.cli import main
from feedback_frontier.config import ExperimentConfig
from feedback_frontier.runners.evaluate_planner import run_experiment
from feedback_frontier.runners import evaluate_planner
from feedback_frontier.schemas import ScheduleScoreRecord, TrajectoryRecord


class _CalibrationFakeModel:
    d = 12
    q = 4

    def conditional_marginals(self, observed):
        del observed
        return np.full((self.d, self.q), 1.0 / self.q)


class _CalibrationFakeReward:
    name = "modular"
    supports = ((0, 1, 2, 3, 4),)

    def __call__(self, tokens):
        return float(np.asarray(tokens).sum())


def test_width_calibration_run_is_development_only_and_freezes_resources(
    tmp_path, monkeypatch
) -> None:
    cfg = ExperimentConfig(
        d=12,
        q=4,
        methods=("confidence",),
        rounds=(1, 2, 3, 4),
        epsilons=(0.05,),
        rollouts=1,
        num_instances=4,
        couplings=(0.0,),
        topologies=("chain", "balanced_tree"),
        rewards=("modular",),
        candidate_libraries=("oracle",),
        seeds=(0,),
        response_directions=1,
        adaptation_trajectories=10,
        width_calibration_minimum=1,
        calibration_requested_per_cell=1,
        calibration_max_per_cell=1,
        bootstrap_replicates=10,
        theory_report_sha256="d0341f361899927fb6616b7ba059b4cc490a793619e9282758e41b88789486aa",
        theory_results_sha256="fae39d94526f283e59971700a39d128848995ca41eb064dd41e6c6863cb1b12c",
    )
    requested_instance_ids = []

    def fake_instance(config, seed, instance_id):
        del config, seed
        requested_instance_ids.append(instance_id)
        return None, _CalibrationFakeModel(), _CalibrationFakeReward(), "product", ()

    def fake_geometry(model, reward, schedule, library, seedbook, example_id, n_samples, ridge_multiplier):
        del schedule, library, seedbook, example_id, n_samples, ridge_multiplier
        model.conditional_marginals({})
        reward(np.zeros(model.d, dtype=int))
        return evaluate_planner.GeometryEstimate(
            gamma_pinv=1.0,
            gamma_ridge=1.0,
            gram_delta=0.0,
            gram_condition=1.0,
            fold_gammas=(1.0,) * 5,
            reward_q75=0.0,
            active_metadata=(((0, 1, 2, 3, 4), 0, 0),),
            b=(1.0,),
            scales=(1.0,),
            gram=((1.0,),),
        )

    def fake_response(model, reward, schedule, geometry, epsilon, directions, rollouts, seedbook, example_id, schedule_id):
        del schedule, geometry, directions, rollouts, seedbook, example_id, schedule_id
        model.conditional_marginals({})
        reward(np.zeros(model.d, dtype=int))
        return evaluate_planner.FisherResponseEstimate(
            response_power=float(epsilon),
            mean_achieved_kl=float(epsilon),
            positive_rank=1,
            saturated=False,
        )

    monkeypatch.setattr(evaluate_planner, "_instance", fake_instance)
    monkeypatch.setattr(evaluate_planner, "_estimate_geometry", fake_geometry)
    monkeypatch.setattr(
        evaluate_planner, "estimate_fisher_response_power", fake_response
    )
    run_dir = tmp_path / "calibration"
    evaluate_planner.run_width_calibration(cfg, run_dir)

    assert sorted(set(requested_instance_ids)) == [0, 1]
    manifest = json.loads((run_dir / "calibration_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["development_ids"] == ["s0-i0", "s0-i1"]
    assert manifest["resources"]["model_calls"] == 10
    assert manifest["resources"]["terminal_label_calls"] == 10
    assert manifest["resources"]["wall_time_sec"] > 0
    coverage = pd.read_csv(run_dir / "calibration_coverage.csv")
    assert set(coverage.width) == {1, 2, 3, 4, 5}
    assert coverage.valid_count.eq(1).all()
    bank = [
        json.loads(line)
        for line in (run_dir / "calibration_schedule_bank.jsonl").read_text().splitlines()
    ]
    assert {row["data_split"] for row in bank} == {"development"}
    assert all("method" not in row for row in bank)
    calibration, digest = evaluate_planner._load_frozen_calibration(
        run_dir / "calibration_manifest.json", cfg
    )
    assert calibration["status"] == "complete"
    assert len(digest) == 64
    with (run_dir / "calibration_schedule_bank.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        evaluate_planner._load_frozen_calibration(
            run_dir / "calibration_manifest.json", cfg
        )


def test_cli_calibrate_dispatches_to_independent_stage_a(tmp_path, monkeypatch) -> None:
    config_path = Path("configs/smoke_calibration.yaml").resolve()
    calls = []

    def fake_calibration(config, run_dir):
        calls.append((config, run_dir))

    monkeypatch.setattr(evaluate_planner, "run_width_calibration", fake_calibration)
    monkeypatch.chdir(tmp_path)
    assert main(
        [
            "calibrate",
            "--config",
            str(config_path),
            "--run-id",
            "stage-a-test",
        ]
    ) == 0
    assert len(calls) == 1
    assert calls[0][1] == Path("outputs/stage-a-test")


def test_tiny_run_and_analysis_contract(tmp_path) -> None:
    cfg = ExperimentConfig(
        d=6,
        q=3,
        methods=("random_balanced", "confidence", "saps_diagonal", "b_saps_budgeted"),
        rounds=(2, 3),
        epsilons=(0.05,),
        rollouts=1,
        num_instances=2,
        couplings=(0.0, 0.7),
        topologies=("chain",),
        rewards=("unary", "pairwise"),
        candidate_libraries=("oracle",),
        seeds=(0,),
        response_directions=2,
        bootstrap_replicates=100,
        theory_report_sha256="d0341f361899927fb6616b7ba059b4cc490a793619e9282758e41b88789486aa",
        theory_results_sha256="fae39d94526f283e59971700a39d128848995ca41eb064dd41e6c6863cb1b12c",
    )
    run_dir = tmp_path / "tiny"
    run_experiment(cfg, run_dir)
    for name in (
        "schedule_scores.csv",
        "trajectory_results.parquet",
        "response_probes.parquet",
        "experiment_manifest.json",
    ):
        assert (run_dir / name).is_file()
    manifest = json.loads((run_dir / "experiment_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["non_reward_dependency_status"] == (
        "pairwise conditional-MI/TC surrogate"
    )
    trajectories = pd.read_parquet(run_dir / "trajectory_results.parquet")
    assert set(trajectories.columns) == {
        field.name for field in fields(TrajectoryRecord)
    }
    expected = (
        cfg.num_instances
        * len(cfg.candidate_libraries)
        * len(cfg.rounds)
        * len(cfg.epsilons)
        * len(cfg.methods)
    )
    assert len(trajectories) == expected
    assert trajectories.groupby(
        ["example_id", "candidate_library", "num_rounds", "epsilon_target"]
    ).scheduler.nunique().eq(len(cfg.methods)).all()
    structured = trajectories.scheduler.isin(
        ["saps_diagonal", "b_saps_budgeted"]
    )
    assert trajectories.loc[structured, "planning_state_evaluations"].gt(0).all()
    assert trajectories.loc[structured, "planning_action_evaluations"].eq(0).all()
    assert trajectories.wall_time_sec.ge(trajectories.planning_time_sec).all()
    assert trajectories.wall_time_sec.gt(trajectories.planning_time_sec).any()
    assert np.isfinite(trajectories.gamma_crossfit).all()
    assert trajectories.adaptation_terminal_labels.eq(
        cfg.adaptation_trajectories
    ).all()
    schedule_scores = pd.read_csv(run_dir / "schedule_scores.csv")
    assert np.isfinite(
        schedule_scores[
            ["score_confidence", "score_entropy", "score_dependency"]
        ]
    ).all().all()
    protocol = manifest["evaluation_protocol"]
    assert protocol["name"] == (
        "held-out_reward_instance_fixed-budget_few-shot_adaptation"
    )
    assert protocol["zero_shot"] is False
    assert protocol["adaptation_gain_rng_disjoint"] is True
    assert protocol["schedule_planning_gain_rng_disjoint"] is True

    report = analyze_run(run_dir, bootstrap_replicates=100)
    assert report["decision"] in {"GO", "NO_GO", "INCONCLUSIVE"}
    assert report["screening_decision"] in {"GO", "NO_GO", "INCONCLUSIVE"}
    assert set(report["mechanism_gates"]) == {
        "M1",
        "M2",
        "M3",
        "M4",
        "M5_exact",
        "M5_fixed_budget",
        "M6",
    }
    assert report["evaluation_protocol"]["zero_shot"] is False
    assert "few-shot adaptation" in (
        run_dir / "gate_report.md"
    ).read_text()
    assert "adaptation_terminal_labels" in pd.read_csv(
        run_dir / "metrics.csv"
    ).columns
    for name in (
        "metrics.csv",
        "bootstrap_intervals.csv",
        "gate_report.json",
        "gate_report.md",
        "projection_vs_gain.png",
        "planner_value_vs_cost.png",
        "finite_budget_crossover.png",
        "latency_frontier.png",
    ):
        assert (run_dir / name).is_file()


def test_smoke_instance_split_is_balanced_within_generator_reward_strata() -> None:
    cfg = ExperimentConfig.from_yaml(Path("configs/smoke.yaml"))
    splits = evaluate_planner._instance_splits(cfg)
    assert list(splits.values()).count("development") == 8
    assert list(splits.values()).count("held_out") == 8
    for regime in ("product", "potts"):
        for reward_name in cfg.rewards:
            ids = [
                instance_id
                for instance_id in range(cfg.num_instances)
                if evaluate_planner._instance_spec(cfg, instance_id)[:2]
                == (reward_name, regime)
            ]
            assert [splits[instance_id] for instance_id in ids] == [
                "development",
                "held_out",
            ]


def test_runner_evaluates_every_candidate_library(tmp_path) -> None:
    cfg = ExperimentConfig(
        d=4,
        q=2,
        methods=("random_balanced",),
        rounds=(2,),
        epsilons=(0.05,),
        rollouts=1,
        num_instances=1,
        couplings=(0.0,),
        topologies=("chain",),
        rewards=("pairwise",),
        candidate_libraries=("oracle", "structural"),
        seeds=(0,),
        response_directions=1,
        bootstrap_replicates=10,
        theory_report_sha256="d0341f361899927fb6616b7ba059b4cc490a793619e9282758e41b88789486aa",
        theory_results_sha256="fae39d94526f283e59971700a39d128848995ca41eb064dd41e6c6863cb1b12c",
    )
    run_dir = tmp_path / "libraries"
    run_experiment(cfg, run_dir)
    trajectories = pd.read_parquet(run_dir / "trajectory_results.parquet")
    assert set(trajectories.candidate_library) == {"oracle", "structural"}
    assert len(trajectories) == 2
    assert trajectories.schedule.nunique() == 1
    assert trajectories.schedule_id.nunique() == 2


def test_runner_uses_projection_aware_saps_and_records_structured_costs(tmp_path) -> None:
    cfg = ExperimentConfig(
        d=4,
        q=2,
        methods=("p_saps_residualized", "p_saps_projection"),
        rounds=(2,),
        epsilons=(0.05,),
        rollouts=1,
        num_instances=1,
        couplings=(0.0,),
        topologies=("chain",),
        rewards=("pairwise",),
        candidate_libraries=("oracle",),
        seeds=(0,),
        response_directions=1,
        bootstrap_replicates=10,
        theory_report_sha256="d0341f361899927fb6616b7ba059b4cc490a793619e9282758e41b88789486aa",
        theory_results_sha256="fae39d94526f283e59971700a39d128848995ca41eb064dd41e6c6863cb1b12c",
    )
    run_dir = tmp_path / "projection-saps-run"
    run_experiment(cfg, run_dir)
    trajectories = pd.read_parquet(run_dir / "trajectory_results.parquet")
    assert set(trajectories.scheduler) == set(cfg.methods)
    assert trajectories.proposal_count.ge(2).all()
    assert trajectories.linear_solve_count.gt(0).all()
    assert trajectories.planning_state_evaluations.gt(0).all()
    assert trajectories.planning_action_evaluations.eq(0).all()
    by_method = trajectories.set_index("scheduler")
    assert by_method.loc[
        "p_saps_projection", "adaptation_terminal_labels"
    ] == (
        cfg.adaptation_trajectories
        * by_method.loc["p_saps_projection", "proposal_count"]
    )
    assert cfg.adaptation_trajectories <= by_method.loc[
        "p_saps_residualized", "adaptation_terminal_labels"
    ] <= by_method.loc[
        "p_saps_projection", "adaptation_terminal_labels"
    ]


def test_runner_emits_one_direct_probe_for_each_observed_frontier_width(tmp_path) -> None:
    cfg = ExperimentConfig(
        d=4,
        q=4,
        methods=("confidence",),
        rounds=(2,),
        epsilons=(0.05,),
        rollouts=1,
        num_instances=1,
        couplings=(0.0,),
        topologies=("chain",),
        rewards=("unary",),
        candidate_libraries=("structural",),
        seeds=(0,),
        response_directions=1,
        bootstrap_replicates=10,
        theory_report_sha256="d0341f361899927fb6616b7ba059b4cc490a793619e9282758e41b88789486aa",
        theory_results_sha256="fae39d94526f283e59971700a39d128848995ca41eb064dd41e6c6863cb1b12c",
    )
    run_dir = tmp_path / "width-probe-run"
    run_experiment(cfg, run_dir)
    scores = pd.read_csv(run_dir / "schedule_scores.csv")
    expected = {
        (row.schedule_id, int(width))
        for row in scores.itertuples()
        for width in json.loads(row.frontier_width_histogram)
    }
    probes = pd.read_parquet(run_dir / "response_probes.parquet")
    actual = set(zip(probes.schedule_id, probes.width, strict=True))
    assert actual == expected
    assert not probes.duplicated(
        ["example_id", "schedule_id", "epsilon", "width"]
    ).any()
    manifest = json.loads((run_dir / "experiment_manifest.json").read_text())
    assert manifest["width_calibration"]["status"] == "inconclusive"
    assert manifest["width_calibration"]["weights"] == {}


def test_held_out_budgeted_saps_uses_development_frozen_width_weights(
    tmp_path,
) -> None:
    cfg = ExperimentConfig(
        d=4,
        q=4,
        methods=("confidence", "b_saps_budgeted"),
        # L=1 exposes the full structural groups (widths 2--4), while L=2
        # supplies width 1; together they identify the complete frozen map.
        rounds=(1, 2),
        epsilons=(0.05,),
        rollouts=1,
        num_instances=4,
        couplings=(0.0,),
        topologies=("chain", "balanced_tree"),
        rewards=("unary",),
        candidate_libraries=("structural",),
        seeds=(0,),
        response_directions=1,
        adaptation_trajectories=10,
        width_calibration_minimum=1,
        bootstrap_replicates=10,
        theory_report_sha256="d0341f361899927fb6616b7ba059b4cc490a793619e9282758e41b88789486aa",
        theory_results_sha256="fae39d94526f283e59971700a39d128848995ca41eb064dd41e6c6863cb1b12c",
    )
    run_dir = tmp_path / "two-stage-width-run"
    run_experiment(cfg, run_dir)
    trajectories = pd.read_parquet(run_dir / "trajectory_results.parquet")
    budgeted = trajectories.loc[trajectories.scheduler == "b_saps_budgeted"]
    assert set(budgeted.data_split) == {"development", "held_out"}
    assert set(
        budgeted.loc[
            budgeted.data_split == "development", "width_weight_source"
        ]
    ) == {"binary_exploratory"}
    assert set(
        budgeted.loc[
            budgeted.data_split == "held_out", "width_weight_source"
        ]
    ) == {"development_isotonic"}
    manifest = json.loads((run_dir / "experiment_manifest.json").read_text())
    assert manifest["width_calibration"]["status"] == "complete"
    assert manifest["width_calibration"]["frozen_before_held_out"] is True


def test_theory_mismatch_leaves_only_failure_record(tmp_path) -> None:
    cfg = ExperimentConfig(
        d=4,
        q=2,
        methods=("random_balanced",),
        rounds=(1,),
        epsilons=(0.05,),
        rollouts=1,
        theory_report_sha256="bad-hash",
    )
    run_dir = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="theory hash mismatch"):
        run_experiment(cfg, run_dir)
    assert sorted(path.name for path in run_dir.iterdir()) == ["failure.json"]
    failure = json.loads((run_dir / "failure.json").read_text())
    assert failure["status"] == "failed"
    assert failure["error_type"] == "RuntimeError"


def test_runner_supplies_shared_oracle_to_subset_planner(tmp_path) -> None:
    cfg = ExperimentConfig(
        d=4,
        q=2,
        methods=("random_balanced", "subset_exact", "subset_beam_8"),
        rounds=(2,),
        epsilons=(0.05,),
        rollouts=1,
        num_instances=1,
        couplings=(0.0,),
        topologies=("chain",),
        rewards=("pairwise",),
        candidate_libraries=("oracle",),
        seeds=(0,),
        response_directions=1,
        bootstrap_replicates=10,
        theory_report_sha256="d0341f361899927fb6616b7ba059b4cc490a793619e9282758e41b88789486aa",
        theory_results_sha256="fae39d94526f283e59971700a39d128848995ca41eb064dd41e6c6863cb1b12c",
    )
    run_dir = tmp_path / "subset-run"
    error = None
    try:
        run_experiment(cfg, run_dir)
    except ValueError as caught:
        error = caught
    assert error is None, f"runner did not supply BatchValueOracle: {error}"
    trajectories = pd.read_parquet(run_dir / "trajectory_results.parquet")
    assert set(trajectories.scheduler) == set(cfg.methods)
    assert set(trajectories.data_split) == {"development"}
    assert trajectories.gram_delta.notna().all()
    assert trajectories.gram_condition.notna().all()
    exact_evaluations = trajectories.loc[
        trajectories.scheduler == "subset_exact", "planning_action_evaluations"
    ]
    assert exact_evaluations.gt(0).all()
    scores = pd.read_csv(run_dir / "schedule_scores.csv")
    assert set(scores.columns) == {
        field.name for field in fields(ScheduleScoreRecord)
    }
    assert set(scores.data_split) == {"development"}
    assert {"gamma_pinv", "gamma_ridge", "gram_delta"} <= set(scores.columns)
    assert scores[["gamma_pinv", "gamma_ridge", "gram_delta"]].notna().all().all()
    assert scores.score_random.notna().all()
    manifest = json.loads((run_dir / "experiment_manifest.json").read_text())
    assert {
        "created_utc",
        "package_version",
        "commit",
        "crossfit_folds",
        "pinv_rtol",
        "ridge_sensitivity",
        "seed_version",
    } <= set(manifest)
    assert manifest["commit"] is None
    assert manifest["crossfit_folds"] == 5
    assert manifest["pinv_rtol"] == 1e-12
    assert manifest["development_ids"] == ["s0-i0"]
    assert manifest["held_out_ids"] == []
    thresholds = manifest["success_thresholds"]["s0-i0"]
    assert set(thresholds) == set(trajectories.schedule_id)
    expected_success = trajectories.apply(
        lambda row: row.terminal_reward
        > thresholds[row.schedule_id],
        axis=1,
    )
    assert trajectories.success.tolist() == expected_success.tolist()
    probes = pd.read_parquet(run_dir / "response_probes.parquet")
    assert set(probes.data_split) == {"development"}
    assert {
        "mean_achieved_kl",
        "positive_rank",
        "kl_saturated",
        "response_space_empty",
    } <= set(probes.columns)
    assert probes.positive_rank.gt(0).all()
    assert np.allclose(probes.mean_achieved_kl, cfg.epsilons[0], atol=1e-5)
    assert manifest["direct_response_status"] == "fisher-whitened-path-kl-v1"


def test_cli_run_preserves_original_ids_in_instance_shard(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "shard.yaml"
    config_path.write_text(
        "\n".join(
            (
                "d: 4",
                "q: 2",
                "methods: [random_balanced]",
                "rounds: [2]",
                "epsilons: [0.05]",
                "rollouts: 1",
                "num_instances: 2",
                "couplings: [0.0]",
                "topologies: [chain]",
                "rewards: [unary]",
                "candidate_libraries: [oracle]",
                "seeds: [0]",
                "response_directions: 1",
                "bootstrap_replicates: 10",
                "theory_report_sha256: d0341f361899927fb6616b7ba059b4cc490a793619e9282758e41b88789486aa",
                "theory_results_sha256: fae39d94526f283e59971700a39d128848995ca41eb064dd41e6c6863cb1b12c",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    error = None
    try:
        main(
            [
                "run",
                "--config",
                str(config_path),
                "--run-id",
                "shard-test",
                "--shard-index",
                "1",
                "--num-shards",
                "2",
            ]
        )
    except SystemExit as caught:
        error = caught
    assert error is None, f"CLI rejected shard arguments: {error}"
    manifest = json.loads(
        (tmp_path / "outputs/shard-test/experiment_manifest.json").read_text()
    )
    assert manifest["instance_ids"] == [1]
    assert sorted(
        manifest["development_ids"] + manifest["held_out_ids"]
    ) == ["s0-i1"]


def test_qary_instance_shards_cannot_bypass_global_calibration_barrier(
    tmp_path,
) -> None:
    cfg = ExperimentConfig(
        d=4,
        q=4,
        methods=("random_balanced",),
        rounds=(1, 2),
        epsilons=(0.05,),
        rollouts=1,
        num_instances=4,
        theory_report_sha256="d0341f361899927fb6616b7ba059b4cc490a793619e9282758e41b88789486aa",
        theory_results_sha256="fae39d94526f283e59971700a39d128848995ca41eb064dd41e6c6863cb1b12c",
    )
    with pytest.raises(ValueError, match="global development calibration"):
        run_experiment(cfg, tmp_path / "invalid-qary-shard", instance_ids=(0, 1))


def test_frozen_development_calibration_allows_qary_heldout_shard(
    tmp_path,
) -> None:
    cfg = ExperimentConfig(
        d=4,
        q=4,
        methods=("b_saps_budgeted",),
        rounds=(2,),
        epsilons=(0.05,),
        rollouts=1,
        num_instances=4,
        candidate_libraries=("structural",),
        adaptation_trajectories=10,
        response_directions=1,
        width_calibration_minimum=1,
        bootstrap_replicates=10,
        theory_report_sha256="d0341f361899927fb6616b7ba059b4cc490a793619e9282758e41b88789486aa",
        theory_results_sha256="fae39d94526f283e59971700a39d128848995ca41eb064dd41e6c6863cb1b12c",
    )
    splits = evaluate_planner._instance_splits(cfg)
    held_id = next(i for i, split in splits.items() if split == "held_out")
    development_ids = [
        f"s{seed}-i{instance_id}"
        for seed in cfg.seeds
        for instance_id, split in sorted(splits.items())
        if split == "development"
    ]
    artifact = tmp_path / "frozen-calibration.json"
    artifact.write_text(
        json.dumps(
            {
                "status": "complete",
                "config": {
                    "d": cfg.d,
                    "q": cfg.q,
                    "instance_design": cfg.instance_design,
                    "epsilons": list(cfg.epsilons),
                    "candidate_libraries": list(cfg.candidate_libraries),
                    "couplings": list(cfg.couplings),
                    "topologies": list(cfg.topologies),
                    "rewards": list(cfg.rewards),
                    "num_instances": cfg.num_instances,
                    "seeds": list(cfg.seeds),
                    "width_calibration_minimum": cfg.width_calibration_minimum,
                },
                "development_ids": development_ids,
                "theory_report_sha256": cfg.theory_report_sha256,
                "theory_results_sha256": cfg.theory_results_sha256,
                "width_calibration": {
                    "status": "complete",
                    "reason": "development-only weighted isotonic fit",
                    "minimum_per_width": 1,
                    "weights": {"0.05": [0.0, 1.0, 0.5, 0.25, 0.125]},
                    "frozen_before_held_out": True,
                },
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "frozen-heldout-shard"
    run_experiment(
        cfg,
        run_dir,
        instance_ids=(held_id,),
        frozen_calibration_path=artifact,
    )
    trajectories = pd.read_parquet(run_dir / "trajectory_results.parquet")
    assert set(trajectories.data_split) == {"held_out"}
    assert set(trajectories.width_weight_source) == {
        "development_isotonic_frozen"
    }
    manifest = json.loads((run_dir / "experiment_manifest.json").read_text())
    assert manifest["frozen_calibration_sha256"] == hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    incompatible = json.loads(artifact.read_text())
    incompatible["config"]["rewards"] = ["pairwise"]
    incompatible_path = tmp_path / "incompatible-calibration.json"
    incompatible_path.write_text(json.dumps(incompatible), encoding="utf-8")
    with pytest.raises(ValueError, match="contract mismatch"):
        evaluate_planner._load_frozen_calibration(incompatible_path, cfg)
    incompatible = json.loads(artifact.read_text())
    incompatible["config"]["instance_design"] = "correlated_potts"
    incompatible_path.write_text(json.dumps(incompatible), encoding="utf-8")
    with pytest.raises(ValueError, match="contract mismatch"):
        evaluate_planner._load_frozen_calibration(incompatible_path, cfg)


def test_cli_forwards_frozen_calibration_artifact(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            (
                "d: 4",
                "q: 4",
                "methods: [b_saps_budgeted]",
                "rounds: [2]",
                "epsilons: [0.05]",
                "rollouts: 1",
                "num_instances: 4",
                "theory_report_sha256: d0341f361899927fb6616b7ba059b4cc490a793619e9282758e41b88789486aa",
                "theory_results_sha256: fae39d94526f283e59971700a39d128848995ca41eb064dd41e6c6863cb1b12c",
            )
        ),
        encoding="utf-8",
    )
    artifact = tmp_path / "calibration.json"
    artifact.write_text("{}", encoding="utf-8")
    captured = {}

    def fake_run(config, run_dir, instance_ids=None, frozen_calibration_path=None):
        captured.update(
            run_dir=run_dir,
            instance_ids=instance_ids,
            frozen_calibration_path=frozen_calibration_path,
        )

    monkeypatch.setattr(evaluate_planner, "run_experiment", fake_run)
    monkeypatch.chdir(tmp_path)
    assert main(
        [
            "run",
            "--config",
            str(config_path),
            "--run-id",
            "frozen-shard",
            "--shard-index",
            "1",
            "--num-shards",
            "4",
            "--frozen-calibration",
            str(artifact),
        ]
    ) == 0
    assert captured["instance_ids"] == (1,)
    assert captured["frozen_calibration_path"] == artifact


def test_merge_shards_requires_complete_disjoint_instance_coverage(
    tmp_path, monkeypatch
) -> None:
    spec = importlib.util.find_spec("feedback_frontier.runners.merge")
    assert spec is not None, "runner must provide strict shard merging"
    from feedback_frontier.runners.merge import merge_shard_runs

    shard_dirs = []
    for instance_id in (0, 1):
        shard = tmp_path / f"shard-{instance_id}"
        shard.mkdir()
        pd.DataFrame(
            {"example_id": [f"s0-i{instance_id}"], "value": [instance_id]}
        ).to_csv(shard / "schedule_scores.csv", index=False)
        pd.DataFrame(
            {"example_id": [f"s0-i{instance_id}"], "value": [instance_id]}
        ).to_parquet(shard / "trajectory_results.parquet", index=False)
        pd.DataFrame(
            {
                "example_id": [f"s0-i{instance_id}"] * 20,
                "data_split": ["development"] * 20,
                "epsilon": [0.05] * 20,
                "width": [1] * 10 + [2] * 10,
                "actual_response_power": [1.0] * 10 + [0.25] * 10,
                "response_space_empty": [False] * 20,
            }
        ).to_parquet(shard / "response_probes.parquet", index=False)
        names = (
            "schedule_scores.csv",
            "trajectory_results.parquet",
            "response_probes.parquet",
        )
        hashes = {
            name: hashlib.sha256((shard / name).read_bytes()).hexdigest()
            for name in names
        }
        (shard / "experiment_manifest.json").write_text(
            json.dumps(
                    {
                        "status": "complete",
                        "config": {
                            "num_instances": 2,
                            "seeds": [0],
                            "q": 4,
                            "epsilons": [0.05],
                        },
                        "instance_ids": [instance_id],
                        "development_ids": [f"s0-i{instance_id}"],
                        "held_out_ids": [],
                        "success_thresholds": {
                            f"s0-i{instance_id}": {"schedule": float(instance_id)}
                        },
                        "pinv_rtol": 1e-12,
                        "crossfit_folds": 5,
                        "seed_version": "blake2b-json-v1",
                        "theory_report_sha256": "report",
                    "theory_results_sha256": "results",
                    "artifact_sha256": hashes,
                }
            ),
            encoding="utf-8",
        )
        shard_dirs.append(shard)

    merged = tmp_path / "merged"
    merge_shard_runs(tuple(shard_dirs), merged)
    assert len(pd.read_parquet(merged / "trajectory_results.parquet")) == 2
    manifest = json.loads((merged / "experiment_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["instance_ids"] == [0, 1]
    assert manifest["development_ids"] == ["s0-i0", "s0-i1"]
    assert manifest["held_out_ids"] == []
    assert set(manifest["success_thresholds"]) == {"s0-i0", "s0-i1"}
    assert manifest["crossfit_folds"] == 5
    assert manifest["width_calibration"]["status"] == "complete"
    assert manifest["width_weights"]["0.05"] == pytest.approx(
        [0.0, 1.0, 0.25]
    )
    monkeypatch.chdir(tmp_path)
    error = None
    try:
        main(
            [
                "merge",
                "--run-id",
                "cli-merged",
                "--shard-dir",
                *(str(path) for path in shard_dirs),
            ]
        )
    except SystemExit as caught:
        error = caught
    assert error is None, f"CLI rejected merge arguments: {error}"
    assert (tmp_path / "outputs/cli-merged/experiment_manifest.json").is_file()
    with pytest.raises(ValueError, match="coverage"):
        merge_shard_runs((shard_dirs[0],), tmp_path / "incomplete")


def test_merge_preserves_one_common_external_frozen_calibration() -> None:
    from feedback_frontier.runners import merge

    calibration = {
        "status": "complete",
        "weights": {"0.05": [0.0, 1.0, 0.5]},
        "frozen_before_held_out": True,
    }
    manifests = [
        {
            "frozen_calibration_sha256": "artifact-hash",
            "width_calibration": calibration,
        },
        {
            "frozen_calibration_sha256": "artifact-hash",
            "width_calibration": calibration,
        },
    ]
    selected, digest = merge._select_external_frozen_calibration(manifests)
    assert selected == calibration
    assert digest == "artifact-hash"
    manifests[1]["frozen_calibration_sha256"] = "different-hash"
    with pytest.raises(ValueError, match="frozen calibration"):
        merge._select_external_frozen_calibration(manifests)
