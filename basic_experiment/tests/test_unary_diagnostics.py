import numpy as np
import json
from pathlib import Path

import pandas as pd

from feedback_frontier.config import ExperimentConfig
from feedback_frontier.cli import main
from feedback_frontier.generators.categorical_product import CategoricalProduct
from feedback_frontier.rewards.synthetic import RewardTerm, SyntheticReward
from feedback_frontier.rng import SeedBook
from feedback_frontier.runners import evaluate_planner
from feedback_frontier.runners.unary_diagnostics import (
    ExactUnaryActionCache,
    diagnostic_rng_domains,
    run_unary_diagnostics,
)
from feedback_frontier.schedulers.base import Schedule


def _unary_reward(d: int, q: int) -> SyntheticReward:
    return SyntheticReward(
        tuple(
            RewardTerm((position,), np.linspace(-position, position + 1, q))
            for position in range(d)
        ),
        "unary",
    )


def test_exact_product_unary_controller_is_schedule_invariant() -> None:
    model = CategoricalProduct(
        np.array(
            [
                [0.2, -0.4, 0.1],
                [-0.3, 0.7, 0.0],
                [0.5, -0.2, 0.1],
                [-0.1, 0.2, 0.6],
            ]
        )
    )
    reward = _unary_reward(model.d, model.q)
    schedules = (
        Schedule(((0, 1), (2, 3)), model.d),
        Schedule(((0, 2), (1, 3)), model.d),
    )
    terminals = []
    for schedule in schedules:
        terminal, _, model_calls, reward_calls = evaluate_planner._decode(
            model,
            reward,
            schedule,
            epsilon=0.05,
            rollouts=2,
            seedbook=SeedBook(17),
            example_id="exact-unary",
            controlled=True,
            rollout_cache=ExactUnaryActionCache(model, reward),
            rng_domain="unary-diagnostic/exact/terminal-0",
        )
        terminals.append(terminal)
        assert model_calls == 0
        assert reward_calls == 0
    np.testing.assert_array_equal(terminals[0], terminals[1])
    assert reward(terminals[0]) - reward(terminals[1]) == 0.0


def test_unary_diagnostic_rng_domains_are_explicit_and_disjoint() -> None:
    controller, terminal = diagnostic_rng_domains(3, 5)
    assert controller == "unary-diagnostic/controller-3"
    assert terminal == "unary-diagnostic/terminal-5"
    assert controller != terminal
    reserved = {"adaptation", "gain-evaluation", "subset-planning", "dprm-planning"}
    assert controller not in reserved
    assert terminal not in reserved


def test_unary_diagnostic_runner_freezes_source_and_reports_costs(
    tmp_path: Path,
) -> None:
    cfg = ExperimentConfig(
        d=4,
        q=3,
        methods=("random_balanced", "confidence"),
        rounds=(2,),
        epsilons=(0.05,),
        rollouts=2,
        num_instances=4,
        couplings=(0.0,),
        topologies=("chain", "balanced_tree"),
        rewards=("unary",),
        candidate_libraries=("oracle",),
        seeds=(0,),
        adaptation_trajectories=10,
    )
    run_dir = tmp_path / "source-run"
    run_dir.mkdir()
    rows = [
        {
            "example_id": "s0-i3",
            "data_split": "held_out",
            "candidate_library": "oracle",
            "scheduler": "random_balanced",
            "num_rounds": 2,
            "epsilon_target": 0.05,
            "schedule_id": "random-id",
            "schedule": "[[0, 1], [2, 3]]",
        },
        {
            "example_id": "s0-i3",
            "data_split": "held_out",
            "candidate_library": "oracle",
            "scheduler": "confidence",
            "num_rounds": 2,
            "epsilon_target": 0.05,
            "schedule_id": "confidence-id",
            "schedule": "[[0, 2], [1, 3]]",
        },
    ]
    pd.DataFrame(rows).to_csv(run_dir / "schedule_scores.csv", index=False)
    (run_dir / "experiment_manifest.json").write_text(
        json.dumps({"status": "complete", "config": {"instance_design": "mixed"}})
    )

    run_unary_diagnostics(
        cfg,
        run_dir,
        controller_replicates=2,
        terminal_replicates=3,
    )

    diagnostics = pd.read_parquet(run_dir / "unary_null_diagnostics.parquet")
    assert set(diagnostics.diagnostic_kind) == {"exact", "fixed_budget"}
    assert len(diagnostics.loc[diagnostics.diagnostic_kind == "exact"]) == 1
    assert diagnostics.loc[
        diagnostics.diagnostic_kind == "exact", "paired_difference"
    ].eq(0.0).all()
    assert len(diagnostics.loc[diagnostics.diagnostic_kind == "fixed_budget"]) == 6
    manifest = json.loads(
        (run_dir / "unary_null_diagnostic_manifest.json").read_text()
    )
    assert manifest["status"] == "complete"
    assert manifest["reward_instance_count"] == 1
    assert manifest["controller_replicates"] == 2
    assert manifest["terminal_replicates"] == 3
    assert manifest["resources"]["model_calls"] > 0
    assert manifest["resources"]["terminal_label_calls"] > 0
    assert manifest["resources"]["wall_time_sec"] > 0
    assert manifest["source_schedule_scores_sha256"]
    assert manifest["artifact_sha256"]["unary_null_diagnostics.parquet"]


def test_cli_dispatches_unary_diagnostics(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_runner(config, source_run_dir, controller_replicates, terminal_replicates):
        calls.append(
            (
                config,
                source_run_dir,
                controller_replicates,
                terminal_replicates,
            )
        )

    monkeypatch.setattr(
        "feedback_frontier.runners.unary_diagnostics.run_unary_diagnostics",
        fake_runner,
    )
    config_path = Path("configs/smoke.yaml").resolve()
    assert main(
        [
            "diagnose-unary",
            "--config",
            str(config_path),
            "--source-run-dir",
            str(tmp_path),
            "--controller-replicates",
            "7",
            "--terminal-replicates",
            "11",
        ]
    ) == 0
    assert len(calls) == 1
    assert calls[0][1:] == (tmp_path, 7, 11)
