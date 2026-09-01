import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from feedback_frontier.analysis.bootstrap import paired_bootstrap
from feedback_frontier.analysis import aggregate


def test_unary_diagnostic_loader_verifies_frozen_artifact_hash(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "diagnostic_kind": ["exact"],
            "paired_difference": [0.0],
        }
    )
    artifact = tmp_path / "unary_null_diagnostics.parquet"
    frame.to_parquet(artifact, index=False)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest_path = tmp_path / "unary_null_diagnostic_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "artifact_sha256": {
                    "unary_null_diagnostics.parquet": digest
                },
            }
        )
    )
    loaded, manifest = aggregate._load_unary_diagnostics(tmp_path)
    pd.testing.assert_frame_equal(loaded, frame)
    assert manifest["status"] == "complete"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "artifact_sha256": {
                    "unary_null_diagnostics.parquet": "bad"
                },
            }
        )
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        aggregate._load_unary_diagnostics(tmp_path)


def test_paired_bootstrap_is_reproducible() -> None:
    frame = pd.DataFrame(
        {
            "example_id": np.repeat(np.arange(20), 2),
            "scheduler": ["random_balanced", "b_saps_budgeted"] * 20,
            "terminal_gain": np.tile([0.0, 1.0], 20),
        }
    )
    a = paired_bootstrap(
        frame, "b_saps_budgeted", "random_balanced", "terminal_gain", 1000, 7
    )
    b = paired_bootstrap(
        frame, "b_saps_budgeted", "random_balanced", "terminal_gain", 1000, 7
    )
    assert a == b
    assert a.estimate == pytest.approx(1.0)
    assert a.ci_low == pytest.approx(1.0)


def test_gate_records_and_decision_are_machine_readable() -> None:
    make_gate = getattr(aggregate, "make_gate", None)
    decide = getattr(aggregate, "gate_decision", None)
    assert callable(make_gate) and callable(decide)
    passed = make_gate("metric", ">= 1", 1.2, 1.1, 1.3, True, "ok")
    failed = make_gate("metric", ">= 1", 0.8, 0.7, 0.9, False, "below")
    unknown = make_gate(
        "metric", ">= 1", None, None, None, None, "missing data"
    )
    required = {
        "metric",
        "threshold",
        "estimate",
        "ci_low",
        "ci_high",
        "pass",
        "reason",
        "theory_assumption_status",
    }
    assert set(passed) == required
    assert decide({"a": passed}) == "GO"
    assert decide({"a": passed, "b": failed}) == "NO_GO"
    assert decide({"a": passed, "b": unknown}) == "INCONCLUSIVE"


def test_primary_gates_are_inconclusive_when_required_diagnostics_are_missing() -> None:
    builder = getattr(aggregate, "build_primary_gates", None)
    assert callable(builder)
    trajectories = pd.DataFrame(
        {
            "example_id": ["e0", "e0"],
            "data_split": ["held_out", "held_out"],
            "scheduler": ["random_balanced", "b_saps_budgeted"],
            "epsilon_target": [0.05, 0.05],
            "terminal_gain": [0.0, 0.1],
            "gamma_pinv": [0.2, 0.3],
        }
    )
    scores = pd.DataFrame(
        {
            "example_id": ["e0", "e0"],
            "scheduler": ["random_balanced", "b_saps_budgeted"],
            "score_diag_um": [0.1, 0.2],
        }
    )
    gates = builder(trajectories, scores, bootstrap_replicates=100)
    assert set(gates) == {"H1", "H2", "H3", "H4"}
    assert all(gate["pass"] is None for gate in gates.values())
    assert aggregate.gate_decision(gates) == "INCONCLUSIVE"


def test_analysis_selects_only_held_out_rows() -> None:
    selector = getattr(aggregate, "held_out_rows", None)
    assert callable(selector)
    frame = pd.DataFrame(
        {
            "example_id": ["dev", "test"],
            "data_split": ["development", "held_out"],
            "terminal_gain": [100.0, 1.0],
        }
    )
    selected = selector(frame)
    assert selected.example_id.tolist() == ["test"]


def test_h2_gate_uses_random_normalized_value_and_action_cost() -> None:
    rows = []
    for example_id in range(50):
        for scheduler, gain, evaluations in (
            ("random_balanced", 0.0, 0),
            ("subset_exact", 1.0, 100),
            ("p_saps_residualized", 0.95, 5),
        ):
            rows.append(
                {
                    "example_id": f"e{example_id}",
                    "data_split": "held_out",
                    "candidate_library": "oracle",
                    "num_rounds": 2,
                    "epsilon_target": 0.05,
                    "scheduler": scheduler,
                    "terminal_gain": gain,
                    "planning_action_evaluations": evaluations,
                }
            )
    gate = aggregate.h2_coordination_gate(
        pd.DataFrame(rows), bootstrap_replicates=500
    )
    assert gate["estimate"] == pytest.approx(0.95)
    assert gate["ci_low"] == pytest.approx(0.95)
    assert gate["pass"] is True
    assert "evaluation_ratio=0.05" in gate["reason"]


def test_h4_gate_uses_measured_latency_response_frontier_auc() -> None:
    rows = []
    for example_id in range(50):
        for scheduler, responses, latencies in (
            ("random_balanced", (0.10, 0.20), (1.0, 2.0)),
            ("b_saps_budgeted", (0.20, 0.40), (1.0, 2.0)),
        ):
            for rounds, response, latency in zip(
                (1, 2), responses, latencies, strict=True
            ):
                rows.append(
                    {
                        "example_id": f"e{example_id}",
                        "data_split": "held_out",
                        "candidate_library": "structural",
                        "epsilon_target": 0.05,
                        "scheduler": scheduler,
                        "num_rounds": rounds,
                        "response_power_direct": response,
                        "wall_time_sec": latency,
                    }
                )
    gate = aggregate.h4_latency_frontier_gate(
        pd.DataFrame(rows), bootstrap_replicates=500
    )
    # Trapezoidal area on the common latency horizon, including the origin.
    assert gate["estimate"] == pytest.approx(0.10)
    assert gate["ci_low"] == pytest.approx(0.10)
    assert gate["pass"] is True
    assert "wall_time_sec" in gate["reason"]


def test_six_mechanism_gates_are_machine_readable_and_pass_on_controls() -> None:
    trajectories = []
    scores = []
    for example_id in range(20):
        empirical_random = 0.4 if example_id % 2 == 0 else 0.6
        for scheduler in ("random_balanced", "b_saps_budgeted"):
            trajectories.append(
                {
                    "example_id": f"e{example_id}",
                    "data_split": "held_out",
                    "reward_name": "unary",
                    "scheduler": scheduler,
                    "epsilon_target": 0.01,
                    "terminal_gain": float(example_id % 3),
                    "response_power_direct": 0.001,
                    "response_positive_rank": 2,
                    "gamma_pinv": 0.1,
                }
            )
        scores.append(
            {
                "example_id": f"e{example_id}",
                "scheduler": "random_balanced",
                "score_random": 0.5,
                "score_budgeted": empirical_random,
            }
        )
    manifest = {
        "theory_report_sha256": (
            "d0341f361899927fb6616b7ba059b4cc490a793619e9282758e41b88789486aa"
        ),
        "theory_results_sha256": (
            "fae39d94526f283e59971700a39d128848995ca41eb064dd41e6c6863cb1b12c"
        ),
    }
    diagnostics = pd.DataFrame(
        [
            {
                "diagnostic_kind": "exact",
                "example_id": f"e{example_id}",
                "scheduler": "b_saps_budgeted",
                "controller_replicate": -1,
                "terminal_replicate": 0,
                "paired_difference": 0.0,
            }
            for example_id in range(20)
        ]
        + [
            {
                "diagnostic_kind": "fixed_budget",
                "example_id": f"e{example_id}",
                "scheduler": "b_saps_budgeted",
                "controller_replicate": controller,
                "terminal_replicate": terminal,
                "paired_difference": 0.0,
            }
            for example_id in range(20)
            for controller in range(8)
            for terminal in range(8)
        ]
    )
    gates = aggregate.build_mechanism_gates(
        pd.DataFrame(trajectories),
        pd.DataFrame(scores),
        manifest,
        bootstrap_replicates=500,
        unary_diagnostics=diagnostics,
        unary_diagnostic_manifest={
            "status": "complete",
            "reward_instance_count": 20,
            "controller_replicates": 8,
            "terminal_replicates": 8,
        },
    )
    assert set(gates) == {
        "M1",
        "M2",
        "M3",
        "M4",
        "M5_exact",
        "M5_fixed_budget",
        "M6",
    }
    assert all(set(gate) == {
        "metric", "threshold", "estimate", "ci_low", "ci_high", "pass",
        "reason", "theory_assumption_status",
    } for gate in gates.values())
    assert aggregate.gate_decision(gates) == "GO"


def test_unary_mechanism_gates_are_inconclusive_without_frozen_diagnostics() -> None:
    gates = aggregate.build_mechanism_gates(
        pd.DataFrame(),
        pd.DataFrame(),
        {},
        bootstrap_replicates=100,
    )
    assert gates["M5_exact"]["pass"] is None
    assert gates["M5_fixed_budget"]["pass"] is None


def test_h1_score_join_is_example_and_method_scoped() -> None:
    trajectories = pd.DataFrame(
        {
            "example_id": [f"e{i}" for i in range(50)],
            "data_split": "held_out",
            "candidate_library": "oracle",
            "scheduler": "b_saps_budgeted",
            "schedule_id": "shared-schedule",
            "gamma_crossfit": np.arange(50, dtype=float),
            "gamma_pinv": np.arange(50, dtype=float),
            "terminal_gain": np.arange(50, dtype=float),
            "num_rounds": 2,
            "epsilon_target": 0.05,
        }
    )
    scores = pd.DataFrame(
        {
            "example_id": [f"e{i}" for i in range(50)],
            "candidate_library": "oracle",
            "scheduler": "b_saps_budgeted",
            "schedule_id": "shared-schedule",
            "num_rounds": 2,
            "epsilon_target": 0.05,
            "score_projection": np.arange(50, dtype=float),
            "score_diag_um": -np.arange(50, dtype=float),
            "score_confidence": 0.0,
            "score_entropy": 0.0,
            "score_dependency": 0.0,
        }
    )
    gates = aggregate.build_primary_gates(
        trajectories, scores, bootstrap_replicates=100
    )
    diagnostics = json.loads(gates["H1"]["reason"])
    assert diagnostics["rho_diag"] == pytest.approx(-1.0)
    assert diagnostics["controlled_projection_coefficient"] == pytest.approx(1.0)
    assert gates["H1"]["pass"] is True


def test_h1_score_join_is_round_and_epsilon_scoped() -> None:
    trajectories = []
    scores = []
    for example_id in range(50):
        for epsilon, score_diag in ((0.05, -float(example_id)), (0.15, 100.0)):
            common = {
                "example_id": f"e{example_id}",
                "candidate_library": "oracle",
                "scheduler": "b_saps_budgeted",
                "schedule_id": "shared-schedule",
                "num_rounds": 2,
                "epsilon_target": epsilon,
            }
            trajectories.append(
                {
                    **common,
                    "data_split": "held_out",
                    "gamma_crossfit": float(example_id),
                    "gamma_pinv": float(example_id),
                    "terminal_gain": float(example_id),
                }
            )
            scores.append(
                {
                    **common,
                    "score_projection": float(example_id),
                    "score_diag_um": score_diag,
                    "score_confidence": 0.0,
                    "score_entropy": 0.0,
                    "score_dependency": 0.0,
                }
            )
    gates = aggregate.build_primary_gates(
        pd.DataFrame(trajectories), pd.DataFrame(scores), bootstrap_replicates=100
    )
    assert gates["H1"]["pass"] is True
