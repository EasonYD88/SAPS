from __future__ import annotations

import hashlib
import json
import itertools
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from feedback_frontier.analysis.bootstrap import paired_bootstrap
from feedback_frontier.analysis.figures import create_figures
from feedback_frontier.candidates.libraries import CandidateLibrary
from feedback_frontier.estimators.projection import (
    projection_energy,
    residualized_marginal,
)
from feedback_frontier.estimators.response_power import (
    binary_response_power,
    budgeted_score,
    first_order_score,
)
from feedback_frontier.schedulers.base import Schedule
from feedback_frontier.schedulers.structured import (
    all_colorings_with_capacities,
    path_capacity_dp,
)
from feedback_frontier.theory import (
    balanced_width_probability,
    exact_binary_interaction_gain,
    inverse_kl_mean,
    kl_rademacher_mean,
)


THEORY_REPORT_SHA256 = (
    "d0341f361899927fb6616b7ba059b4cc490a793619e9282758e41b88789486aa"
)
THEORY_RESULTS_SHA256 = (
    "fae39d94526f283e59971700a39d128848995ca41eb064dd41e6c6863cb1b12c"
)


def _load_unary_diagnostics(
    run_dir: Path,
) -> tuple[pd.DataFrame | None, dict | None]:
    artifact = run_dir / "unary_null_diagnostics.parquet"
    manifest_path = run_dir / "unary_null_diagnostic_manifest.json"
    if not artifact.exists() and not manifest_path.exists():
        return None, None
    if not artifact.exists() or not manifest_path.exists():
        raise ValueError("unary diagnostic artifact set is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("artifact_sha256", {}).get(artifact.name)
    actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if manifest.get("status") != "complete" or expected != actual:
        raise ValueError("unary diagnostic artifact hash mismatch")
    return pd.read_parquet(artifact), manifest


def make_gate(
    metric: str,
    threshold: str,
    estimate: float | None,
    ci_low: float | None,
    ci_high: float | None,
    passed: bool | None,
    reason: str,
    theory_assumption_status: str = "satisfied",
) -> dict:
    return {
        "metric": metric,
        "threshold": threshold,
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "pass": passed,
        "reason": reason,
        "theory_assumption_status": theory_assumption_status,
    }


def gate_decision(gates: dict[str, dict]) -> str:
    outcomes = [gate.get("pass") for gate in gates.values()]
    if any(outcome is False for outcome in outcomes):
        return "NO_GO"
    if outcomes and all(outcome is True for outcome in outcomes):
        return "GO"
    return "INCONCLUSIVE"


def held_out_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "data_split" not in frame:
        return frame.iloc[0:0].copy()
    return frame.loc[frame["data_split"] == "held_out"].copy()


def h2_coordination_gate(
    trajectories: pd.DataFrame,
    bootstrap_replicates: int,
    minimum_examples: int = 50,
) -> dict:
    held_out = held_out_rows(trajectories)
    required_columns = {
        "example_id",
        "scheduler",
        "terminal_gain",
        "planning_action_evaluations",
    }
    required_methods = {
        "random_balanced",
        "subset_exact",
        "p_saps_residualized",
    }
    if not required_columns <= set(held_out) or not required_methods <= set(
        held_out.get("scheduler", ())
    ):
        return make_gate(
            "random-normalized approximation ratio and action evaluations",
            "ratio >= 0.90 and evaluations <= subset_exact/10",
            None,
            None,
            None,
            None,
            "required held-out methods or planner counters unavailable",
        )
    pair_columns = [
        column
        for column in (
            "example_id",
            "candidate_library",
            "num_rounds",
            "epsilon_target",
        )
        if column in held_out
    ]
    values = held_out.pivot_table(
        index=pair_columns,
        columns="scheduler",
        values="terminal_gain",
        aggfunc="mean",
    )
    costs = held_out.pivot_table(
        index=pair_columns,
        columns="scheduler",
        values="planning_action_evaluations",
        aggfunc="mean",
    )
    values = values.dropna(subset=list(required_methods))
    costs = costs.dropna(subset=["subset_exact", "p_saps_residualized"])
    denominator = values["subset_exact"] - values["random_balanced"]
    valid = denominator.abs() >= 1e-8
    excluded = int((~valid).sum())
    ratios = (
        values.loc[valid, "p_saps_residualized"]
        - values.loc[valid, "random_balanced"]
    ) / denominator.loc[valid]
    if not len(ratios):
        return make_gate(
            "random-normalized approximation ratio and action evaluations",
            "ratio >= 0.90 and evaluations <= subset_exact/10",
            None,
            None,
            None,
            None,
            f"no valid paired denominators; excluded={excluded}",
        )
    example_ratios = ratios.groupby(level="example_id").mean()
    if len(example_ratios) < minimum_examples:
        return make_gate(
            "random-normalized approximation ratio and action evaluations",
            "ratio >= 0.90 and evaluations <= subset_exact/10",
            float(example_ratios.mean()),
            None,
            None,
            None,
            f"fewer than {minimum_examples} held-out examples; excluded={excluded}",
        )
    exact_cost = float(costs["subset_exact"].mean())
    method_cost = float(costs["p_saps_residualized"].mean())
    evaluation_ratio = (
        method_cost / exact_cost if exact_cost > 0 else float("nan")
    )
    rng = np.random.default_rng(20260825)
    samples = rng.choice(
        example_ratios.to_numpy(),
        size=(bootstrap_replicates, len(example_ratios)),
        replace=True,
    ).mean(axis=1)
    estimate = float(example_ratios.mean())
    ci_low, ci_high = (float(value) for value in np.quantile(samples, [0.025, 0.975]))
    passed = bool(
        estimate >= 0.90
        and np.isfinite(evaluation_ratio)
        and evaluation_ratio <= 0.10
    )
    return make_gate(
        "random-normalized approximation ratio and action evaluations",
        "ratio >= 0.90 and evaluations <= subset_exact/10",
        estimate,
        ci_low,
        ci_high,
        passed,
        f"evaluation_ratio={evaluation_ratio:.6g}; excluded_denominators={excluded}",
    )


def _latency_frontier_auc(
    latency: np.ndarray,
    response: np.ndarray,
    grid: np.ndarray,
    horizon: float,
) -> float:
    """Area under the monotone response frontier on a shared time horizon."""
    order = np.argsort(latency, kind="stable")
    x = np.asarray(latency, dtype=float)[order]
    y = np.maximum.accumulate(np.asarray(response, dtype=float)[order])
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0)
    x, y = x[valid], y[valid]
    if not len(x) or not np.isfinite(horizon) or horizon <= 0:
        return float("nan")
    # Repeated measured times represent the best response achieved by then.
    frontier = pd.DataFrame({"x": x, "y": y}).groupby("x", as_index=False)[
        "y"
    ].max()
    x = np.concatenate(([0.0], frontier["x"].to_numpy()))
    y = np.concatenate(([0.0], frontier["y"].to_numpy()))
    values = np.interp(grid, x, y, left=0.0, right=float(y[-1]))
    return float(np.trapezoid(values, grid) / horizon)


def h4_latency_frontier_gate(
    trajectories: pd.DataFrame,
    bootstrap_replicates: int,
    minimum_examples: int = 50,
) -> dict:
    """Compare B-SAPS and random response frontiers using measured wall time."""
    held_out = held_out_rows(trajectories)
    required = {
        "example_id",
        "scheduler",
        "num_rounds",
        "response_power_direct",
        "wall_time_sec",
    }
    methods = {"random_balanced", "b_saps_budgeted"}
    if not required <= set(held_out) or not methods <= set(
        held_out.get("scheduler", ())
    ):
        return make_gate(
            "finite-budget response-vs-latency Pareto AUC",
            "budgeted-minus-random 95% CI lower bound > 0",
            None,
            None,
            None,
            None,
            "required held-out methods, direct response, or wall_time_sec unavailable",
        )
    strata = [
        column
        for column in ("example_id", "candidate_library", "epsilon_target")
        if column in held_out
    ]
    reduced = (
        held_out.loc[held_out["scheduler"].isin(methods)]
        .groupby(strata + ["scheduler", "num_rounds"], as_index=False)
        .agg(
            response_power_direct=("response_power_direct", "mean"),
            wall_time_sec=("wall_time_sec", "mean"),
        )
    )
    differences: list[dict[str, object]] = []
    for stratum, group in reduced.groupby(strata, sort=False):
        by_method = {
            method: group.loc[group["scheduler"] == method]
            for method in methods
        }
        if any(frame.empty for frame in by_method.values()):
            continue
        horizon = float(group["wall_time_sec"].max())
        grid = np.unique(
            np.concatenate(
                ([0.0, horizon], group["wall_time_sec"].to_numpy(dtype=float))
            )
        )
        aucs = {
            method: _latency_frontier_auc(
                frame["wall_time_sec"].to_numpy(),
                frame["response_power_direct"].to_numpy(),
                grid,
                horizon,
            )
            for method, frame in by_method.items()
        }
        example_id = stratum[0] if isinstance(stratum, tuple) else stratum
        difference = aucs["b_saps_budgeted"] - aucs["random_balanced"]
        if np.isfinite(difference):
            differences.append(
                {"example_id": str(example_id), "auc_difference": difference}
            )
    if not differences:
        return make_gate(
            "finite-budget response-vs-latency Pareto AUC",
            "budgeted-minus-random 95% CI lower bound > 0",
            None,
            None,
            None,
            None,
            "no complete method-paired latency frontiers",
        )
    per_example = (
        pd.DataFrame(differences).groupby("example_id")["auc_difference"].mean()
    )
    estimate = float(per_example.mean())
    if len(per_example) < minimum_examples:
        return make_gate(
            "finite-budget response-vs-latency Pareto AUC",
            "budgeted-minus-random 95% CI lower bound > 0",
            estimate,
            None,
            None,
            None,
            f"fewer than {minimum_examples} held-out examples; latency=wall_time_sec",
        )
    rng = np.random.default_rng(20260825)
    samples = rng.choice(
        per_example.to_numpy(),
        size=(bootstrap_replicates, len(per_example)),
        replace=True,
    ).mean(axis=1)
    ci_low, ci_high = (
        float(value) for value in np.quantile(samples, [0.025, 0.975])
    )
    return make_gate(
        "finite-budget response-vs-latency Pareto AUC",
        "budgeted-minus-random 95% CI lower bound > 0",
        estimate,
        ci_low,
        ci_high,
        bool(ci_low > 0),
        "paired bootstrap over example_id; latency=wall_time_sec; normalized common horizon",
    )


def _bootstrap_mean_interval(
    values: np.ndarray, replicates: int, seed: int
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(
        array, size=(replicates, len(array)), replace=True
    ).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(array.mean()), float(low), float(high)


def _theory_regression_gate(manifest: dict) -> dict:
    checks: dict[str, bool] = {}
    try:
        checks["theory_hashes"] = (
            manifest.get("theory_report_sha256") == THEORY_REPORT_SHA256
            and manifest.get("theory_results_sha256") == THEORY_RESULTS_SHA256
        )
        F = np.array([[2.0, 0.3], [0.3, 1.0]])
        b = np.array([0.4, -0.7])
        transform = np.array([[1.0, 0.2], [-0.3, 1.1]])
        base = projection_energy(b, F).value
        reparameterized = projection_energy(
            transform @ b, transform @ F @ transform.T
        ).value
        duplicate = projection_energy(
            np.array([0.73, 0.73]), np.ones((2, 2))
        ).value
        checks["duplicate_reparameterization"] = bool(
            np.isclose(base, reparameterized, rtol=1e-10)
            and np.isclose(duplicate, 0.73**2, rtol=1e-10)
        )
        F3 = np.array(
            [[1.2, 0.2, 0.1], [0.2, 1.1, -0.1], [0.1, -0.1, 0.9]]
        )
        b3 = np.array([0.5, -0.2, 0.7])
        checks["schur_identity"] = bool(
            np.isclose(
                projection_energy(b3, F3).value
                - projection_energy(b3[:1], F3[:1, :1]).value,
                residualized_marginal(b3, F3, (0,), (1, 2)),
                rtol=1e-10,
                atol=1e-12,
            )
        )
        mean = inverse_kl_mean(0.05 / 3)
        checks["binary_finite_kl"] = bool(
            np.isclose(kl_rademacher_mean(mean), 0.05 / 3, rtol=1e-12)
            and np.isclose(
                exact_binary_interaction_gain(3, 0.05),
                0.006035048130208016,
                rtol=1e-12,
            )
        )
        probabilities = [
            balanced_width_probability(9, (3, 3, 3), 5, width)
            for width in range(1, 6)
        ]
        checks["balanced_width"] = bool(np.isclose(sum(probabilities), 1.0))
        edge_weights = (0.4, 1.2, 0.7, 0.5)
        result = path_capacity_dp(edge_weights, (3, 2), lambda_same=0.2)
        brute = max(
            sum(
                weight * (1.0 if colors[i] != colors[i + 1] else 0.2)
                for i, weight in enumerate(edge_weights)
            )
            for colors in all_colorings_with_capacities(5, (3, 2))
        )
        checks["pairwise_dp"] = bool(np.isclose(result.objective, brute))
    except Exception as error:  # pragma: no cover - converted to a gate record
        return make_gate(
            "frozen theory and implementation regressions",
            "all required regressions pass",
            None,
            None,
            None,
            False,
            f"regression raised {type(error).__name__}: {error}",
            "violated",
        )
    passed = all(checks.values())
    return make_gate(
        "frozen theory and implementation regressions",
        "all required regressions pass",
        float(sum(checks.values()) / len(checks)),
        None,
        None,
        passed,
        json.dumps(checks, sort_keys=True),
        "satisfied" if passed else "violated",
    )


def _coordination_construction_gate() -> dict:
    vertices = tuple(range(6))
    itemwise = set(vertices[:3])
    exact = {0, 2, 4}

    def cut(batch: set[int]) -> int:
        return sum(
            (position in batch) != ((position + 1) % 6 in batch)
            for position in vertices
        )

    itemwise_value, exact_value = cut(itemwise), cut(exact)
    passed = exact != itemwise and exact_value > itemwise_value
    return make_gate(
        "regular-cycle exact-subset coordination construction",
        "exact batch differs from itemwise and has strictly larger value",
        float(exact_value - itemwise_value),
        None,
        None,
        passed,
        f"itemwise={sorted(itemwise)} value={itemwise_value}; "
        f"exact={sorted(exact)} value={exact_value}",
    )


def _finite_budget_tie_gate() -> dict:
    library = CandidateLibrary(
        (
            (0, 1, 2, 4),
            (0, 1, 3),
            (0, 5),
            (1, 3, 4, 5),
            (1, 5),
        ),
        "mechanism-control",
        (1.0,) * 5,
    )
    first = Schedule(((0, 1, 2), (3, 4, 5)), 6)
    second = Schedule(((0, 1, 4), (2, 3, 5)), 6)
    epsilon = 0.05
    weights = {
        epsilon: (
            0.0,
            *(binary_response_power(width, epsilon) for width in range(1, 5)),
        )
    }
    first_order = (first_order_score(first, library), first_order_score(second, library))
    finite = (
        budgeted_score(first, library, epsilon, weights),
        budgeted_score(second, library, epsilon, weights),
    )
    passed = np.isclose(*first_order) and not np.isclose(*finite)
    return make_gate(
        "first-order tie separated by finite-budget objective",
        "first-order scores tied and finite-budget scores distinct",
        float(abs(finite[1] - finite[0])),
        None,
        None,
        bool(passed),
        f"first_order={first_order}; finite_budget={finite}",
    )


def _small_epsilon_response_gate(
    trajectories: pd.DataFrame, bootstrap_replicates: int
) -> dict:
    held_out = held_out_rows(trajectories)
    required = {
        "example_id",
        "epsilon_target",
        "response_power_direct",
        "response_positive_rank",
        "gamma_pinv",
    }
    if not required <= set(held_out) or held_out.empty:
        return make_gate(
            "small-epsilon direct/local response ratio",
            "95% CI contains 1",
            None,
            None,
            None,
            None,
            "required held-out Fisher response diagnostics unavailable",
        )
    epsilon = float(held_out["epsilon_target"].min())
    frame = held_out.loc[held_out["epsilon_target"] == epsilon].copy()
    denominator = (
        2
        * frame["epsilon_target"]
        / frame["response_positive_rank"]
        * frame["gamma_pinv"]
    )
    frame["ratio"] = frame["response_power_direct"] / denominator
    frame = frame.loc[
        np.isfinite(frame["ratio"])
        & (denominator > 0)
        & (frame["response_positive_rank"] > 0)
    ]
    ratios = frame.groupby("example_id")["ratio"].mean().to_numpy()
    estimate, low, high = _bootstrap_mean_interval(
        ratios, bootstrap_replicates, 20260826
    )
    if not np.isfinite(estimate):
        passed = None
        reason = "no finite positive-rank ratios"
    else:
        passed = bool(low <= 1.0 <= high)
        reason = f"epsilon={epsilon:.6g}; examples={len(ratios)}"
    return make_gate(
        "small-epsilon direct/local response ratio",
        "95% CI contains 1",
        estimate,
        low if np.isfinite(low) else None,
        high if np.isfinite(high) else None,
        passed,
        reason,
    )


def _unary_null_gate(
    trajectories: pd.DataFrame, bootstrap_replicates: int
) -> dict:
    held_out = held_out_rows(trajectories)
    required = {"example_id", "reward_name", "scheduler", "terminal_gain"}
    if not required <= set(held_out):
        return make_gate(
            "unary scheduling null relative to random",
            "every method's paired 95% CI contains 0",
            None,
            None,
            None,
            None,
            "required held-out unary gain rows unavailable",
        )
    unary = held_out.loc[held_out["reward_name"] == "unary"]
    methods = sorted(set(unary.get("scheduler", ())) - {"random_balanced"})
    intervals = {
        method: paired_bootstrap(
            unary,
            method,
            "random_balanced",
            "terminal_gain",
            bootstrap_replicates,
            20260827,
        )
        for method in methods
    }
    valid = {
        method: interval
        for method, interval in intervals.items()
        if np.isfinite(interval.estimate)
    }
    if not valid:
        return make_gate(
            "unary scheduling null relative to random",
            "every method's paired 95% CI contains 0",
            None,
            None,
            None,
            None,
            "no unary method/random pairs",
        )
    passed = all(
        interval.ci_low <= 0 <= interval.ci_high
        for interval in valid.values()
    ) and len(valid) == len(methods)
    details = {
        method: asdict(interval) for method, interval in valid.items()
    }
    return make_gate(
        "unary scheduling null relative to random",
        "every method's paired 95% CI contains 0",
        float(max(abs(interval.estimate) for interval in valid.values())),
        float(min(interval.ci_low for interval in valid.values())),
        float(max(interval.ci_high for interval in valid.values())),
        passed,
        json.dumps(details, sort_keys=True),
    )


def _exact_unary_diagnostic_gate(
    diagnostics: pd.DataFrame | None,
    diagnostic_manifest: dict | None,
) -> dict:
    required = {"diagnostic_kind", "paired_difference"}
    if (
        diagnostics is None
        or diagnostic_manifest is None
        or diagnostic_manifest.get("status") != "complete"
        or not required <= set(diagnostics)
    ):
        return make_gate(
            "exact product-unary schedule invariance",
            "maximum absolute paired difference <= 1e-12",
            None,
            None,
            None,
            None,
            "frozen exact-unary diagnostic artifact unavailable",
        )
    exact = diagnostics.loc[diagnostics["diagnostic_kind"] == "exact"]
    differences = pd.to_numeric(
        exact["paired_difference"], errors="coerce"
    ).dropna()
    if differences.empty:
        return make_gate(
            "exact product-unary schedule invariance",
            "maximum absolute paired difference <= 1e-12",
            None,
            None,
            None,
            None,
            "exact-unary diagnostic rows unavailable",
        )
    maximum = float(differences.abs().max())
    return make_gate(
        "exact product-unary schedule invariance",
        "maximum absolute paired difference <= 1e-12",
        maximum,
        None,
        None,
        maximum <= 1e-12,
        f"rows={len(differences)}; reward_instances="
        f"{diagnostic_manifest.get('reward_instance_count')}",
    )


def _fixed_budget_unary_diagnostic_gate(
    diagnostics: pd.DataFrame | None,
    diagnostic_manifest: dict | None,
    bootstrap_replicates: int,
) -> dict:
    required = {
        "diagnostic_kind",
        "scheduler",
        "controller_replicate",
        "terminal_replicate",
        "paired_difference",
    }
    minimum_replicates = 8
    if (
        diagnostics is None
        or diagnostic_manifest is None
        or diagnostic_manifest.get("status") != "complete"
        or not required <= set(diagnostics)
    ):
        return make_gate(
            "fixed-budget unary controller robustness",
            "every method's controller-replicate 95% CI contains 0",
            None,
            None,
            None,
            None,
            "frozen fixed-budget unary diagnostic artifact unavailable",
            "finite-budget operational diagnostic",
        )
    if (
        int(diagnostic_manifest.get("controller_replicates", 0))
        < minimum_replicates
        or int(diagnostic_manifest.get("terminal_replicates", 0))
        < minimum_replicates
    ):
        return make_gate(
            "fixed-budget unary controller robustness",
            "every method's controller-replicate 95% CI contains 0",
            None,
            None,
            None,
            None,
            "requires at least 8 controller and 8 terminal replicates",
            "finite-budget operational diagnostic",
        )
    fixed = diagnostics.loc[
        diagnostics["diagnostic_kind"] == "fixed_budget"
    ].copy()
    fixed["paired_difference"] = pd.to_numeric(
        fixed["paired_difference"], errors="coerce"
    )
    fixed = fixed.dropna(subset=["paired_difference"])
    intervals: dict[str, dict[str, float]] = {}
    for method, method_rows in fixed.groupby("scheduler", sort=True):
        controller_means = (
            method_rows.groupby("controller_replicate")["paired_difference"]
            .mean()
            .to_numpy(dtype=float)
        )
        if len(controller_means) < minimum_replicates:
            continue
        estimate, low, high = _bootstrap_mean_interval(
            controller_means,
            bootstrap_replicates,
            20260829,
        )
        intervals[str(method)] = {
            "estimate": estimate,
            "ci_low": low,
            "ci_high": high,
            "controller_replicates": float(len(controller_means)),
        }
    methods = set(fixed.get("scheduler", ()))
    if not intervals or set(intervals) != methods:
        return make_gate(
            "fixed-budget unary controller robustness",
            "every method's controller-replicate 95% CI contains 0",
            None,
            None,
            None,
            None,
            "one or more methods lack eight controller replicates",
            "finite-budget operational diagnostic",
        )
    passed = all(
        interval["ci_low"] <= 0 <= interval["ci_high"]
        for interval in intervals.values()
    )
    details = {
        "reward_instance_count": diagnostic_manifest.get(
            "reward_instance_count"
        ),
        "replication_scope": (
            "conditional algorithmic randomness; not reward-instance replication"
        ),
        "methods": intervals,
    }
    return make_gate(
        "fixed-budget unary controller robustness",
        "every method's controller-replicate 95% CI contains 0",
        float(max(abs(value["estimate"]) for value in intervals.values())),
        float(min(value["ci_low"] for value in intervals.values())),
        float(max(value["ci_high"] for value in intervals.values())),
        passed,
        json.dumps(details, sort_keys=True),
        "finite-budget operational diagnostic",
    )


def _random_baseline_gate(
    schedule_scores: pd.DataFrame, bootstrap_replicates: int
) -> dict:
    required = {
        "example_id",
        "scheduler",
        "score_random",
        "score_budgeted",
    }
    if not required <= set(schedule_scores):
        return make_gate(
            "analytic balanced-random baseline error",
            "Monte Carlo 95% CI contains 0",
            None,
            None,
            None,
            None,
            "required random schedule score rows unavailable",
        )
    random_rows = schedule_scores.loc[
        schedule_scores["scheduler"] == "random_balanced"
    ].copy()
    random_rows["error"] = (
        random_rows["score_budgeted"] - random_rows["score_random"]
    )
    errors = random_rows.groupby("example_id")["error"].mean().to_numpy()
    estimate, low, high = _bootstrap_mean_interval(
        errors, bootstrap_replicates, 20260828
    )
    if not np.isfinite(estimate):
        passed = None
        reason = "no empirical random schedules"
    else:
        passed = bool(low <= 0 <= high)
        reason = f"examples={len(errors)}"
    return make_gate(
        "analytic balanced-random baseline error",
        "Monte Carlo 95% CI contains 0",
        estimate,
        low if np.isfinite(low) else None,
        high if np.isfinite(high) else None,
        passed,
        reason,
    )


def build_mechanism_gates(
    trajectories: pd.DataFrame,
    schedule_scores: pd.DataFrame,
    manifest: dict,
    bootstrap_replicates: int,
    unary_diagnostics: pd.DataFrame | None = None,
    unary_diagnostic_manifest: dict | None = None,
) -> dict[str, dict]:
    return {
        "M1": _theory_regression_gate(manifest),
        "M2": _coordination_construction_gate(),
        "M3": _finite_budget_tie_gate(),
        "M4": _small_epsilon_response_gate(
            trajectories, bootstrap_replicates
        ),
        "M5_exact": _exact_unary_diagnostic_gate(
            unary_diagnostics, unary_diagnostic_manifest
        ),
        "M5_fixed_budget": _fixed_budget_unary_diagnostic_gate(
            unary_diagnostics,
            unary_diagnostic_manifest,
            bootstrap_replicates,
        ),
        "M6": _random_baseline_gate(schedule_scores, bootstrap_replicates),
    }


def _controlled_projection_diagnostics(
    frame: pd.DataFrame, bootstrap_replicates: int
) -> dict[str, float] | None:
    predictors = [
        "gamma_crossfit",
        "score_confidence",
        "score_entropy",
        "score_dependency",
        "num_rounds",
        "epsilon_target",
    ]
    predictors = [column for column in predictors if column in frame]
    required = {"gamma_crossfit", "terminal_gain", "example_id"}
    if not required <= set(frame):
        return None
    working = frame[["example_id", "terminal_gain", *predictors]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if working["example_id"].nunique() < 50:
        return None
    y = working["terminal_gain"].to_numpy(dtype=float)
    y_scale = float(np.std(y, ddof=1))
    if y_scale < 1e-12:
        return None
    active_predictors = [
        column
        for column in predictors
        if float(working[column].std(ddof=1)) >= 1e-12
    ]
    if "gamma_crossfit" not in active_predictors:
        return None
    X = np.column_stack(
        [
            (
                working[column].to_numpy(dtype=float)
                - float(working[column].mean())
            )
            / float(working[column].std(ddof=1))
            for column in active_predictors
        ]
    )
    y = (y - float(y.mean())) / y_scale
    design = np.column_stack([np.ones(len(X)), X])
    projection_index = 1 + active_predictors.index("gamma_crossfit")
    cluster_codes, cluster_names = pd.factorize(
        working["example_id"], sort=True
    )
    cluster_count = len(cluster_names)
    xtx = np.zeros((cluster_count, design.shape[1], design.shape[1]))
    xty = np.zeros((cluster_count, design.shape[1]))
    for cluster_id in range(cluster_count):
        selected = cluster_codes == cluster_id
        cluster_design = design[selected]
        xtx[cluster_id] = cluster_design.T @ cluster_design
        xty[cluster_id] = cluster_design.T @ y[selected]
    ridge = 1e-10 * np.eye(design.shape[1])
    estimate = float(
        np.linalg.solve(xtx.sum(axis=0) + ridge, xty.sum(axis=0))[
            projection_index
        ]
    )
    rng = np.random.default_rng(20260829)
    counts = rng.multinomial(
        cluster_count,
        np.full(cluster_count, 1.0 / cluster_count),
        size=bootstrap_replicates,
    )
    bootstrap_xtx = np.einsum("re,epq->rpq", counts, xtx, optimize=True)
    bootstrap_xty = np.einsum("re,ep->rp", counts, xty, optimize=True)
    coefficients = np.linalg.solve(
        bootstrap_xtx + ridge[None, :, :], bootstrap_xty[..., None]
    )[:, projection_index, 0]
    coefficient_low, coefficient_high = np.quantile(
        coefficients, [0.025, 0.975]
    )

    ranks = working[["terminal_gain", *active_predictors]].rank().to_numpy()
    ranked_y = ranks[:, 0]
    ranked_projection = ranks[
        :, 1 + active_predictors.index("gamma_crossfit")
    ]
    control_indices = [
        1 + index
        for index, column in enumerate(active_predictors)
        if column != "gamma_crossfit"
    ]
    if control_indices:
        controls = np.column_stack(
            [np.ones(len(ranks)), ranks[:, control_indices]]
        )
        ranked_y = ranked_y - controls @ np.linalg.lstsq(
            controls, ranked_y, rcond=None
        )[0]
        ranked_projection = ranked_projection - controls @ np.linalg.lstsq(
            controls, ranked_projection, rcond=None
        )[0]
    partial_spearman = float(np.corrcoef(ranked_projection, ranked_y)[0, 1])
    return {
        "controlled_projection_coefficient": estimate,
        "coefficient_ci_low": float(coefficient_low),
        "coefficient_ci_high": float(coefficient_high),
        "partial_spearman": partial_spearman,
    }


def build_primary_gates(
    trajectories: pd.DataFrame,
    schedule_scores: pd.DataFrame,
    bootstrap_replicates: int,
) -> dict[str, dict]:
    held_out = held_out_rows(trajectories)
    enough = held_out["example_id"].nunique() >= 50
    h1_join_keys = [
        "example_id",
        "candidate_library",
        "scheduler",
        "schedule_id",
        "num_rounds",
        "epsilon_target",
    ]
    h1_score_columns = [
        "score_diag_um",
        "score_confidence",
        "score_entropy",
        "score_dependency",
    ]
    h1_required = {"score_projection", *h1_score_columns, *h1_join_keys}
    if (
        not enough
        or "gamma_crossfit" not in held_out
        or not h1_required <= set(schedule_scores)
    ):
        h1 = make_gate(
            "held-out Spearman(gamma_pinv, gain)",
            "rho >= 0.75 and rho - rho_diag >= 0.10",
            None,
            None,
            None,
            None,
            "insufficient held-out rows or projection score columns",
            "fixed-budget per-instance adaptation; gain evaluation isolated",
        )
    else:
        joined = held_out.merge(
            schedule_scores[[*h1_join_keys, *h1_score_columns]].drop_duplicates(),
            on=h1_join_keys,
            how="left",
            validate="many_to_one",
        )
        rho = float(joined["gamma_crossfit"].corr(joined["terminal_gain"], method="spearman"))
        rho_diag = float(joined["score_diag_um"].corr(joined["terminal_gain"], method="spearman"))
        diagnostics = _controlled_projection_diagnostics(
            joined, bootstrap_replicates
        )
        if diagnostics is None:
            h1 = make_gate(
                "held-out Spearman(gamma_crossfit, gain)",
                "rho >= 0.75 and rho - rho_diag >= 0.10",
                rho,
                None,
                None,
                None,
                "controlled regression/partial-Spearman diagnostics unavailable",
                "fixed-budget per-instance adaptation; gain evaluation isolated",
            )
        else:
            reason = json.dumps(
                {"rho_diag": rho_diag, **diagnostics}, sort_keys=True
            )
            h1 = make_gate(
                "held-out Spearman(gamma_crossfit, gain)",
                "rho >= 0.75 and rho - rho_diag >= 0.10",
                rho,
                None,
                None,
                bool(rho >= 0.75 and rho - rho_diag >= 0.10),
                reason,
            )

    h2 = h2_coordination_gate(trajectories, bootstrap_replicates)

    h3_frame = held_out.loc[
        held_out.get("epsilon_target", pd.Series(index=held_out.index)).isin(
            [0.05, 0.15]
        )
    ]
    h3_methods = {"b_saps_budgeted", "saps_diagonal"}
    if not enough or not h3_methods <= set(h3_frame.get("scheduler", ())):
        h3 = make_gate(
            "paired actual gain: budgeted minus diagonal SAPS",
            "95% CI lower bound > 0",
            None,
            None,
            None,
            None,
            "fewer than 50 held-out examples or required methods missing",
        )
    else:
        interval = paired_bootstrap(
            h3_frame,
            "b_saps_budgeted",
            "saps_diagonal",
            "terminal_gain",
            bootstrap_replicates,
            20260825,
        )
        h3 = make_gate(
            "paired actual gain: budgeted minus diagonal SAPS",
            "95% CI lower bound > 0",
            interval.estimate,
            interval.ci_low,
            interval.ci_high,
            bool(interval.ci_low > 0),
            "paired bootstrap over example_id",
        )

    h4 = h4_latency_frontier_gate(
        trajectories, bootstrap_replicates=bootstrap_replicates
    )
    return {"H1": h1, "H2": h2, "H3": h3, "H4": h4}


def analyze_run(run_dir: Path, bootstrap_replicates: int = 10_000) -> dict:
    frame = pd.read_parquet(run_dir / "trajectory_results.parquet")
    schedule_scores = pd.read_csv(run_dir / "schedule_scores.csv")
    manifest = json.loads((run_dir / "experiment_manifest.json").read_text())
    unary_diagnostics, unary_diagnostic_manifest = _load_unary_diagnostics(
        run_dir
    )
    evaluation_protocol = manifest.get(
        "evaluation_protocol",
        {
            "name": "unspecified",
            "zero_shot": None,
            "claim_guardrail": "evaluation protocol missing from manifest",
        },
    )
    evaluation = held_out_rows(frame)
    metrics = (
        evaluation.groupby("scheduler", as_index=False)
        .agg(
            mean_gain=("terminal_gain", "mean"),
            success_rate=("success", "mean"),
            response_power=("response_power_direct", "mean"),
            planning_time_sec=("planning_time_sec", "mean"),
            n_model_calls=("n_model_calls", "mean"),
            n_reward_calls=("n_reward_calls", "mean"),
            adaptation_terminal_labels=("adaptation_terminal_labels", "mean"),
        )
        .sort_values("scheduler")
    )
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    intervals = []
    for method in sorted(set(evaluation.scheduler) - {"random_balanced"}):
        interval = paired_bootstrap(
            evaluation,
            method,
            "random_balanced",
            "terminal_gain",
            bootstrap_replicates,
            20260825,
        )
        intervals.append({"method": method, **asdict(interval)})
    pd.DataFrame(intervals).to_csv(run_dir / "bootstrap_intervals.csv", index=False)
    gates = build_primary_gates(frame, schedule_scores, bootstrap_replicates)
    decision = gate_decision(gates)
    mechanism_gates = build_mechanism_gates(
        frame,
        schedule_scores,
        manifest,
        bootstrap_replicates,
        unary_diagnostics=unary_diagnostics,
        unary_diagnostic_manifest=unary_diagnostic_manifest,
    )
    screening_decision = gate_decision(mechanism_gates)
    reason = (
        "all primary gates passed"
        if decision == "GO"
        else "at least one primary gate failed"
        if decision == "NO_GO"
        else "one or more primary gates are not estimable"
    )
    report = {
        "decision": decision,
        "reason": reason,
        "gates": gates,
        "mechanism_gates": mechanism_gates,
        "screening_decision": screening_decision,
        "screening_eligible": screening_decision == "GO",
        "evaluation_protocol": evaluation_protocol,
        "unary_diagnostic": unary_diagnostic_manifest,
    }
    (run_dir / "gate_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "gate_report.md").write_text(
        (
            f"# Gate report\n\nDecision: **{decision}**\n\nReason: {reason}.\n\n"
            f"Mechanism smoke decision: **{screening_decision}**.\n\n"
            "Evaluation protocol: held-out reward instance with fixed-budget "
            "few-shot adaptation (not zero-shot).\n"
        ),
        encoding="utf-8",
    )
    create_figures(evaluation, run_dir)
    return report
