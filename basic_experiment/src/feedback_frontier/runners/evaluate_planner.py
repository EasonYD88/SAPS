from __future__ import annotations

import hashlib
import itertools
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from feedback_frontier import __version__
from feedback_frontier.candidates.libraries import (
    CandidateLibrary,
    build_oracle_library,
    build_random_matched_library,
    build_structural_library,
)
from feedback_frontier.config import ExperimentConfig
from feedback_frontier.controllers.rollout_local import RolloutCache, calibrate_alpha
from feedback_frontier.estimators.response_power import (
    balanced_random_budgeted_baseline,
    binary_response_power,
    budgeted_score,
    first_order_score,
    width_calibration_summary,
)
from feedback_frontier.estimators.projection import (
    crossfit_geometry,
    fit_moments,
    projection_energy,
    ridge_projection_energy,
    standardize_scores,
)
from feedback_frontier.features.path_scores import PathTrace, path_score
from feedback_frontier.generators.categorical_product import CategoricalProduct
from feedback_frontier.generators.potts_tree import TreePotts
from feedback_frontier.rewards.synthetic import make_reward
from feedback_frontier.rng import SeedBook
from feedback_frontier.runners.calibration_bank import (
    CALIBRATION_RNG_NAMESPACE,
    CalibrationTarget,
    build_schedule_bank,
    summarize_calibration_coverage,
    write_calibration_artifacts,
)
from feedback_frontier.runners.probe_response import (
    FisherResponseEstimate,
    estimate_fisher_response_power,
    restrict_geometry_to_width,
)
from feedback_frontier.schedulers.base import Schedule, balanced_capacities
from feedback_frontier.schedulers.dprm import process_value
from feedback_frontier.schedulers.non_reward import (
    confidence_schedule,
    dependency_matrix,
    dependency_cmi_schedule,
    entropy_schedule,
    min_within_batch_tc_schedule,
    random_balanced_schedule,
)
from feedback_frontier.schedulers.saps import (
    exhaustive_saps,
    rerank_schedule_shortlist,
    residualized_group_weights,
)
from feedback_frontier.schedulers.structured import (
    StructuredResult,
    capacity_preserving_swaps,
)
from feedback_frontier.schedulers.subset import (
    BatchValueOracle,
    beam_subset,
    exact_subset,
)
from feedback_frontier.theory import frontier_width, gram_delta


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _theory_root() -> Path:
    return Path(__file__).resolve().parents[4] / "math_theory"


def _verify_theory(config: ExperimentConfig) -> None:
    root = _theory_root()
    expected = (
        ("correlated_budgeted_feedback_frontier_report.md", config.theory_report_sha256),
        ("correlated_budgeted_feedback_frontier_results.json", config.theory_results_sha256),
    )
    for name, digest in expected:
        if digest and _sha256(root / name) != digest:
            raise RuntimeError(f"theory hash mismatch: {name}")


def _load_frozen_calibration(
    path: Path, config: ExperimentConfig
) -> tuple[dict[str, object], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    is_targeted_bank = payload.get("rng_namespace") == CALIBRATION_RNG_NAMESPACE
    calibration = payload.get("width_calibration", {})
    if payload.get("status") != "complete" or calibration.get("status") != "complete":
        raise ValueError("frozen calibration artifact must be complete")
    source_config = {
        "instance_design": "mixed",
        **payload.get("config", {}),
    }
    contracts = {
        "d": config.d,
        "q": config.q,
        "instance_design": config.instance_design,
        "epsilons": list(config.epsilons),
        "candidate_libraries": list(config.candidate_libraries),
        "couplings": list(config.couplings),
        "topologies": list(config.topologies),
        "rewards": list(config.rewards),
        "num_instances": config.num_instances,
        "seeds": list(config.seeds),
        "width_calibration_minimum": config.width_calibration_minimum,
    }
    if is_targeted_bank:
        contracts.update(
            {
                "calibration_requested_per_cell": (
                    config.calibration_requested_per_cell
                ),
                "calibration_max_per_cell": config.calibration_max_per_cell,
                "calibration_rng_seed": config.calibration_rng_seed,
            }
        )
    mismatched = {
        key: (source_config.get(key), expected)
        for key, expected in contracts.items()
        if source_config.get(key) != expected
    }
    if mismatched:
        raise ValueError(f"frozen calibration contract mismatch: {mismatched}")
    expected_development = sorted(
        f"s{seed}-i{instance_id}"
        for seed in config.seeds
        for instance_id, split in _instance_splits(config).items()
        if split == "development"
    )
    if sorted(payload.get("development_ids", ())) != expected_development:
        raise ValueError("frozen calibration lacks complete development coverage")
    if (
        payload.get("theory_report_sha256") != config.theory_report_sha256
        or payload.get("theory_results_sha256") != config.theory_results_sha256
    ):
        raise ValueError("frozen calibration theory hashes do not match")
    weights = calibration.get("weights", {})
    if set(weights) != {str(value) for value in config.epsilons}:
        raise ValueError("frozen calibration epsilon coverage mismatch")
    if is_targeted_bank:
        expected_flows = {
            "held-out evaluation -> calibration": "forbidden",
            "method results -> calibration": "forbidden",
            "calibration probes -> held-out gain estimates": "forbidden",
        }
        if payload.get("forbidden_information_flows") != expected_flows:
            raise ValueError("frozen calibration information-flow contract mismatch")
        digest_path = path.with_suffix(".sha256")
        if (
            not digest_path.is_file()
            or digest_path.read_text(encoding="utf-8").strip() != _sha256(path)
        ):
            raise ValueError("frozen calibration manifest hash mismatch")
        for name, expected_digest in payload.get("artifact_sha256", {}).items():
            artifact = path.parent / name
            if not artifact.is_file() or _sha256(artifact) != expected_digest:
                raise ValueError(f"frozen calibration artifact hash mismatch: {name}")
        coverage_path = path.parent / "calibration_coverage.csv"
        bank_path = path.parent / "calibration_schedule_bank.jsonl"
        if not coverage_path.is_file() or not bank_path.is_file():
            raise ValueError("frozen calibration bank artifacts are missing")
        coverage = pd.read_csv(coverage_path)
        if (
            coverage.empty
            or not coverage["passed"].astype(bool).all()
            or not (
                coverage["valid_count"]
                >= config.width_calibration_minimum
            ).all()
        ):
            raise ValueError("frozen calibration coverage is incomplete")
        bank_records = [
            json.loads(line)
            for line in bank_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any(
            record.get("data_split") != "development"
            or record.get("example_id") not in expected_development
            for record in bank_records
        ):
            raise ValueError("frozen calibration bank is not development-only")
    return dict(calibration), _sha256(path)


def _schedule_from_order(order: list[int], capacities: tuple[int, ...]) -> Schedule:
    batches = []
    start = 0
    for capacity in capacities:
        batches.append(tuple(sorted(order[start : start + capacity])))
        start += capacity
    return Schedule(tuple(batches), len(order))


def _schedule_control_scores(
    model, schedule: Schedule, pairwise_dependency: np.ndarray
) -> tuple[float, float, float]:
    """Reward-blind schedule alignment controls used by the H1 regression."""
    probabilities = model.conditional_marginals({})
    confidence = np.max(probabilities, axis=1)
    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-300, 1.0)), axis=1
    )
    round_ids = np.asarray(schedule.round_of_position, dtype=float)
    confidence_alignment = -float(np.mean(round_ids * confidence))
    entropy_alignment = float(np.mean(round_ids * entropy))
    within_batch_dependency = float(
        sum(
            pairwise_dependency[left, right]
            for batch in schedule.batches
            for left, right in itertools.combinations(batch, 2)
        )
    )
    return confidence_alignment, entropy_alignment, -within_batch_dependency


def _itemwise_schedule(
    d: int, capacities: tuple[int, ...], groups: tuple[tuple[int, ...], ...]
) -> Schedule:
    score = np.zeros(d)
    for group in groups:
        score[list(group)] += 1 / len(group)
    order = sorted(range(d), key=lambda i: (-score[i], i))
    return _schedule_from_order(order, capacities)


def _dprm_schedule(
    model,
    reward,
    rounds: int,
    rollouts: int,
    beta: float,
    seedbook: SeedBook,
    example_id: str,
    method: str = "dprm_myopic",
    epsilon: float = 0.0,
) -> tuple[Schedule, int]:
    if method not in {"dprm_myopic", "dprm_rollout"}:
        raise ValueError(f"unknown DPRM method: {method}")
    confidence_order = tuple(
        position
        for batch in confidence_schedule(model, rounds).batches
        for position in batch
    )
    controller_cache = RolloutCache(
        model, reward, seedbook, example_id, rng_domain="dprm-planning"
    )

    def continuation_reward(
        initial_position: int,
        initial_token: int,
        rollout_id: int,
    ) -> float:
        observed = {initial_position: initial_token}
        remaining = tuple(
            position
            for position in confidence_order
            if position != initial_position
        )
        if not remaining:
            return float(reward(np.asarray([initial_token], dtype=int)))
        future_rounds = min(len(remaining), max(1, rounds - 1))
        capacities = balanced_capacities(len(remaining), future_rounds)
        start = 0
        for round_id, capacity in enumerate(capacities):
            batch = remaining[start : start + capacity]
            start += capacity
            base_probabilities = model.conditional_marginals(observed)
            selected: dict[int, int] = {}
            for position in batch:
                probabilities = base_probabilities[position]
                if method == "dprm_rollout":
                    action_values = controller_cache.action_values(
                        observed, position, rollouts
                    )
                    probabilities = calibrate_alpha(
                        probabilities, action_values, epsilon
                    ).probabilities
                uniform = seedbook.rng(
                    "rollout",
                    example_id,
                    "dprm-continuation",
                    rollout_id,
                    round_id,
                    position,
                ).random()
                selected[position] = int(
                    np.searchsorted(np.cumsum(probabilities), uniform)
                )
            observed.update(selected)
        terminal = np.asarray([observed[i] for i in range(model.d)], dtype=int)
        return float(reward(terminal))

    scores = []
    for position in range(model.d):
        token_values = []
        for token in range(model.q):
            rewards = np.asarray(
                [
                    continuation_reward(position, token, rollout_id)
                    for rollout_id in range(rollouts)
                ]
            )
            token_values.append(process_value(rewards, beta))
        token_values_array = np.asarray(token_values)
        scores.append(
            process_value(token_values_array, beta)
            - float(token_values_array.mean())
        )
    scores = np.asarray(scores)
    order = sorted(range(model.d), key=lambda i: (-scores[i], i))
    return _schedule_from_order(order, balanced_capacities(model.d, rounds)), model.d


def _subset_schedule(
    method: str,
    d: int,
    capacities: tuple[int, ...],
    confidence_order: tuple[int, ...],
    oracle: BatchValueOracle,
) -> Schedule:
    unresolved = tuple(range(d))
    capacity = capacities[0]
    if method == "subset_exact":
        result = exact_subset(unresolved, capacity, oracle, d)
    elif method.startswith("subset_beam_"):
        beam_width = int(method.rsplit("_", 1)[1])
        result = beam_subset(unresolved, capacity, oracle, beam_width)
    else:
        raise ValueError(f"unknown subset method: {method}")
    return _complete_schedule(result.batch, d, capacities, confidence_order)


def _complete_schedule(
    first_batch: tuple[int, ...],
    d: int,
    capacities: tuple[int, ...],
    confidence_order: tuple[int, ...],
) -> Schedule:
    remaining = tuple(i for i in confidence_order if i not in first_batch)
    batches = [tuple(sorted(first_batch))]
    start = 0
    for following_capacity in capacities[1:]:
        batches.append(tuple(sorted(remaining[start : start + following_capacity])))
        start += following_capacity
    return Schedule(tuple(batches), d)


def _make_batch_value_oracle(
    model,
    reward,
    rounds: int,
    epsilon: float,
    rollouts: int,
    seedbook: SeedBook,
    example_id: str,
) -> BatchValueOracle:
    capacities = balanced_capacities(model.d, rounds)
    confidence_order = tuple(
        position
        for batch in confidence_schedule(model, rounds).batches
        for position in batch
    )
    rollout_cache = RolloutCache(
        model, reward, seedbook, example_id, rng_domain="subset-planning"
    )

    def value(first_batch: tuple[int, ...]) -> float:
        schedule = _complete_schedule(
            first_batch, model.d, capacities, confidence_order
        )
        terminal, _, _, _ = _decode(
            model,
            reward,
            schedule,
            epsilon,
            rollouts,
            seedbook,
            example_id,
            True,
            rollout_cache=rollout_cache,
            rng_domain="subset-planning",
        )
        return float(reward(terminal))

    return BatchValueOracle(value)


def _saps_result(
    d: int,
    capacities: tuple[int, ...],
    library,
    lambdas: tuple[float, ...],
    seedbook: SeedBook,
    example_id: str,
) -> StructuredResult:
    if d <= 8:
        return exhaustive_saps(d, capacities, library, lambdas)
    initial = random_balanced_schedule(d, len(capacities), seedbook, example_id)
    return capacity_preserving_swaps(initial, library, lambdas)


def _saps_schedule(
    d: int,
    capacities: tuple[int, ...],
    library,
    lambdas: tuple[float, ...],
    seedbook: SeedBook,
    example_id: str,
) -> Schedule:
    return _saps_result(
        d, capacities, library, lambdas, seedbook, example_id
    ).schedule


def _choose_schedule(
    method: str,
    model,
    rounds: int,
    library,
    epsilon: float,
    seedbook: SeedBook,
    example_id: str,
    batch_oracle: BatchValueOracle | None = None,
    reward=None,
    rollouts: int | None = None,
    dprm_beta: float = 1.0,
) -> Schedule:
    capacities = balanced_capacities(model.d, rounds)
    if method == "random_balanced":
        return random_balanced_schedule(model.d, rounds, seedbook, example_id)
    if method == "confidence":
        return confidence_schedule(model, rounds)
    if method == "entropy":
        return entropy_schedule(model, rounds)
    if method == "dependency_cmi":
        return dependency_cmi_schedule(model, rounds)
    if method == "min_within_batch_tc":
        return min_within_batch_tc_schedule(model, rounds)
    if method in {"dprm_myopic", "dprm_rollout"}:
        if reward is None or rollouts is None:
            raise ValueError(f"{method} requires reward and rollout count")
        return _dprm_schedule(
            model,
            reward,
            rounds,
            rollouts,
            dprm_beta,
            seedbook,
            example_id,
            method=method,
            epsilon=epsilon,
        )[0]
    if method.startswith("subset_"):
        if batch_oracle is None:
            raise ValueError(f"{method} requires a shared batch value oracle")
        confidence_order = tuple(
            position
            for batch in confidence_schedule(model, rounds).batches
            for position in batch
        )
        return _subset_schedule(
            method,
            model.d,
            capacities,
            confidence_order,
            batch_oracle,
        )
    max_width = max(map(len, library.groups), default=1)
    if method == "saps_diagonal":
        lambdas = (0.0, 1.0, *((0.0,) * (max_width - 1)))
    else:
        lambdas = (0.0,) + tuple(
            binary_response_power(width, epsilon)
            for width in range(1, max_width + 1)
        )
    return _saps_schedule(
        model.d, capacities, library, lambdas, seedbook, example_id
    )


def _decode(
    model,
    reward,
    schedule: Schedule,
    epsilon: float,
    rollouts: int,
    seedbook: SeedBook,
    example_id: str,
    controlled: bool,
    rollout_cache: RolloutCache | None = None,
    rng_domain: str = "gain-evaluation",
) -> tuple[np.ndarray, float, int, int]:
    observed: dict[int, int] = {}
    kls = []
    cache = rollout_cache or RolloutCache(
        model, reward, seedbook, example_id, rng_domain=rng_domain
    )
    for batch in schedule.batches:
        probabilities = model.conditional_marginals(observed)
        selected: dict[int, int] = {}
        for position in batch:
            base = probabilities[position]
            if controlled:
                values = cache.action_values(observed, position, rollouts)
                calibration = calibrate_alpha(base, values, epsilon)
                probs = calibration.probabilities
                kls.append(calibration.achieved_kl)
            else:
                probs = base
                kls.append(0.0)
            uniform = seedbook.rng(
                "base", example_id, rng_domain, epsilon, position
            ).random()
            selected[position] = int(np.searchsorted(np.cumsum(probs), uniform))
        observed.update(selected)
    terminal = np.array([observed[i] for i in range(model.d)], dtype=int)
    return (
        terminal,
        float(np.mean(kls)),
        cache.n_model_calls,
        cache.n_reward_calls,
    )


@dataclass(frozen=True)
class GeometryEstimate:
    gamma_pinv: float
    gamma_ridge: float
    gram_delta: float
    gram_condition: float
    fold_gammas: tuple[float, ...]
    reward_q75: float
    active_metadata: tuple[tuple[tuple[int, ...], int, int], ...]
    b: tuple[float, ...]
    scales: tuple[float, ...]
    gram: tuple[tuple[float, ...], ...]


def _sample_base_trace(
    model,
    schedule: Schedule,
    seedbook: SeedBook,
    example_id: str,
    sample_id: int,
) -> PathTrace:
    observed: dict[int, int] = {}
    tokens = np.full(model.d, -1, dtype=int)
    probabilities_by_round: list[np.ndarray] = []
    for round_id, batch in enumerate(schedule.batches):
        probabilities = model.conditional_marginals(observed)
        probabilities_by_round.append(probabilities.copy())
        selected: dict[int, int] = {}
        for position in batch:
            uniform = seedbook.rng(
                "base", example_id, "adaptation", sample_id, round_id, position
            ).random()
            token = int(
                np.searchsorted(np.cumsum(probabilities[position]), uniform)
            )
            selected[position] = token
            tokens[position] = token
        observed.update(selected)
    return PathTrace(tokens, tuple(probabilities_by_round))


def _estimate_geometry(
    model,
    reward,
    schedule: Schedule,
    library,
    seedbook: SeedBook,
    example_id: str,
    n_samples: int,
    ridge_multiplier: float,
) -> GeometryEstimate:
    if n_samples < 10:
        raise ValueError("geometry requires at least 10 base trajectories")
    score_rows = []
    rewards = []
    metadata = None
    for sample_id in range(n_samples):
        trace = _sample_base_trace(
            model, schedule, seedbook, example_id, sample_id
        )
        scored = path_score(schedule, trace, library)
        if metadata is None:
            metadata = scored.metadata
        elif scored.metadata != metadata:
            raise RuntimeError("path-score metadata changed across base trajectories")
        score_rows.append(scored.values)
        rewards.append(float(reward(trace.tokens)))
    scores = np.vstack(score_rows)
    reward_values = np.asarray(rewards)
    order = seedbook.rng("bootstrap", "geometry-folds", example_id).permutation(
        n_samples
    )
    fold_ids = np.empty(n_samples, dtype=int)
    fold_ids[order] = np.arange(n_samples) % 5
    crossfit = crossfit_geometry(scores, reward_values, fold_ids)
    standardized = standardize_scores(scores, np.arange(n_samples))
    moments = fit_moments(standardized.values, reward_values)
    pinv = projection_energy(moments.b, moments.F)
    ridge = ridge_projection_energy(
        moments.b, moments.F, ridge_multiplier
    )
    return GeometryEstimate(
        gamma_pinv=pinv.value,
        gamma_ridge=ridge.value,
        gram_delta=gram_delta(moments.F),
        gram_condition=float(np.linalg.cond(moments.F)),
        fold_gammas=tuple(fold.gamma_pinv for fold in crossfit.folds),
        reward_q75=float(np.quantile(reward_values, 0.75)),
        active_metadata=tuple(
            metadata[index] for index in standardized.active_columns
        ),
        b=tuple(float(value) for value in moments.b),
        scales=tuple(float(value) for value in standardized.scale),
        gram=tuple(tuple(float(value) for value in row) for row in moments.F),
    )


@dataclass(frozen=True)
class ProjectionSapsPlan:
    residualized: Schedule
    projection: Schedule
    proposal_count: int
    linear_solve_count: int
    state_evaluations: int
    residualized_geometry_count: int
    projection_geometry_count: int


def _projection_saps_schedules(
    model,
    reward,
    rounds: int,
    library: CandidateLibrary,
    epsilon: float,
    seedbook: SeedBook,
    example_id: str,
    n_samples: int,
    ridge_multiplier: float,
    geometry_cache: dict[tuple[tuple[int, ...], ...], GeometryEstimate],
    shortlist_limit: int = 128,
    frozen_budgeted_lambdas: tuple[float, ...] | None = None,
) -> ProjectionSapsPlan:
    if shortlist_limit < 1:
        raise ValueError("shortlist_limit must be positive")
    capacities = balanced_capacities(model.d, rounds)
    max_width = max(map(len, library.groups), default=1)
    diagonal_lambdas = (0.0, 1.0, *((0.0,) * (max_width - 1)))
    budgeted_lambdas = frozen_budgeted_lambdas or (
        (0.0,)
        + tuple(
            binary_response_power(width, epsilon)
            for width in range(1, max_width + 1)
        )
    )
    if len(budgeted_lambdas) <= max_width:
        raise ValueError(
            f"width weights stop at {len(budgeted_lambdas) - 1}, "
            f"but candidate width reaches {max_width}"
        )

    def geometry(schedule: Schedule) -> GeometryEstimate:
        if schedule.batches not in geometry_cache:
            geometry_cache[schedule.batches] = _estimate_geometry(
                model,
                reward,
                schedule,
                library,
                seedbook,
                example_id,
                n_samples,
                ridge_multiplier,
            )
        return geometry_cache[schedule.batches]

    diagonal = _saps_result(
        model.d,
        capacities,
        library,
        diagonal_lambdas,
        seedbook,
        f"{example_id}:p-saps-diagonal",
    )
    reference_geometry = geometry(diagonal.schedule)
    residual_weights, residual_solves = residualized_group_weights(
        diagonal.schedule,
        library,
        np.asarray(reference_geometry.b),
        np.asarray(reference_geometry.gram),
        reference_geometry.active_metadata,
    )
    residual_library = CandidateLibrary(
        library.groups, library.source, residual_weights
    )
    residualized = _saps_result(
        model.d,
        capacities,
        residual_library,
        diagonal_lambdas,
        seedbook,
        f"{example_id}:p-saps-residualized",
    )
    budgeted = _saps_result(
        model.d,
        capacities,
        library,
        budgeted_lambdas,
        seedbook,
        f"{example_id}:p-saps-budgeted",
    )
    candidates = [
        diagonal.schedule,
        residualized.schedule,
        budgeted.schedule,
        confidence_schedule(model, rounds),
        entropy_schedule(model, rounds),
        dependency_cmi_schedule(model, rounds),
        min_within_batch_tc_schedule(model, rounds),
    ]
    state_evaluations = (
        diagonal.state_evaluations
        + residualized.state_evaluations
        + budgeted.state_evaluations
    )
    feasible_schedule_count = math.factorial(model.d)
    for capacity in capacities:
        feasible_schedule_count //= math.factorial(capacity)
    target_count = min(shortlist_limit, feasible_schedule_count)
    proposal_id = 0
    while len({schedule.batches for schedule in candidates}) < target_count:
        initial = random_balanced_schedule(
            model.d,
            rounds,
            seedbook,
            f"{example_id}:p-saps-proposal:{proposal_id}",
        )
        proposal_id += 1
        candidates.append(initial)
        improved = capacity_preserving_swaps(
            initial, residual_library, diagonal_lambdas
        )
        candidates.append(improved.schedule)
        state_evaluations += improved.state_evaluations
        if proposal_id >= target_count * 4:
            break
    unique_candidates = tuple(
        {schedule.batches: schedule for schedule in candidates}.values()
    )[:target_count]
    reranked = rerank_schedule_shortlist(
        unique_candidates,
        lambda schedule: geometry(schedule).gamma_pinv,
        lambda schedule: float(len(schedule.batches)),
    )
    return ProjectionSapsPlan(
        residualized=residualized.schedule,
        projection=reranked.schedule,
        proposal_count=reranked.proposal_count,
        linear_solve_count=residual_solves + reranked.linear_solve_count,
        state_evaluations=state_evaluations,
        residualized_geometry_count=len(
            {diagonal.schedule.batches, residualized.schedule.batches}
        ),
        projection_geometry_count=reranked.proposal_count,
    )


def _instance_spec(
    config: ExperimentConfig, instance_id: int
) -> tuple[str, str, str | None, float | None]:
    reward_count = len(config.rewards)
    reward_name = config.rewards[instance_id % reward_count]
    stratum_rank = instance_id // reward_count
    if config.instance_design == "correlated_potts":
        cell_count = len(config.topologies) * len(config.couplings)
        cell_rank = stratum_rank % cell_count
        topology = config.topologies[cell_rank % len(config.topologies)]
        coupling = config.couplings[cell_rank // len(config.topologies)]
        return reward_name, "potts", topology, coupling
    regime = stratum_rank % 3
    if regime == 0:
        return reward_name, "product", None, None
    preferred_topology = "chain" if regime == 1 else "balanced_tree"
    topology = (
        preferred_topology
        if preferred_topology in config.topologies
        else config.topologies[(regime - 1) % len(config.topologies)]
    )
    coupling = config.couplings[(stratum_rank // 3) % len(config.couplings)]
    return reward_name, "potts", topology, coupling


def _instance_splits(config: ExperimentConfig) -> dict[int, str]:
    strata: dict[tuple[object, ...], list[int]] = {}
    for instance_id in range(config.num_instances):
        reward_name, regime, topology, coupling = _instance_spec(
            config, instance_id
        )
        key = (
            (regime, reward_name, topology, coupling)
            if config.instance_design == "correlated_potts"
            else (regime, reward_name)
        )
        strata.setdefault(key, []).append(instance_id)
    assignments: dict[int, str] = {}
    for ids in strata.values():
        for rank, instance_id in enumerate(sorted(ids)):
            assignments[instance_id] = (
                "development" if rank % 2 == 0 else "held_out"
            )
    return assignments


def _instance_coverage(
    config: ExperimentConfig, instance_ids: tuple[int, ...]
) -> list[dict[str, object]]:
    splits = _instance_splits(config)
    counts: dict[tuple[object, ...], int] = {}
    for instance_id in instance_ids:
        reward_name, regime, topology, coupling = _instance_spec(
            config, instance_id
        )
        key = (
            splits[instance_id],
            reward_name,
            regime,
            topology,
            coupling,
        )
        counts[key] = counts.get(key, 0) + len(config.seeds)
    return [
        {
            "data_split": key[0],
            "reward_name": key[1],
            "generator_regime": key[2],
            "topology": key[3],
            "coupling": key[4],
            "count": count,
        }
        for key, count in sorted(
            counts.items(), key=lambda item: tuple(map(str, item[0]))
        )
    ]


def _shard_instance_ids(
    num_instances: int, shard_index: int, num_shards: int
) -> tuple[int, ...]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(
            f"invalid shard index/count: index={shard_index}, count={num_shards}"
        )
    return tuple(range(shard_index, num_instances, num_shards))


def _instance(config: ExperimentConfig, seed: int, instance_id: int):
    book = SeedBook(seed)
    rng = (
        book.rng("instance", instance_id)
        if config.instance_design == "mixed"
        else book.rng("instance", config.instance_design, instance_id)
    )
    reward_name, regime, topology, coupling = _instance_spec(config, instance_id)
    edges = tuple((i, i + 1) for i in range(config.d - 1))
    if regime == "product":
        model = CategoricalProduct.random(config.d, config.q, rng)
        generator_name = "product"
    else:
        assert topology is not None and coupling is not None
        model = TreePotts.random(config.d, config.q, topology, coupling, rng)
        generator_name = f"potts_{topology}_{coupling:g}"
        edges = model.edges
    reward = make_reward(reward_name, config.d, config.q, edges, rng)
    return book, model, reward, generator_name, edges


def _run_experiment(
    config: ExperimentConfig,
    run_dir: Path,
    instance_ids: tuple[int, ...],
    supplied_width_calibration: dict[str, object] | None = None,
    frozen_calibration_sha256: str | None = None,
) -> None:
    staging = run_dir / ".staging"
    staging.mkdir()
    trajectories: list[dict] = []
    scores: list[dict] = []
    probes: list[dict] = []
    success_thresholds: dict[str, dict[str, float]] = {}
    started = time.perf_counter()
    instance_splits = _instance_splits(config)
    experiment_cells = sorted(
        itertools.product(
            config.seeds,
            instance_ids,
            config.candidate_libraries,
        ),
        key=lambda cell: (
            0 if instance_splits[cell[1]] == "development" else 1,
            cell,
        ),
    )
    external_frozen_calibration = supplied_width_calibration is not None
    frozen_width_calibration = supplied_width_calibration
    frozen_width_weights: dict[float, tuple[float, ...]] = (
        {
            float(epsilon): tuple(float(value) for value in values)
            for epsilon, values in supplied_width_calibration["weights"].items()
        }
        if supplied_width_calibration is not None
        else {}
    )
    entered_held_out = external_frozen_calibration
    for seed, instance_id, library_name in experiment_cells:
            data_split = instance_splits[instance_id]
            if data_split == "held_out" and not entered_held_out:
                frozen_width_calibration = width_calibration_summary(
                    pd.DataFrame(probes),
                    config.q,
                    config.epsilons,
                    minimum_per_width=config.width_calibration_minimum,
                )
                if frozen_width_calibration["status"] not in {
                    "complete",
                    "analytic_binary",
                }:
                    raise RuntimeError(
                        "calibration_inconclusive before held-out evaluation: "
                        f"{frozen_width_calibration['reason']}"
                    )
                frozen_width_weights = {
                    float(epsilon): tuple(float(value) for value in values)
                    for epsilon, values in frozen_width_calibration[
                        "weights"
                    ].items()
                }
                entered_held_out = True
            book, model, reward, generator_name, edges = _instance(
                config, seed, instance_id
            )
            example_id = f"s{seed}-i{instance_id}"
            oracle = build_oracle_library(reward)
            if library_name == "structural":
                library = build_structural_library(config.d, edges)
            elif library_name == "random_matched":
                structural = build_structural_library(config.d, edges)
                library = build_random_matched_library(
                    structural, book.rng("proposal", example_id)
                )
            else:
                library = oracle
            schedule_cache: dict[tuple[str, int, float], Schedule] = {}
            batch_oracle_cache: dict[tuple[int, float], BatchValueOracle] = {}
            decode_cache: dict[
                tuple[tuple[tuple[int, ...], ...], float, bool],
                tuple[np.ndarray, float, int, int],
            ] = {}
            geometry_cache: dict[
                tuple[tuple[int, ...], ...], GeometryEstimate
            ] = {}
            geometry_time_cache: dict[tuple[tuple[int, ...], ...], float] = {}
            decode_time_cache: dict[
                tuple[tuple[tuple[int, ...], ...], float, bool], float
            ] = {}
            response_cache: dict[
                tuple[tuple[tuple[int, ...], ...], float],
                FisherResponseEstimate,
            ] = {}
            width_response_cache: dict[
                tuple[tuple[tuple[int, ...], ...], float, int],
                FisherResponseEstimate,
            ] = {}
            schedule_control_cache: dict[
                tuple[tuple[int, ...], ...], tuple[float, float, float]
            ] = {}
            pairwise_dependency = dependency_matrix(model)
            emitted_probe_keys: set[tuple[str, float, int]] = set()
            projection_plan_cache: dict[
                tuple[int, float], tuple[ProjectionSapsPlan, float]
            ] = {}
            for rounds in config.rounds:
                for epsilon in config.epsilons:
                    maximum_library_width = max(
                        map(len, library.groups), default=1
                    )
                    if config.q > 2 and external_frozen_calibration:
                        cell_lambdas = frozen_width_weights[float(epsilon)]
                        width_weight_source = "development_isotonic_frozen"
                    elif config.q > 2 and data_split == "held_out":
                        cell_lambdas = frozen_width_weights[float(epsilon)]
                        width_weight_source = "development_isotonic"
                    else:
                        cell_lambdas = (0.0,) + tuple(
                            binary_response_power(width, epsilon)
                            for width in range(1, maximum_library_width + 1)
                        )
                        width_weight_source = (
                            "analytic_binary"
                            if config.q == 2
                            else "binary_exploratory"
                        )
                    if len(cell_lambdas) <= maximum_library_width:
                        raise RuntimeError(
                            "frozen width calibration does not cover candidate library: "
                            f"required={maximum_library_width}, "
                            f"available={len(cell_lambdas) - 1}"
                        )
                    for method in config.methods:
                        planning_start = time.perf_counter()
                        schedule_key = (method, rounds, epsilon)
                        planning_action_evaluations = 0
                        planning_state_evaluations = 0
                        proposal_count = 0
                        linear_solve_count = 0
                        adaptation_terminal_labels = (
                            config.adaptation_trajectories
                        )
                        projection_planning_time = None
                        if method in {
                            "p_saps_residualized",
                            "p_saps_projection",
                        }:
                            projection_key = (rounds, epsilon)
                            if projection_key not in projection_plan_cache:
                                projection_started = time.perf_counter()
                                projection_plan = _projection_saps_schedules(
                                    model,
                                    reward,
                                    rounds,
                                    library,
                                    epsilon,
                                    book,
                                    example_id,
                                    n_samples=config.adaptation_trajectories,
                                    ridge_multiplier=config.ridge_multipliers[0],
                                    geometry_cache=geometry_cache,
                                    shortlist_limit=16,
                                    frozen_budgeted_lambdas=cell_lambdas,
                                )
                                projection_plan_cache[projection_key] = (
                                    projection_plan,
                                    time.perf_counter() - projection_started,
                                )
                            projection_plan, projection_planning_time = (
                                projection_plan_cache[projection_key]
                            )
                            schedule_cache[schedule_key] = (
                                projection_plan.residualized
                                if method == "p_saps_residualized"
                                else projection_plan.projection
                            )
                            planning_state_evaluations = (
                                projection_plan.state_evaluations
                            )
                            proposal_count = projection_plan.proposal_count
                            linear_solve_count = (
                                projection_plan.linear_solve_count
                            )
                            geometry_count = (
                                projection_plan.residualized_geometry_count
                                if method == "p_saps_residualized"
                                else projection_plan.projection_geometry_count
                            )
                            adaptation_terminal_labels = (
                                config.adaptation_trajectories * geometry_count
                            )
                        if method in {
                            "saps_diagonal",
                            "b_saps_budgeted",
                        } and schedule_key not in schedule_cache:
                            max_width = max(
                                map(len, library.groups), default=1
                            )
                            if method == "saps_diagonal":
                                saps_lambdas = (
                                    0.0,
                                    1.0,
                                    *((0.0,) * (max_width - 1)),
                                )
                            else:
                                saps_lambdas = cell_lambdas
                            saps_result = _saps_result(
                                model.d,
                                balanced_capacities(model.d, rounds),
                                library,
                                saps_lambdas,
                                book,
                                example_id,
                            )
                            schedule_cache[schedule_key] = saps_result.schedule
                            planning_state_evaluations = (
                                saps_result.state_evaluations
                            )
                            proposal_count = 1
                        if schedule_key not in schedule_cache:
                            batch_oracle = None
                            if method.startswith("subset_"):
                                oracle_key = (rounds, epsilon)
                                if oracle_key not in batch_oracle_cache:
                                    batch_oracle_cache[oracle_key] = (
                                        _make_batch_value_oracle(
                                            model,
                                            reward,
                                            rounds,
                                            epsilon,
                                            config.rollouts,
                                            book,
                                            example_id,
                                        )
                                    )
                                batch_oracle = batch_oracle_cache[oracle_key]
                            oracle_evaluations_before = (
                                batch_oracle.evaluations
                                if batch_oracle is not None
                                else 0
                            )
                            schedule_cache[schedule_key] = _choose_schedule(
                                method,
                                model,
                                rounds,
                                library,
                                epsilon,
                                book,
                                example_id,
                                batch_oracle=batch_oracle,
                                reward=reward,
                                rollouts=config.rollouts,
                                dprm_beta=config.dprm_beta,
                            )
                            if batch_oracle is not None:
                                planning_action_evaluations = (
                                    batch_oracle.evaluations
                                    - oracle_evaluations_before
                                )
                            elif method in {"dprm_myopic", "dprm_rollout"}:
                                planning_action_evaluations = model.d
                        schedule = schedule_cache[schedule_key]
                        planning_time = (
                            projection_planning_time
                            if projection_planning_time is not None
                            else time.perf_counter() - planning_start
                        )
                        if schedule.batches not in geometry_cache:
                            geometry_started = time.perf_counter()
                            geometry_cache[schedule.batches] = _estimate_geometry(
                                model,
                                reward,
                                schedule,
                                library,
                                book,
                                example_id,
                                n_samples=config.adaptation_trajectories,
                                ridge_multiplier=config.ridge_multipliers[0],
                            )
                            geometry_time_cache[schedule.batches] = (
                                time.perf_counter() - geometry_started
                            )
                        geometry = geometry_cache[schedule.batches]
                        base_key = (schedule.batches, epsilon, False)
                        if base_key not in decode_cache:
                            decode_started = time.perf_counter()
                            decode_cache[base_key] = _decode(
                                model, reward, schedule, epsilon, config.rollouts,
                                book, example_id, False
                            )
                            decode_time_cache[base_key] = (
                                time.perf_counter() - decode_started
                            )
                        base, _, _, _ = decode_cache[base_key]
                        controlled_key = (schedule.batches, epsilon, True)
                        if controlled_key not in decode_cache:
                            decode_started = time.perf_counter()
                            decode_cache[controlled_key] = _decode(
                                model, reward, schedule, epsilon, config.rollouts,
                                book, example_id, True
                            )
                            decode_time_cache[controlled_key] = (
                                time.perf_counter() - decode_started
                            )
                        terminal, path_kl, model_calls, reward_calls = decode_cache[
                            controlled_key
                        ]
                        base_reward = reward(base)
                        terminal_reward = reward(terminal)
                        gain = terminal_reward - base_reward
                        # Reuse cached results for throughput, but charge every
                        # method the measured standalone adaptation/decode cost.
                        method_wall_time = (
                            planning_time
                            + geometry_time_cache.get(schedule.batches, 0.0)
                            + decode_time_cache[base_key]
                            + decode_time_cache[controlled_key]
                        )
                        schedule_json = json.dumps(schedule.batches)
                        schedule_identity = json.dumps(
                            {
                                "candidate_library": library.source,
                                "schedule": schedule.batches,
                            },
                            sort_keys=True,
                        )
                        schedule_id = hashlib.sha256(
                            schedule_identity.encode()
                        ).hexdigest()[:16]
                        success_thresholds.setdefault(example_id, {})[
                            schedule_id
                        ] = geometry.reward_q75
                        response_key = (schedule.batches, epsilon)
                        if response_key not in response_cache:
                            response_cache[response_key] = (
                                estimate_fisher_response_power(
                                    model,
                                    reward,
                                    schedule,
                                    geometry,
                                    epsilon,
                                    config.response_directions,
                                    config.rollouts,
                                    book,
                                    example_id,
                                    schedule_id,
                                )
                            )
                        response = response_cache[response_key]
                        if schedule.batches not in schedule_control_cache:
                            schedule_control_cache[schedule.batches] = (
                                _schedule_control_scores(
                                    model, schedule, pairwise_dependency
                                )
                            )
                        (
                            confidence_control,
                            entropy_control,
                            dependency_control,
                        ) = schedule_control_cache[schedule.batches]
                        lambdas = cell_lambdas
                        diag = first_order_score(schedule, library)
                        budgeted = budgeted_score(
                            schedule, library, epsilon, {epsilon: lambdas}
                        )
                        random_baseline = float(
                            sum(
                                weight
                                * balanced_random_budgeted_baseline(
                                    model.d,
                                    schedule.capacities,
                                    len(group),
                                    lambdas,
                                )
                                for group, weight in zip(
                                    library.groups, library.weights
                                )
                            )
                        )
                        width_histogram: dict[int, int] = {}
                        for group in library.groups:
                            width = frontier_width(schedule.round_of_position, group)
                            width_histogram[width] = width_histogram.get(width, 0) + 1
                        trajectories.append(
                            {
                                "example_id": example_id,
                                "data_split": data_split,
                                "seed": seed,
                                "generator_name": generator_name,
                                "reward_name": reward.name,
                                "candidate_library": library.source,
                                "scheduler": method,
                                "controller": "rollout_local",
                                "num_rounds": rounds,
                                "batch_sizes": json.dumps(schedule.capacities),
                                "schedule": schedule_json,
                                "schedule_id": schedule_id,
                                "mask_fraction": 1.0,
                                "epsilon_target": epsilon,
                                "path_token_kl": path_kl,
                                "schedule_policy_kl": 0.0,
                                "kl_saturated": False,
                                "n_model_calls": model_calls,
                                "n_reward_calls": reward_calls,
                                "wall_time_sec": method_wall_time,
                                "planning_time_sec": planning_time,
                                "planning_action_evaluations": planning_action_evaluations,
                                "planning_state_evaluations": planning_state_evaluations,
                                "proposal_count": proposal_count,
                                "linear_solve_count": linear_solve_count,
                                "adaptation_terminal_labels": adaptation_terminal_labels,
                                "width_weight_source": width_weight_source,
                                "terminal_reward": terminal_reward,
                                "uncontrolled_reward": base_reward,
                                "terminal_gain": gain,
                                "success": terminal_reward > geometry.reward_q75,
                                "validity": 1.0,
                                "diversity_group": hashlib.sha256(terminal.tobytes()).hexdigest()[:8],
                                "response_power_direct": response.response_power,
                                "response_achieved_kl": response.mean_achieved_kl,
                                "response_positive_rank": response.positive_rank,
                                "gram_delta": geometry.gram_delta,
                                "gram_condition": geometry.gram_condition,
                                "gamma_crossfit": float(
                                    np.mean(geometry.fold_gammas)
                                ),
                                "gamma_pinv": geometry.gamma_pinv,
                                "gamma_ridge": geometry.gamma_ridge,
                                "leakage_rho": np.nan,
                                "leakage_bound": np.nan,
                                "theory_report_sha256": config.theory_report_sha256,
                            }
                        )
                        scores.append(
                            {
                                "example_id": example_id,
                                "data_split": data_split,
                                "seed": seed,
                                "candidate_library": library.source,
                                "scheduler": method,
                                "num_rounds": rounds,
                                "epsilon_target": epsilon,
                                "schedule_id": schedule_id,
                                "schedule": schedule_json,
                                "score_random": random_baseline,
                                "score_confidence": confidence_control,
                                "score_entropy": entropy_control,
                                "score_dependency": dependency_control,
                                "score_dprm_itemwise": np.nan,
                                "score_diag_um": diag,
                                "score_projection": geometry.gamma_pinv,
                                "score_residualized": geometry.gamma_pinv,
                                "score_budgeted": budgeted,
                                "actual_response_power": response.response_power,
                                "actual_terminal_gain": gain,
                                "frontier_width_histogram": json.dumps(width_histogram),
                                "gamma_pinv": geometry.gamma_pinv,
                                "gamma_ridge": geometry.gamma_ridge,
                                "gram_delta": geometry.gram_delta,
                                "ridge_multiplier": config.ridge_multipliers[0],
                                "used_pinv": True,
                                "adaptation_terminal_labels": adaptation_terminal_labels,
                                "width_weight_source": width_weight_source,
                            }
                        )
                        for width in sorted(width_histogram):
                            probe_key = (schedule_id, epsilon, width)
                            if probe_key in emitted_probe_keys:
                                continue
                            emitted_probe_keys.add(probe_key)
                            width_key = (schedule.batches, epsilon, width)
                            if width_key not in width_response_cache:
                                width_geometry = restrict_geometry_to_width(
                                    schedule, geometry, width
                                )
                                if width_geometry is None:
                                    width_response_cache[width_key] = (
                                        FisherResponseEstimate(
                                            float("nan"), 0.0, 0, True
                                        )
                                    )
                                else:
                                    width_response_cache[width_key] = (
                                        estimate_fisher_response_power(
                                            model,
                                            reward,
                                            schedule,
                                            width_geometry,
                                            epsilon,
                                            config.response_directions,
                                            config.rollouts,
                                            book,
                                            example_id,
                                            schedule_id,
                                        )
                                    )
                            width_response = width_response_cache[width_key]
                            probes.append(
                                {
                                    "example_id": example_id,
                                    "data_split": data_split,
                                    "schedule_id": schedule_id,
                                    "epsilon": epsilon,
                                    "width": width,
                                    "actual_response_power": width_response.response_power,
                                    "mean_achieved_kl": width_response.mean_achieved_kl,
                                    "positive_rank": width_response.positive_rank,
                                    "kl_saturated": width_response.saturated,
                                    "response_space_empty": width_response.positive_rank == 0,
                                }
                            )
            print(
                f"completed example={example_id} library={library.source} "
                f"rows={len(trajectories)}",
                flush=True,
            )
    trajectory_frame = pd.DataFrame(trajectories)
    score_frame = pd.DataFrame(scores)
    probe_frame = pd.DataFrame(probes)
    width_calibration = frozen_width_calibration or width_calibration_summary(
        probe_frame,
        config.q,
        config.epsilons,
        minimum_per_width=config.width_calibration_minimum,
    )
    width_calibration = {
        **width_calibration,
        "frozen_before_held_out": entered_held_out,
    }
    score_frame.to_csv(staging / "schedule_scores.csv", index=False)
    trajectory_frame.to_parquet(staging / "trajectory_results.parquet", index=False)
    probe_frame.to_parquet(staging / "response_probes.parquet", index=False)
    for name in ("schedule_scores.csv", "trajectory_results.parquet", "response_probes.parquet"):
        (staging / name).replace(run_dir / name)
    artifact_hashes = {
        name: _sha256(run_dir / name)
        for name in ("schedule_scores.csv", "trajectory_results.parquet", "response_probes.parquet")
    }
    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "commit": None,
        "config": asdict(config),
        "instance_ids": instance_ids,
        "instance_coverage": _instance_coverage(config, instance_ids),
        "crossfit_folds": 5,
        "pinv_rtol": 1e-12,
        "ridge_sensitivity": {
            "multipliers": config.ridge_multipliers,
            "selected_rule": "first configured multiplier; steering gain unused",
        },
        "seed_version": "blake2b-json-v1",
        "direct_response_status": "fisher-whitened-path-kl-v1",
        "non_reward_dependency_status": (
            "pairwise conditional-MI/TC surrogate"
        ),
        "width_weights": width_calibration["weights"],
        "width_calibration": width_calibration,
        "frozen_calibration_sha256": frozen_calibration_sha256,
        "evaluation_protocol": {
            "name": "held-out_reward_instance_fixed-budget_few-shot_adaptation",
            "zero_shot": False,
            "reward_instance_specific_b": True,
            "adaptation_trajectories_per_schedule": config.adaptation_trajectories,
            "adaptation_budget_accounting": (
                "per-method terminal labels equal the per-schedule budget times "
                "the number of unique schedule geometries evaluated"
            ),
            "adaptation_gain_rng_disjoint": True,
            "schedule_planning_gain_rng_disjoint": True,
            "adaptation_rng_key": "base/example/adaptation/sample/round/position",
            "gain_evaluation_rng_key": (
                "base/example/gain-evaluation/epsilon/position; "
                "rollout/example/gain-evaluation/state/position/token/rollout"
            ),
            "schedule_planning_rng_key": (
                "base/example/subset-planning/epsilon/position; "
                "rollout/example/{subset,dprm}-planning/..."
            ),
            "claim_guardrail": "not zero-shot held-out-instance evaluation",
        },
        "success_thresholds": success_thresholds,
        "development_ids": [
            f"s{seed}-i{instance_id}"
            for seed in config.seeds
            for instance_id, split in sorted(instance_splits.items())
            if split == "development" and instance_id in instance_ids
        ],
        "held_out_ids": [
            f"s{seed}-i{instance_id}"
            for seed in config.seeds
            for instance_id, split in sorted(instance_splits.items())
            if split == "held_out" and instance_id in instance_ids
        ],
        "row_counts": {
            "trajectory_results": len(trajectory_frame),
            "schedule_scores": len(score_frame),
            "response_probes": len(probe_frame),
        },
        "artifact_sha256": artifact_hashes,
        "theory_report_sha256": config.theory_report_sha256,
        "theory_results_sha256": config.theory_results_sha256,
        "wall_time_sec": time.perf_counter() - started,
    }
    (staging / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (staging / "experiment_manifest.json").replace(
        run_dir / "experiment_manifest.json"
    )
    staging.rmdir()


def run_experiment(
    config: ExperimentConfig,
    run_dir: Path,
    instance_ids: tuple[int, ...] | None = None,
    frozen_calibration_path: Path | None = None,
) -> None:
    selected_ids = (
        tuple(range(config.num_instances))
        if instance_ids is None
        else tuple(instance_ids)
    )
    if (
        not selected_ids
        or len(set(selected_ids)) != len(selected_ids)
        or any(not 0 <= instance_id < config.num_instances for instance_id in selected_ids)
    ):
        raise ValueError(
            f"instance_ids must be unique values in [0, {config.num_instances}): "
            f"{selected_ids}"
        )
    supplied_calibration = None
    calibration_sha256 = None
    if frozen_calibration_path is not None:
        supplied_calibration, calibration_sha256 = _load_frozen_calibration(
            frozen_calibration_path, config
        )
    if (
        config.q > 2
        and selected_ids != tuple(range(config.num_instances))
        and supplied_calibration is None
    ):
        raise ValueError(
            "q>2 instance shards cannot preserve the global development calibration "
            "barrier; run the complete instance set or implement a separate frozen-"
            "calibration input artifact"
        )
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        _verify_theory(config)
        _run_experiment(
            config,
            run_dir,
            selected_ids,
            supplied_width_calibration=supplied_calibration,
            frozen_calibration_sha256=calibration_sha256,
        )
    except Exception as error:
        for artifact in run_dir.iterdir():
            if artifact.is_dir():
                shutil.rmtree(artifact)
            else:
                artifact.unlink()
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "message": str(error),
        }
        (run_dir / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8"
        )
        raise


class _CalibrationSeedBook:
    """Map every calibration draw into one isolated registered namespace."""

    def __init__(self, master_seed: int):
        self._seedbook = SeedBook(master_seed)

    def rng(self, namespace: str, *keys: object) -> np.random.Generator:
        return self._seedbook.rng(
            CALIBRATION_RNG_NAMESPACE, namespace, *keys
        )


class _CountingModel:
    def __init__(self, model, resources: dict[str, float]):
        self._model = model
        self._resources = resources

    def conditional_marginals(self, observed):
        self._resources["model_calls"] += 1
        return self._model.conditional_marginals(observed)

    def __getattr__(self, name: str):
        return getattr(self._model, name)


class _CountingReward:
    def __init__(self, reward, resources: dict[str, float]):
        self._reward = reward
        self._resources = resources

    def __call__(self, tokens):
        self._resources["terminal_label_calls"] += 1
        return self._reward(tokens)

    def __getattr__(self, name: str):
        return getattr(self._reward, name)


def run_width_calibration(config: ExperimentConfig, run_dir: Path) -> None:
    """Fit and freeze q-ary width weights using development data only."""
    if config.q <= 2:
        raise ValueError("width-targeted calibration bank is only for q>2")
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        _verify_theory(config)
        splits = _instance_splits(config)
        instance_cache: dict[tuple[int, int], tuple[object, object]] = {}
        targets: list[CalibrationTarget] = []
        for seed in config.seeds:
            for instance_id, data_split in sorted(splits.items()):
                if data_split != "development":
                    continue
                _, model, reward, _, _ = _instance(config, seed, instance_id)
                instance_cache[(seed, instance_id)] = (model, reward)
                for support in sorted(set(reward.supports)):
                    targets.append(
                        CalibrationTarget(
                            seed=seed,
                            instance_id=instance_id,
                            example_id=f"s{seed}-i{instance_id}",
                            data_split="development",
                            reward_name=reward.name,
                            support=tuple(support),
                        )
                    )
        maximum_width = max((len(target.support) for target in targets), default=0)
        widths = tuple(range(1, maximum_width + 1))
        bank = build_schedule_bank(
            d=config.d,
            rounds=config.rounds,
            targets=targets,
            widths=widths,
            maximum_per_width=config.calibration_max_per_cell,
            rng_seed=config.calibration_rng_seed,
        )
        resources: dict[str, float] = {
            "model_calls": 0,
            "terminal_label_calls": 0,
            "wall_time_sec": 0.0,
        }
        probe_rows: list[dict[str, object]] = []
        attempted_bank = []
        valid_counts = {
            (float(epsilon), width): 0
            for epsilon in config.epsilons
            for width in widths
        }
        calibration_seedbook = _CalibrationSeedBook(config.calibration_rng_seed)
        for width in widths:
            for record in (item for item in bank if item.target_width == width):
                if all(
                    valid_counts[(float(epsilon), width)]
                    >= config.calibration_requested_per_cell
                    for epsilon in config.epsilons
                ):
                    break
                attempted_bank.append(record)
                raw_model, raw_reward = instance_cache[
                    (record.seed, record.instance_id)
                ]
                model = _CountingModel(raw_model, resources)
                reward = _CountingReward(raw_reward, resources)
                library = CandidateLibrary(
                    (record.support,), CALIBRATION_RNG_NAMESPACE
                )
                calibration_example_id = (
                    f"{record.example_id}/width-calibration/{record.probe_id}"
                )
                geometry = None
                geometry_error = None
                try:
                    geometry = _estimate_geometry(
                        model,
                        reward,
                        record.schedule,
                        library,
                        calibration_seedbook,
                        calibration_example_id,
                        n_samples=config.adaptation_trajectories,
                        ridge_multiplier=config.ridge_multipliers[0],
                    )
                except Exception as error:
                    geometry_error = f"{type(error).__name__}: {error}"
                for epsilon in config.epsilons:
                    cell = (float(epsilon), width)
                    if (
                        valid_counts[cell]
                        >= config.calibration_requested_per_cell
                    ):
                        continue
                    response = FisherResponseEstimate(
                        float("nan"), 0.0, 0, True
                    )
                    error_message = geometry_error
                    if geometry is not None:
                        try:
                            response = estimate_fisher_response_power(
                                model,
                                reward,
                                record.schedule,
                                geometry,
                                float(epsilon),
                                config.response_directions,
                                config.rollouts,
                                calibration_seedbook,
                                calibration_example_id,
                                record.probe_id,
                            )
                        except Exception as error:
                            error_message = f"{type(error).__name__}: {error}"
                    valid = bool(
                        response.positive_rank > 0
                        and np.isfinite(response.response_power)
                    )
                    if valid:
                        valid_counts[cell] += 1
                    probe_rows.append(
                        {
                            "probe_id": record.probe_id,
                            "schedule_id": record.schedule_id,
                            "example_id": record.example_id,
                            "data_split": "development",
                            "epsilon": float(epsilon),
                            "width": width,
                            "support_size": record.support_size,
                            "actual_response_power": response.response_power,
                            "mean_achieved_kl": response.mean_achieved_kl,
                            "positive_rank": response.positive_rank,
                            "response_space_empty": response.positive_rank == 0,
                            "valid": valid,
                            "error": error_message,
                        }
                    )
        coverage = summarize_calibration_coverage(
            probe_rows,
            epsilons=config.epsilons,
            widths=widths,
            requested_per_cell=config.calibration_requested_per_cell,
            minimum_valid_per_cell=config.width_calibration_minimum,
            maximum_per_cell=config.calibration_max_per_cell,
        )
        probe_frame = pd.DataFrame(probe_rows)
        width_calibration = width_calibration_summary(
            probe_frame,
            config.q,
            config.epsilons,
            minimum_per_width=config.width_calibration_minimum,
        )
        width_calibration = {
            **width_calibration,
            "frozen_before_held_out": True,
            "source": "development-only width-targeted schedule bank",
        }
        resources["wall_time_sec"] = time.perf_counter() - started
        development_ids = sorted(
            f"s{seed}-i{instance_id}"
            for seed in config.seeds
            for instance_id, split in splits.items()
            if split == "development"
        )
        metadata = {
            "config": asdict(config),
            "epsilon_grid": list(config.epsilons),
            "width_grid": list(widths),
            "support_size_grid": sorted(
                {len(target.support) for target in targets}
            ),
            "batch_capacities": [
                list(balanced_capacities(config.d, rounds))
                for rounds in config.rounds
            ],
            "rounds": list(config.rounds),
            "requested_probes_per_cell": config.calibration_requested_per_cell,
            "minimum_valid_probes_per_cell": config.width_calibration_minimum,
            "maximum_probes_per_cell": config.calibration_max_per_cell,
            "rng_seed": config.calibration_rng_seed,
            "rng_namespace": CALIBRATION_RNG_NAMESPACE,
            "reward_instance_ids": development_ids,
            "development_ids": development_ids,
            "theory_report_sha256": config.theory_report_sha256,
            "theory_results_sha256": config.theory_results_sha256,
            "method_selection_use": "forbidden",
        }
        write_calibration_artifacts(
            run_dir,
            bank=attempted_bank,
            coverage=coverage,
            metadata=metadata,
            resources=resources,
            width_calibration=width_calibration,
            probes=probe_frame,
        )
        if (
            not bool(len(coverage))
            or not bool(coverage["passed"].all())
            or width_calibration["status"] != "complete"
        ):
            raise RuntimeError(
                "calibration_inconclusive: one or more (epsilon,width) cells "
                "have fewer than the minimum valid development probes"
            )
    except Exception as error:
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "message": str(error),
        }
        (run_dir / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8"
        )
        raise
