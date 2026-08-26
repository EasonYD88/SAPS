from __future__ import annotations

import itertools
import inspect
import math

import numpy as np
import pytest

from feedback_frontier.candidates.libraries import CandidateLibrary
from feedback_frontier.controllers.rollout_local import (
    RolloutCache,
    calibrate_alpha,
    categorical_kl,
)
from feedback_frontier.generators.categorical_product import CategoricalProduct
from feedback_frontier.rng import SeedBook
from feedback_frontier.runners import evaluate_planner
from feedback_frontier.schedulers.base import Schedule
from feedback_frontier.schedulers.dprm import process_value, select_top_itemwise
from feedback_frontier.schedulers.saps import (
    exhaustive_saps,
    residualized_group_weights,
    rerank_schedule_shortlist,
)
from feedback_frontier.schedulers.structured import (
    LaminarNode,
    all_colorings_with_capacities,
    capacity_preserving_swaps,
    laminar_dp,
    pairwise_milp,
    path_capacity_dp,
)
from feedback_frontier.schedulers.subset import BatchValueOracle, beam_subset, exact_subset


def test_alpha_calibration_hits_target_kl() -> None:
    base = np.array([0.6, 0.3, 0.1])
    values = np.array([-1.0, 0.2, 1.4])
    result = calibrate_alpha(base, values, 0.05)
    assert categorical_kl(result.probabilities, base) == pytest.approx(0.05, abs=1e-6)
    constant = calibrate_alpha(base, np.ones(3), 0.05)
    assert constant.alpha == 0.0 and constant.saturated


def test_process_value_has_mean_limit() -> None:
    rewards = np.array([-1.0, 0.0, 2.0])
    assert process_value(rewards, 0.0) == pytest.approx(rewards.mean())
    assert process_value(rewards, 1.0) > rewards.mean()


def test_rollout_cache_reuses_action_values() -> None:
    model = CategoricalProduct(np.zeros((3, 2)))
    cache = RolloutCache(
        model,
        lambda x: float(x.sum()),
        SeedBook(4),
        "example",
    )
    first = cache.action_values({}, 0, 3)
    calls = (cache.n_model_calls, cache.n_reward_calls)
    second = cache.action_values({}, 0, 3)
    np.testing.assert_array_equal(first, second)
    assert (cache.n_model_calls, cache.n_reward_calls) == calls == (6, 6)


def test_exact_subset_matches_direct_enumeration_and_cache() -> None:
    values = {
        batch: float(sum(batch) + (10 if batch == (1, 4, 5) else 0))
        for batch in itertools.combinations(range(6), 3)
    }
    oracle = BatchValueOracle(lambda batch: values[batch])
    result = exact_subset(tuple(range(6)), 3, oracle, d=6)
    assert result.batch == max(values, key=values.get)
    calls = oracle.evaluations
    assert exact_subset(tuple(range(6)), 3, oracle, d=6).batch == result.batch
    assert oracle.evaluations == calls
    assert beam_subset(tuple(range(6)), 3, oracle, beam_width=32).batch == result.batch


def test_regular_cycle_itemwise_tie_has_exponential_coordination_gap() -> None:
    m = 6
    batches = list(itertools.combinations(range(2 * m), m))
    cuts = [
        sum(((i in batch) != ((i + 1) % (2 * m) in batch)) for i in range(2 * m))
        for batch in map(set, batches)
    ]
    assert sum(cut == 2 * m for cut in cuts) / len(cuts) == pytest.approx(
        2 / math.comb(2 * m, m)
    )
    assert select_top_itemwise(np.ones(2 * m), m) == tuple(range(m))


def test_pairwise_milp_matches_balanced_enumeration() -> None:
    edges = ((0, 1, 1.0), (1, 2, 2.0), (2, 3, 0.5), (3, 4, 1.2), (4, 5, 0.8), (5, 0, 1.3))
    capacities = (3, 3)
    result = pairwise_milp(6, capacities, edges)
    brute = max(
        sum(w for i, j, w in edges if colors[i] != colors[j])
        for colors in all_colorings_with_capacities(6, capacities)
    )
    assert result.objective == pytest.approx(brute)
    assert result.schedule.capacities == capacities


def test_laminar_dp_matches_bruteforce() -> None:
    root = LaminarNode(
        (0, 1, 2, 3),
        1.0,
        (
            LaminarNode((0, 1), 0.6, (LaminarNode((0,), 0), LaminarNode((1,), 0))),
            LaminarNode((2, 3), 0.4, (LaminarNode((2,), 0), LaminarNode((3,), 0))),
        ),
    )
    lambdas = (0.0, 1.0, 0.2, 0.05, 0.01)
    result = laminar_dp(root, (2, 2), lambdas)
    nodes = (root, *root.children)
    brute = -math.inf
    for colors in all_colorings_with_capacities(4, (2, 2)):
        value = 0.0
        for node in nodes:
            final = max(colors[i] for i in node.leaves)
            width = sum(colors[i] == final for i in node.leaves)
            value += node.weight * lambdas[width]
        brute = max(brute, value)
    assert result.objective == pytest.approx(brute)


def test_path_dp_matches_bruteforce() -> None:
    weights = (0.4, 1.2, 0.7, 0.5)
    result = path_capacity_dp(weights, (3, 2), lambda_same=0.2)
    brute = max(
        sum(
            weight * (1.0 if colors[i] != colors[i + 1] else 0.2)
            for i, weight in enumerate(weights)
        )
        for colors in all_colorings_with_capacities(5, (3, 2))
    )
    assert result.objective == pytest.approx(brute)


def test_exhaustive_budgeted_saps_returns_optimum() -> None:
    library = CandidateLibrary(((0, 1, 2), (2, 3, 4)), "oracle", (0.7, 0.3))
    lambdas = (0.0, 1.0, 0.25, 0.05)
    result = exhaustive_saps(6, (2, 2, 2), library, lambdas)
    values = []
    for colors in all_colorings_with_capacities(6, (2, 2, 2)):
        schedule = Schedule(
            tuple(tuple(i for i, c in enumerate(colors) if c == ell) for ell in range(3)),
            6,
        )
        values.append(
            sum(
                weight * lambdas[
                    sum(
                        colors[i] == max(colors[j] for j in group)
                        for i in group
                    )
                ]
                for group, weight in zip(library.groups, library.weights)
            )
        )
    assert result.objective == pytest.approx(max(values))


def test_capacity_preserving_swaps_never_decrease_objective() -> None:
    library = CandidateLibrary(((0, 1), (1, 2), (2, 3)), "structural", (1.0, 2.0, 1.0))
    lambdas = (0.0, 1.0, 0.2)
    initial = Schedule(((0, 1), (2, 3)), 4)
    result = capacity_preserving_swaps(initial, library, lambdas)
    from feedback_frontier.schedulers.saps import budgeted_objective

    assert result.objective >= budgeted_objective(initial, library, lambdas)
    assert result.schedule.capacities == initial.capacities


def test_residualized_group_weights_do_not_double_count_duplicate_blocks() -> None:
    schedule = Schedule(((1,), (0,)), 2)
    library = CandidateLibrary(((0,), (0, 1)), "structural")
    metadata = (((0,), 0, 0), ((0, 1), 0, 0))
    weights, linear_solves = residualized_group_weights(
        schedule,
        library,
        np.array([1.0, 1.0]),
        np.ones((2, 2)),
        metadata,
    )
    assert weights == pytest.approx((1.0, 0.0))
    assert linear_solves == 2


def test_projection_shortlist_reranking_is_unique_and_deterministic() -> None:
    first = Schedule(((0, 1), (2, 3)), 4)
    second = Schedule(((0, 2), (1, 3)), 4)
    third = Schedule(((0, 3), (1, 2)), 4)
    values = {
        first.batches: 0.5,
        second.batches: 0.8,
        third.batches: 0.8,
    }
    result = rerank_schedule_shortlist(
        (third, first, second, third),
        lambda schedule: values[schedule.batches],
        lambda schedule: 2.0,
    )
    assert result.schedule == second
    assert result.objective == pytest.approx(0.8)
    assert result.proposal_count == 3
    assert result.linear_solve_count == 3


def test_projection_saps_builds_residualized_and_reranked_schedules() -> None:
    planner = getattr(evaluate_planner, "_projection_saps_schedules", None)
    assert callable(planner), "runner must expose projection-aware SAPS planning"
    model = CategoricalProduct(np.zeros((4, 2)))
    library = CandidateLibrary(((0, 1), (0, 2), (2, 3)), "structural")

    def reward(tokens: np.ndarray) -> float:
        return float((tokens[0] == tokens[2]) + 0.5 * (tokens[2] == tokens[3]))

    result = planner(
        model,
        reward,
        rounds=2,
        library=library,
        epsilon=0.05,
        seedbook=SeedBook(19),
        example_id="projection-saps",
        n_samples=20,
        ridge_multiplier=1e-3,
        geometry_cache={},
        shortlist_limit=8,
    )
    assert result.residualized.capacities == (2, 2)
    assert result.projection.capacities == (2, 2)
    assert result.proposal_count >= 2
    assert result.linear_solve_count >= len(library.groups)
    assert result.state_evaluations > 0


def test_projection_saps_caps_shortlist_at_feasible_schedule_count() -> None:
    model = CategoricalProduct(np.zeros((4, 2)))
    library = CandidateLibrary(((0, 1),), "structural")
    result = evaluate_planner._projection_saps_schedules(
        model,
        lambda tokens: float(tokens.sum()),
        rounds=1,
        library=library,
        epsilon=0.05,
        seedbook=SeedBook(23),
        example_id="single-round-projection-saps",
        n_samples=20,
        ridge_multiplier=1e-3,
        geometry_cache={},
        shortlist_limit=8,
    )
    assert result.proposal_count == 1
    assert result.state_evaluations <= 4


def test_runner_subset_exact_uses_batch_value_oracle() -> None:
    planner = getattr(evaluate_planner, "_subset_schedule", None)
    assert callable(planner), "runner must expose subset planning through BatchValueOracle"
    target = (1, 4, 5)
    oracle = BatchValueOracle(lambda batch: 10.0 if batch == target else 0.0)
    schedule = planner(
        "subset_exact",
        d=6,
        capacities=(3, 3),
        confidence_order=(0, 1, 2, 3, 4, 5),
        oracle=oracle,
    )
    assert schedule.batches == (target, (0, 2, 3))
    assert oracle.evaluations == math.comb(6, 3)


def test_choose_schedule_routes_subset_methods_to_shared_oracle() -> None:
    assert "batch_oracle" in inspect.signature(
        evaluate_planner._choose_schedule
    ).parameters
    model = CategoricalProduct(np.zeros((6, 2)))
    library = CandidateLibrary(((0, 1),), "oracle", (1.0,))
    target = (1, 4, 5)
    oracle = BatchValueOracle(lambda batch: 10.0 if batch == target else 0.0)
    schedule = evaluate_planner._choose_schedule(
        "subset_exact",
        model,
        2,
        library,
        0.05,
        SeedBook(3),
        "example",
        batch_oracle=oracle,
    )
    assert schedule.batches[0] == target


def test_batch_value_oracle_reuses_rollout_cache_across_candidate_batches() -> None:
    class CountingProduct(CategoricalProduct):
        def __init__(self) -> None:
            super().__init__(np.zeros((4, 2)))
            self.sample_conditioned_calls = 0

        def sample_conditioned(self, observed, uniforms):
            self.sample_conditioned_calls += 1
            return super().sample_conditioned(observed, uniforms)

    model = CountingProduct()
    oracle = evaluate_planner._make_batch_value_oracle(
        model,
        lambda tokens: float(tokens.sum()),
        rounds=2,
        epsilon=0.05,
        rollouts=1,
        seedbook=SeedBook(31),
        example_id="shared-subset-rollouts",
    )
    oracle.value((0, 1))
    oracle.value((0, 2))
    # Without a cache shared across candidates this is 16: eight rollout
    # samples per decode. Position 0 at the empty observed state is reusable.
    assert model.sample_conditioned_calls < 16


def test_runner_dprm_uses_reward_conditioned_process_values() -> None:
    planner = getattr(evaluate_planner, "_dprm_schedule", None)
    assert callable(planner), "runner must rank DPRM positions by process value"
    model = CategoricalProduct(np.zeros((4, 2)))
    weights = np.array([0.0, 0.0, 2.0, 4.0])

    def reward(tokens: np.ndarray) -> float:
        return float(weights @ tokens)

    schedule, evaluations = planner(
        model,
        reward,
        rounds=2,
        rollouts=8,
        beta=1.0,
        seedbook=SeedBook(11),
        example_id="dprm-example",
    )
    assert schedule.batches[0] == (2, 3)
    assert evaluations == model.d


def test_myopic_and_rollout_dprm_use_distinct_continuation_policies() -> None:
    class CountingProduct:
        d = 4
        q = 2

        def __init__(self) -> None:
            self.completion_calls = 0

        def conditional_marginals(
            self, observed: dict[int, int]
        ) -> np.ndarray:
            probabilities = np.full((self.d, self.q), 0.5)
            for position, token in observed.items():
                probabilities[position] = 0.0
                probabilities[position, token] = 1.0
            return probabilities

        def sample_conditioned(
            self,
            observed: dict[int, int],
            uniforms: np.ndarray,
        ) -> np.ndarray:
            self.completion_calls += 1
            tokens = (np.asarray(uniforms) >= 0.5).astype(int)
            for position, token in observed.items():
                tokens[position] = token
            return tokens

    reward = lambda tokens: float(np.asarray(tokens).sum())
    myopic_model = CountingProduct()
    rollout_model = CountingProduct()
    _, myopic_evaluations = evaluate_planner._dprm_schedule(
        myopic_model,
        reward,
        rounds=3,
        rollouts=2,
        beta=1.0,
        seedbook=SeedBook(31),
        example_id="dprm-policy",
        method="dprm_myopic",
        epsilon=0.05,
    )
    _, rollout_evaluations = evaluate_planner._dprm_schedule(
        rollout_model,
        reward,
        rounds=3,
        rollouts=2,
        beta=1.0,
        seedbook=SeedBook(31),
        example_id="dprm-policy",
        method="dprm_rollout",
        epsilon=0.05,
    )
    assert myopic_model.completion_calls == 0
    assert rollout_model.completion_calls > 0
    assert myopic_evaluations == rollout_evaluations == myopic_model.d


def test_choose_schedule_routes_dprm_to_process_values() -> None:
    parameters = inspect.signature(evaluate_planner._choose_schedule).parameters
    assert {"reward", "rollouts", "dprm_beta"} <= set(parameters)
    model = CategoricalProduct(np.zeros((4, 2)))
    weights = np.array([0.0, 0.0, 2.0, 4.0])

    def reward(tokens: np.ndarray) -> float:
        return float(weights @ tokens)

    schedule = evaluate_planner._choose_schedule(
        "dprm_myopic",
        model,
        2,
        CandidateLibrary(((0,), (1,)), "oracle", (0.5, 0.5)),
        0.05,
        SeedBook(11),
        "dprm-example",
        reward=reward,
        rollouts=8,
        dprm_beta=1.0,
    )
    assert schedule.batches[0] == (2, 3)
