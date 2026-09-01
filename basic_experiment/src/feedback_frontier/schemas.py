from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrajectoryRecord:
    example_id: str
    data_split: str
    seed: int
    generator_name: str
    reward_name: str
    candidate_library: str
    scheduler: str
    controller: str
    num_rounds: int
    batch_sizes: str
    schedule: str
    schedule_id: str
    mask_fraction: float
    epsilon_target: float
    path_token_kl: float
    schedule_policy_kl: float
    kl_saturated: bool
    n_model_calls: int
    n_reward_calls: int
    wall_time_sec: float
    planning_time_sec: float
    planning_action_evaluations: int
    planning_state_evaluations: int
    proposal_count: int
    linear_solve_count: int
    adaptation_terminal_labels: int
    width_weight_source: str
    terminal_reward: float
    uncontrolled_reward: float
    terminal_gain: float
    success: bool
    validity: float
    diversity_group: str
    response_power_direct: float
    response_achieved_kl: float
    response_positive_rank: int
    gram_delta: float
    gram_condition: float
    gamma_crossfit: float
    gamma_pinv: float
    gamma_ridge: float
    leakage_rho: float
    leakage_bound: float
    theory_report_sha256: str


@dataclass(frozen=True)
class ScheduleScoreRecord:
    example_id: str
    data_split: str
    seed: int
    candidate_library: str
    scheduler: str
    num_rounds: int
    epsilon_target: float
    schedule_id: str
    schedule: str
    score_random: float
    score_confidence: float
    score_entropy: float
    score_dependency: float
    score_dprm_itemwise: float
    score_diag_um: float
    score_projection: float
    score_residualized: float
    score_budgeted: float
    actual_response_power: float
    actual_terminal_gain: float
    frontier_width_histogram: str
    gamma_pinv: float
    gamma_ridge: float
    gram_delta: float
    ridge_multiplier: float
    used_pinv: bool
    adaptation_terminal_labels: int
    width_weight_source: str
