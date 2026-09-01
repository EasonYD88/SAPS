from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


VALID_REWARDS = {"unary", "pairwise", "modular", "mixed"}
VALID_INSTANCE_DESIGNS = {"mixed", "correlated_potts"}
VALID_METHODS = {
    "random_balanced", "confidence", "entropy", "dependency_cmi",
    "min_within_batch_tc", "dprm_myopic", "dprm_rollout", "subset_exact",
    "subset_beam_8", "subset_beam_32", "saps_diagonal",
    "p_saps_residualized", "p_saps_projection", "b_saps_budgeted",
}


@dataclass(frozen=True)
class ExperimentConfig:
    d: int
    q: int
    methods: tuple[str, ...]
    rounds: tuple[int, ...]
    epsilons: tuple[float, ...]
    rollouts: int
    instance_design: str = "mixed"
    num_instances: int = 1
    couplings: tuple[float, ...] = (0.0,)
    topologies: tuple[str, ...] = ("chain",)
    rewards: tuple[str, ...] = ("unary",)
    candidate_libraries: tuple[str, ...] = ("oracle",)
    seeds: tuple[int, ...] = (0,)
    response_directions: int = 8
    adaptation_trajectories: int = 20
    width_calibration_minimum: int = 20
    calibration_requested_per_cell: int = 24
    calibration_max_per_cell: int = 32
    calibration_rng_seed: int = 20260826
    dprm_beta: float = 1.0
    ridge_multipliers: tuple[float, ...] = (1e-3,)
    bootstrap_replicates: int = 10_000
    theory_report_sha256: str = ""
    theory_results_sha256: str = ""

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("configuration must be a mapping")
        tuple_fields = {
            "methods", "rounds", "epsilons", "couplings", "topologies",
            "rewards", "candidate_libraries", "seeds", "ridge_multipliers",
        }
        allowed = {field.name for field in fields(cls)}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
        for key in tuple_fields & raw.keys():
            raw[key] = tuple(raw[key])
        cfg = cls(**raw)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.d < 1 or self.q < 2:
            raise ValueError(f"d and q must be positive; got d={self.d}, q={self.q}")
        if "subset_exact" in self.methods and self.d > 14:
            raise ValueError("exact_subset requires d <= 14")
        unknown_methods = set(self.methods) - VALID_METHODS
        if unknown_methods:
            raise ValueError(f"unknown methods: {sorted(unknown_methods)}")
        unknown_rewards = set(self.rewards) - VALID_REWARDS
        if unknown_rewards:
            raise ValueError(f"unknown rewards: {sorted(unknown_rewards)}")
        if self.instance_design not in VALID_INSTANCE_DESIGNS:
            raise ValueError(
                "unknown instance_design: "
                f"{self.instance_design!r}; expected one of "
                f"{sorted(VALID_INSTANCE_DESIGNS)}"
            )
        if self.num_instances % len(self.rewards):
            raise ValueError(
                "num_instances must be divisible by the reward count; got "
                f"{self.num_instances} and {len(self.rewards)}"
            )
        if self.instance_design == "correlated_potts":
            cells = len(self.topologies) * len(self.couplings)
            repeats, remainder = divmod(
                self.num_instances // len(self.rewards), cells
            )
            if remainder or repeats < 2 or repeats % 2:
                raise ValueError(
                    "correlated_potts requires an even number of at least two "
                    "instances per (reward, topology, coupling) cell"
                )
            if any(coupling <= 0 for coupling in self.couplings):
                raise ValueError(
                    "correlated_potts couplings must be strictly positive"
                )
        if self.rollouts < 1:
            raise ValueError(f"rollouts must be positive, got {self.rollouts}")
        if self.adaptation_trajectories < 10:
            raise ValueError(
                "adaptation_trajectories must be at least 10, got "
                f"{self.adaptation_trajectories}"
            )
        if self.width_calibration_minimum < 1:
            raise ValueError(
                "width_calibration_minimum must be positive, got "
                f"{self.width_calibration_minimum}"
            )
        if not (
            self.width_calibration_minimum
            <= self.calibration_requested_per_cell
            <= self.calibration_max_per_cell
        ):
            raise ValueError(
                "calibration probe budgets require minimum <= requested <= maximum; "
                f"got {self.width_calibration_minimum} <= "
                f"{self.calibration_requested_per_cell} <= "
                f"{self.calibration_max_per_cell}"
            )
        if self.calibration_rng_seed < 0:
            raise ValueError("calibration_rng_seed must be nonnegative")
        if any(not 1 <= rounds <= self.d for rounds in self.rounds):
            raise ValueError(f"rounds must be within [1, d], got {self.rounds}")
        if any(epsilon <= 0.0 for epsilon in self.epsilons):
            raise ValueError(f"epsilons must be positive, got {self.epsilons}")
