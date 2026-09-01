from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from feedback_frontier.config import ExperimentConfig
from feedback_frontier.controllers.rollout_local import RolloutCache
from feedback_frontier.generators.categorical_product import CategoricalProduct
from feedback_frontier.rewards.synthetic import SyntheticReward
from feedback_frontier.runners import evaluate_planner
from feedback_frontier.schedulers.base import Schedule


def diagnostic_rng_domains(
    controller_replicate: int, terminal_replicate: int
) -> tuple[str, str]:
    if controller_replicate < 0 or terminal_replicate < 0:
        raise ValueError("diagnostic replicate ids must be nonnegative")
    return (
        f"unary-diagnostic/controller-{controller_replicate}",
        f"unary-diagnostic/terminal-{terminal_replicate}",
    )


class ExactUnaryActionCache:
    """Exact action values for a product model with a unary reward."""

    def __init__(
        self, model: CategoricalProduct, reward: SyntheticReward
    ) -> None:
        if not isinstance(model, CategoricalProduct):
            raise TypeError("exact unary diagnostics require a product model")
        if reward.name != "unary" or any(len(term.support) != 1 for term in reward.terms):
            raise ValueError("exact unary diagnostics require a unary reward")
        self.model = model
        self.reward = reward
        self.n_model_calls = 0
        self.n_reward_calls = 0
        tables = np.zeros((model.d, model.q), dtype=float)
        for term in reward.terms:
            position = term.support[0]
            if term.table.shape != (model.q,):
                raise ValueError("unary reward table shape must match model q")
            tables[position] += np.asarray(term.table, dtype=float)
        self._tables = tables

    def action_values(
        self, observed: dict[int, int], position: int, rollouts: int
    ) -> NDArray[np.float64]:
        del rollouts
        if position in observed:
            raise ValueError("action-value position must be unresolved")
        constant = sum(
            self._tables[index, token] for index, token in observed.items()
        )
        constant += sum(
            float(self.model.probs[index] @ self._tables[index])
            for index in range(self.model.d)
            if index not in observed and index != position
        )
        return self._tables[position].copy() + float(constant)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_example_id(example_id: str) -> tuple[int, int]:
    match = re.fullmatch(r"s(\d+)-i(\d+)", example_id)
    if match is None:
        raise ValueError(f"invalid example_id in frozen schedules: {example_id}")
    return int(match.group(1)), int(match.group(2))


def _schedule(serialized: str, d: int) -> Schedule:
    batches = tuple(tuple(int(value) for value in batch) for batch in json.loads(serialized))
    return Schedule(batches, d)


def run_unary_diagnostics(
    config: ExperimentConfig,
    source_run_dir: Path,
    controller_replicates: int,
    terminal_replicates: int,
) -> None:
    if controller_replicates < 2 or terminal_replicates < 2:
        raise ValueError("unary diagnostics require at least two replicates per axis")
    started = time.perf_counter()
    scores_path = source_run_dir / "schedule_scores.csv"
    source_manifest_path = source_run_dir / "experiment_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "complete":
        raise ValueError("unary diagnostics require a complete frozen source run")
    scores = pd.read_csv(scores_path)
    required = {
        "example_id",
        "data_split",
        "candidate_library",
        "scheduler",
        "num_rounds",
        "epsilon_target",
        "schedule_id",
        "schedule",
    }
    if not required <= set(scores):
        raise ValueError("frozen schedule scores lack unary diagnostic columns")
    scores = scores.loc[scores["data_split"] == "held_out", sorted(required)]
    rows: list[dict[str, object]] = []
    model_calls = 0
    terminal_label_calls = 0
    reward_instances: set[str] = set()
    group_columns = [
        "example_id",
        "candidate_library",
        "num_rounds",
        "epsilon_target",
    ]
    for group_key, group in scores.groupby(group_columns, sort=True):
        example_id, candidate_library, rounds, epsilon = group_key
        seed, instance_id = _parse_example_id(str(example_id))
        reward_name, regime, _, _ = evaluate_planner._instance_spec(
            config, instance_id
        )
        if reward_name != "unary" or regime != "product":
            continue
        if evaluate_planner._instance_splits(config)[instance_id] != "held_out":
            raise ValueError("source marks a non-held-out instance as held-out")
        random_rows = group.loc[group["scheduler"] == "random_balanced"]
        method_rows = group.loc[group["scheduler"] != "random_balanced"]
        if len(random_rows) != 1 or method_rows.empty:
            continue
        reward_instances.add(str(example_id))
        book, model, reward, generator_name, _ = evaluate_planner._instance(
            config, seed, instance_id
        )
        if generator_name != "product" or not isinstance(model, CategoricalProduct):
            raise ValueError("exact unary diagnostics require product instances")
        random_row = random_rows.iloc[0]
        random_schedule = _schedule(str(random_row["schedule"]), config.d)

        exact_random, _, _, _ = evaluate_planner._decode(
            model,
            reward,
            random_schedule,
            float(epsilon),
            config.rollouts,
            book,
            str(example_id),
            True,
            rollout_cache=ExactUnaryActionCache(model, reward),
            rng_domain="unary-diagnostic/exact/terminal-0",
        )
        exact_random_reward = float(reward(exact_random))
        terminal_label_calls += 1
        for method_row in method_rows.itertuples(index=False):
            method_schedule = _schedule(str(method_row.schedule), config.d)
            exact_terminal, _, _, _ = evaluate_planner._decode(
                model,
                reward,
                method_schedule,
                float(epsilon),
                config.rollouts,
                book,
                str(example_id),
                True,
                rollout_cache=ExactUnaryActionCache(model, reward),
                rng_domain="unary-diagnostic/exact/terminal-0",
            )
            exact_reward = float(reward(exact_terminal))
            terminal_label_calls += 1
            rows.append(
                {
                    "diagnostic_kind": "exact",
                    "example_id": example_id,
                    "candidate_library": candidate_library,
                    "scheduler": method_row.scheduler,
                    "num_rounds": int(rounds),
                    "epsilon_target": float(epsilon),
                    "schedule_id": method_row.schedule_id,
                    "random_schedule_id": random_row["schedule_id"],
                    "controller_replicate": -1,
                    "terminal_replicate": 0,
                    "method_terminal_reward": exact_reward,
                    "random_terminal_reward": exact_random_reward,
                    "paired_difference": exact_reward - exact_random_reward,
                }
            )

        for controller_id in range(controller_replicates):
            controller_domain, _ = diagnostic_rng_domains(controller_id, 0)
            cache = RolloutCache(
                model,
                reward,
                book,
                str(example_id),
                rng_domain=controller_domain,
            )
            for terminal_id in range(terminal_replicates):
                _, terminal_domain = diagnostic_rng_domains(
                    controller_id, terminal_id
                )
                random_terminal, _, _, _ = evaluate_planner._decode(
                    model,
                    reward,
                    random_schedule,
                    float(epsilon),
                    config.rollouts,
                    book,
                    str(example_id),
                    True,
                    rollout_cache=cache,
                    rng_domain=terminal_domain,
                )
                random_reward = float(reward(random_terminal))
                terminal_label_calls += 1
                for method_row in method_rows.itertuples(index=False):
                    method_terminal, _, _, _ = evaluate_planner._decode(
                        model,
                        reward,
                        _schedule(str(method_row.schedule), config.d),
                        float(epsilon),
                        config.rollouts,
                        book,
                        str(example_id),
                        True,
                        rollout_cache=cache,
                        rng_domain=terminal_domain,
                    )
                    method_reward = float(reward(method_terminal))
                    terminal_label_calls += 1
                    rows.append(
                        {
                            "diagnostic_kind": "fixed_budget",
                            "example_id": example_id,
                            "candidate_library": candidate_library,
                            "scheduler": method_row.scheduler,
                            "num_rounds": int(rounds),
                            "epsilon_target": float(epsilon),
                            "schedule_id": method_row.schedule_id,
                            "random_schedule_id": random_row["schedule_id"],
                            "controller_replicate": controller_id,
                            "terminal_replicate": terminal_id,
                            "method_terminal_reward": method_reward,
                            "random_terminal_reward": random_reward,
                            "paired_difference": method_reward - random_reward,
                        }
                    )
            model_calls += cache.n_model_calls
            terminal_label_calls += cache.n_reward_calls
    if not rows:
        raise ValueError("no held-out product-unary schedule pairs were found")
    output_path = source_run_dir / "unary_null_diagnostics.parquet"
    temporary_path = source_run_dir / ".unary_null_diagnostics.parquet.tmp"
    pd.DataFrame(rows).to_parquet(temporary_path, index=False)
    temporary_path.replace(output_path)
    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(source_run_dir),
        "source_schedule_scores_sha256": _sha256(scores_path),
        "source_experiment_manifest_sha256": _sha256(source_manifest_path),
        "source_columns_used": sorted(required),
        "reward_instance_count": len(reward_instances),
        "reward_instance_ids": sorted(reward_instances),
        "controller_replicates": controller_replicates,
        "terminal_replicates": terminal_replicates,
        "rollouts_per_action_value": config.rollouts,
        "rng_contract": {
            "controller": "rollout/example/unary-diagnostic/controller-{id}/...",
            "terminal": "base/example/unary-diagnostic/terminal-{id}/...",
            "disjoint_from": [
                "adaptation",
                "gain-evaluation",
                "subset-planning",
                "dprm-planning",
                "width_calibration_bank",
            ],
        },
        "interpretation": (
            "replicates estimate conditional algorithmic randomness and do not "
            "increase the held-out reward-instance sample size"
        ),
        "information_flow": {
            "held-out gain values -> diagnostics": "forbidden",
            "diagnostics -> calibration": "forbidden",
            "diagnostics -> schedule construction": "forbidden",
        },
        "resources": {
            "model_calls": model_calls,
            "terminal_label_calls": terminal_label_calls,
            "wall_time_sec": time.perf_counter() - started,
        },
        "row_count": len(rows),
        "artifact_sha256": {
            "unary_null_diagnostics.parquet": _sha256(output_path)
        },
    }
    manifest_path = source_run_dir / "unary_null_diagnostic_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
