from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="feedback-frontier")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--config", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--shard-index", type=int)
    run.add_argument("--num-shards", type=int)
    run.add_argument("--frozen-calibration", type=Path)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--run-id", required=True)
    merge.add_argument("--shard-dir", type=Path, nargs="+", required=True)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "validate-config":
        cfg = ExperimentConfig.from_yaml(args.config)
        print(f"valid: d={cfg.d} q={cfg.q} instances={cfg.num_instances}")
        return 0
    if args.command == "run":
        from .runners.evaluate_planner import (
            _shard_instance_ids,
            run_experiment,
        )

        cfg = ExperimentConfig.from_yaml(args.config)
        if (args.shard_index is None) != (args.num_shards is None):
            parser.error("--shard-index and --num-shards must be provided together")
        instance_ids = (
            None
            if args.shard_index is None
            else _shard_instance_ids(
                cfg.num_instances, args.shard_index, args.num_shards
            )
        )
        run_experiment(
            cfg,
            Path("outputs") / args.run_id,
            instance_ids=instance_ids,
            frozen_calibration_path=args.frozen_calibration,
        )
        return 0
    if args.command == "merge":
        from .runners.merge import merge_shard_runs

        merge_shard_runs(tuple(args.shard_dir), Path("outputs") / args.run_id)
        return 0
    if args.command == "analyze":
        from .analysis.aggregate import analyze_run

        analyze_run(args.run_dir)
        return 0
    raise AssertionError(args.command)
