from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from feedback_frontier.estimators.response_power import width_calibration_summary


RAW_ARTIFACTS = (
    "schedule_scores.csv",
    "trajectory_results.parquet",
    "response_probes.parquet",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_frame(run_dir: Path, name: str) -> pd.DataFrame:
    return (
        pd.read_csv(run_dir / name)
        if name.endswith(".csv")
        else pd.read_parquet(run_dir / name)
    )


def _select_external_frozen_calibration(
    manifests: list[dict],
) -> tuple[dict | None, str | None]:
    digests = [manifest.get("frozen_calibration_sha256") for manifest in manifests]
    if all(digest is None for digest in digests):
        return None, None
    if any(digest is None for digest in digests) or len(set(digests)) != 1:
        raise ValueError("shards must use one common frozen calibration artifact")
    calibrations = [manifest.get("width_calibration") for manifest in manifests]
    serialized = {json.dumps(value, sort_keys=True) for value in calibrations}
    if None in calibrations or len(serialized) != 1:
        raise ValueError("shards have mismatched frozen calibration payloads")
    return dict(calibrations[0]), str(digests[0])


def _merge_instance_coverage(manifests: list[dict]) -> list[dict[str, object]]:
    count_by_cell: dict[tuple[object, ...], int] = {}
    fields = (
        "data_split",
        "reward_name",
        "generator_regime",
        "topology",
        "coupling",
    )
    for manifest in manifests:
        for row in manifest.get("instance_coverage", ()):
            key = tuple(row.get(field) for field in fields)
            count_by_cell[key] = count_by_cell.get(key, 0) + int(row["count"])
    return [
        {
            **dict(zip(fields, key, strict=True)),
            "count": count,
        }
        for key, count in sorted(
            count_by_cell.items(), key=lambda item: tuple(map(str, item[0]))
        )
    ]


def merge_shard_runs(shard_dirs: tuple[Path, ...], run_dir: Path) -> None:
    if not shard_dirs:
        raise ValueError("shard coverage is empty")
    run_dir.mkdir(parents=True, exist_ok=False)
    staging = run_dir / ".staging"
    staging.mkdir()
    try:
        manifests = [
            json.loads((path / "experiment_manifest.json").read_text())
            for path in shard_dirs
        ]
        if any(manifest.get("status") != "complete" for manifest in manifests):
            raise ValueError("shard coverage includes an incomplete run")
        reference = manifests[0]
        invariant_keys = (
            "package_version",
            "commit",
            "crossfit_folds",
            "pinv_rtol",
            "ridge_sensitivity",
            "seed_version",
            "direct_response_status",
            "non_reward_dependency_status",
            "evaluation_protocol",
        )
        for manifest in manifests[1:]:
            if manifest.get("config") != reference.get("config"):
                raise ValueError("shard coverage has mismatched configs")
            if (
                manifest.get("theory_report_sha256")
                != reference.get("theory_report_sha256")
                or manifest.get("theory_results_sha256")
                != reference.get("theory_results_sha256")
            ):
                raise ValueError("shard coverage has mismatched theory hashes")
            for key in invariant_keys:
                if manifest.get(key) != reference.get(key):
                    raise ValueError(f"shard coverage has mismatched {key}")
        instance_ids = [
            int(instance_id)
            for manifest in manifests
            for instance_id in manifest.get("instance_ids", ())
        ]
        expected = list(range(int(reference["config"]["num_instances"])))
        if len(instance_ids) != len(set(instance_ids)) or sorted(instance_ids) != expected:
            raise ValueError(
                f"shard coverage must be disjoint and complete: "
                f"got={sorted(instance_ids)}, expected={expected}"
            )
        for shard_dir, manifest in zip(shard_dirs, manifests):
            for name in RAW_ARTIFACTS:
                expected_hash = manifest.get("artifact_sha256", {}).get(name)
                if expected_hash is None or _sha256(shard_dir / name) != expected_hash:
                    raise ValueError(f"shard artifact hash mismatch: {shard_dir / name}")

        merged_frames: dict[str, pd.DataFrame] = {}
        for name in RAW_ARTIFACTS:
            frames = [_read_frame(path, name) for path in shard_dirs]
            columns = [tuple(frame.columns) for frame in frames]
            if any(column_set != columns[0] for column_set in columns[1:]):
                raise ValueError(f"shard schema mismatch: {name}")
            merged_frames[name] = pd.concat(frames, ignore_index=True)
        merged_frames["schedule_scores.csv"].to_csv(
            staging / "schedule_scores.csv", index=False
        )
        merged_frames["trajectory_results.parquet"].to_parquet(
            staging / "trajectory_results.parquet", index=False
        )
        merged_frames["response_probes.parquet"].to_parquet(
            staging / "response_probes.parquet", index=False
        )
        for name in RAW_ARTIFACTS:
            (staging / name).replace(run_dir / name)
        hashes = {name: _sha256(run_dir / name) for name in RAW_ARTIFACTS}
        development_ids = sorted(
            example_id
            for shard_manifest in manifests
            for example_id in shard_manifest.get("development_ids", ())
        )
        held_out_ids = sorted(
            example_id
            for shard_manifest in manifests
            for example_id in shard_manifest.get("held_out_ids", ())
        )
        success_thresholds: dict[str, dict[str, float]] = {}
        for shard_manifest in manifests:
            for example_id, thresholds in shard_manifest.get(
                "success_thresholds", {}
            ).items():
                if example_id in success_thresholds:
                    raise ValueError(
                        f"shard coverage duplicates success thresholds: {example_id}"
                    )
                success_thresholds[example_id] = thresholds
        config = reference["config"]
        external_calibration, frozen_calibration_sha256 = (
            _select_external_frozen_calibration(manifests)
        )
        if external_calibration is None:
            width_calibration = width_calibration_summary(
                merged_frames["response_probes.parquet"],
                int(config.get("q", 0)),
                tuple(float(value) for value in config.get("epsilons", ())),
                minimum_per_width=int(config.get("width_calibration_minimum", 20)),
            )
            width_calibration["frozen_before_held_out"] = all(
                bool(
                    manifest.get("width_calibration", {}).get(
                        "frozen_before_held_out"
                    )
                )
                for manifest in manifests
            )
        else:
            width_calibration = external_calibration
        manifest = {
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": reference["config"],
            "instance_ids": expected,
            "instance_coverage": _merge_instance_coverage(manifests),
            "merged_from": [str(path) for path in shard_dirs],
            "development_ids": development_ids,
            "held_out_ids": held_out_ids,
            "success_thresholds": success_thresholds,
            "width_weights": width_calibration["weights"],
            "width_calibration": width_calibration,
            "frozen_calibration_sha256": frozen_calibration_sha256,
            "theory_report_sha256": reference.get("theory_report_sha256"),
            "theory_results_sha256": reference.get("theory_results_sha256"),
            "row_counts": {
                name.rsplit(".", 1)[0]: len(frame)
                for name, frame in merged_frames.items()
            },
            "artifact_sha256": hashes,
        }
        for key in invariant_keys:
            manifest[key] = reference.get(key)
        (staging / "experiment_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        (staging / "experiment_manifest.json").replace(
            run_dir / "experiment_manifest.json"
        )
        staging.rmdir()
    except Exception as error:
        for artifact in run_dir.iterdir():
            if artifact.is_dir():
                shutil.rmtree(artifact)
            else:
                artifact.unlink()
        (run_dir / "failure.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        raise
