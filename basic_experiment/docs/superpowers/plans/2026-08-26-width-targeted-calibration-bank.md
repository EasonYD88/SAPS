# Width-Targeted Calibration Schedule Bank Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a frozen, development-only schedule bank that gives every locked smoke `(epsilon, width)` calibration cell at least 20 valid probes and then resubmit the calibration -> held-out array -> merge/analyze DAG.

**Architecture:** Put schedule construction and artifact serialization in a focused `calibration_bank.py` module. The runner invokes it only for q-ary Stage A development calibration, evaluates bank schedules with a dedicated RNG namespace, freezes coverage/resources/hashes, and refuses to enter held-out evaluation unless every cell passes. Existing Stage B continues to consume only the frozen artifact.

**Tech Stack:** Python 3.13, NumPy, pandas, PyArrow, pytest, SLURM.

## Global Constraints

- Preserve d=12, q=4, num_instances=16, rounds=[1,2,3,4], and balanced capacities.
- Preserve modular support sizes 3--5 and width cells 1--5.
- Request 24 probes per cell, allow at most 32 attempts, require valid_count >= 20.
- Use only development reward instances and a dedicated `width_calibration_bank` RNG namespace.
- Do not use held-out data, held-out gain, method scores, method rankings, or method performance.
- Do not interpolate, merge cells, or borrow probes across widths/epsilons.
- Freeze the bank, coverage, manifest, SHA-256 hashes, forbidden information flows, and separate resource totals.

---

### Task 1: Deterministic width-targeted schedules

**Files:**
- Create: `src/feedback_frontier/runners/calibration_bank.py`
- Test: `tests/test_calibration_bank.py`

**Interfaces:**
- Produces: `construct_targeted_schedule(d, capacities, support, target_width, seedbook, example_id, replicate) -> Schedule`
- Produces: `build_schedule_bank(config, development_instances, requested_per_cell=24, max_per_cell=32) -> list[CalibrationSchedule]`

- [ ] Write tests asserting exact target widths 1--5, balanced capacities, determinism, distinct calibration RNG namespace, and development-only IDs.
- [ ] Run `PYTHONPATH=src pytest tests/test_calibration_bank.py -q`; expect failure because the module does not exist.
- [ ] Implement canonical targeted construction and deterministic bank selection without method inputs.
- [ ] Re-run the focused tests; expect PASS.

### Task 2: Coverage and frozen artifact contract

**Files:**
- Modify: `src/feedback_frontier/runners/calibration_bank.py`
- Modify: `src/feedback_frontier/estimators/response_power.py`
- Test: `tests/test_calibration_bank.py`
- Test: `tests/test_response_power.py`

**Interfaces:**
- Produces: `summarize_calibration_coverage(records, epsilons, widths, requested, minimum) -> pandas.DataFrame`
- Produces: `write_calibration_artifacts(run_dir, bank, coverage, metadata, resources) -> dict[str, str]`

- [ ] Write failing tests for independent cell counts, `valid_count >= 20`, no borrowing, inconclusive short cells, JSONL/CSV/manifest schemas, forbidden flows, and SHA-256 verification.
- [ ] Run focused tests and confirm expected failures.
- [ ] Implement coverage validation, serialization, hashes, and resource schema.
- [ ] Re-run focused tests; expect PASS.

### Task 3: Stage A runner integration and Stage B validation

**Files:**
- Modify: `src/feedback_frontier/runners/evaluate_planner.py`
- Modify: `src/feedback_frontier/runners/merge.py`
- Modify: `tests/test_runner_analysis.py`
- Modify: `tests/test_smoke_contract.py`

**Interfaces:**
- Stage A evaluates bank schedules before any held-out cell and writes four frozen calibration artifacts.
- Stage B loader verifies calibration-manifest and schedule-bank hashes and accepts only complete coverage.

- [ ] Write failing runner tests proving bank probes are development-only, method-independent, frozen before held-out, and separately account model/terminal-label calls/wall time.
- [ ] Run focused runner tests and confirm expected failures.
- [ ] Integrate bank evaluation and loader/hash validation; remove dependence on configured method schedules for calibration coverage.
- [ ] Re-run focused runner tests; expect PASS.

### Task 4: Configuration, documentation, and local verification

**Files:**
- Modify: `src/feedback_frontier/config.py`
- Modify: `configs/smoke_calibration.yaml`
- Modify: `configs/smoke.yaml`
- Modify: `configs/screening.yaml`
- Modify: `configs/phase1_main.yaml`
- Modify: `README.md`
- Modify: `tests/test_config.py`

**Interfaces:**
- Adds locked fields `calibration_requested_per_cell=24` and `calibration_max_per_cell=32` with validation `minimum <= requested <= max`.

- [ ] Write failing config/contract tests for exact values and invalid ordering.
- [ ] Run focused config tests and confirm expected failures.
- [ ] Add fields, configs, manifests, and documentation without changing the locked grid.
- [ ] Run `PYTHONPATH=src python -m pytest -q`; expect all tests PASS.

### Task 5: SLURM DAG submission and monitoring

**Files:**
- Modify only if required: `slurm/run_calibration.sbatch`
- Modify only if required: `slurm/run_frozen_array.sbatch`
- Modify only if required: `slurm/merge_and_analyze.sbatch`

- [ ] Validate smoke and calibration configs locally.
- [ ] Submit Stage A to the fastest legal CPU partition with a fresh run ID.
- [ ] Submit the 16-task held-out array with `afterok:<stage-a-job>` and the frozen calibration manifest path.
- [ ] Submit merge/analyze with `afterok:<array-job>`.
- [ ] Verify states/reasons and report job IDs plus stdout/stderr paths.
