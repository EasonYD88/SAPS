# Correlated Mechanism v3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the locked 16-instance smoke while adding an independent correlated-Potts mechanism bank and separating exact unary invariance from fixed-budget controller robustness.

**Architecture:** `ExperimentConfig.instance_design` selects either the frozen mixed design or a correlated-only design. The correlated design uses only coupling 0.7 and deterministically gives every `(reward, topology)` two instances per seed, one development and one held-out. A separate unary diagnostic runner replays only frozen schedule identities with independent diagnostic RNG domains; it never changes schedules, calibration, or held-out gain estimates.

**Tech Stack:** Python 3.11+, NumPy, pandas, PyArrow, pytest, SLURM CPU jobs.

## Global Constraints

- Keep `configs/smoke.yaml` at 16 instances, 2 seeds, 2 rollouts, 40 adaptation trajectories, 4 rounds, and the existing epsilon grid.
- Do not reuse calibration probes, method gains, or held-out labels to construct the correlated bank.
- Correlated diagnostic instances use coupling 0.7 only and a new run ID.
- Unary robustness repetitions are conditional algorithmic diagnostics, not additional reward-instance samples.
- Preserve the v2 `NO_GO`; v3 artifacts and gate semantics use new run IDs.
- Record model calls, terminal-label calls, wall time, RNG keys, source run hash, and artifact SHA-256.

---

### Task 1: Instance-design contract

**Files:**
- Modify: `basic_experiment/src/feedback_frontier/config.py`
- Modify: `basic_experiment/src/feedback_frontier/runners/evaluate_planner.py`
- Modify: `basic_experiment/tests/test_smoke_contract.py`

**Interfaces:**
- Produces: `ExperimentConfig.instance_design: str`
- Produces: `_instance_spec(config, instance_id)` and `_instance_splits(config)` with correlated coverage validation.

- [x] Write a failing test asserting that `correlated_potts` with 16 instances, four rewards, two topologies, and coupling 0.7 produces exactly one development and one held-out instance for every `(reward, topology, 0.7)` cell.
- [x] Run `python -m pytest tests/test_smoke_contract.py -q` and confirm the new test fails because `instance_design` is missing.
- [x] Add the minimal config validation and deterministic mapping.
- [x] Re-run the focused tests and the frozen smoke-contract tests.

### Task 2: Correlated diagnostic bank and coverage manifest

**Files:**
- Create: `basic_experiment/configs/smoke_correlated_diagnostic.yaml`
- Modify: `basic_experiment/src/feedback_frontier/runners/evaluate_planner.py`
- Modify: `basic_experiment/src/feedback_frontier/runners/merge.py`
- Test: `basic_experiment/tests/test_runner_analysis.py`

**Interfaces:**
- Produces manifest field `instance_coverage`, keyed by split/reward/generator/topology/coupling.

- [x] Write a failing integration test requiring manifest coverage to contain every correlated cell and no product or zero-coupling cell.
- [x] Run the focused test and confirm the missing-field failure.
- [x] Add coverage serialization and merge-time aggregation; reject duplicate or missing correlated cells.
- [x] Add the frozen correlated config without changing `configs/smoke.yaml`.
- [x] Verify focused tests pass.

### Task 3: Exact and fixed-budget unary diagnostics

**Files:**
- Create: `basic_experiment/src/feedback_frontier/runners/unary_diagnostics.py`
- Modify: `basic_experiment/src/feedback_frontier/cli.py`
- Modify: `basic_experiment/src/feedback_frontier/analysis/aggregate.py`
- Create: `basic_experiment/tests/test_unary_diagnostics.py`
- Modify: `basic_experiment/tests/test_analysis.py`

**Interfaces:**
- Produces: `run_unary_diagnostics(config, source_run_dir, controller_replicates, terminal_replicates)`.
- Produces: `unary_null_diagnostics.parquet` and `unary_null_diagnostic_manifest.json`.
- Produces mechanism records `M5_exact` and `M5_fixed_budget`; the latter resamples controller replicate IDs and explicitly reports the fixed held-out reward-instance count.

- [x] Write failing tests showing exact product-unary action values produce zero paired differences for different schedules under common uniforms.
- [x] Write failing tests requiring controller and terminal RNG keys to be disjoint from adaptation, gain evaluation, and calibration.
- [x] Implement the smallest exact cache and repeated fixed-budget replay over frozen schedule identities.
- [x] Write failing analysis tests for `M5_exact` and `M5_fixed_budget`, including `INCONCLUSIVE` when diagnostics are absent or reward-instance coverage is insufficient.
- [x] Implement the split gate records without treating repetitions as reward-instance replication.
- [x] Verify diagnostic and analysis tests pass.

### Task 4: v3 DAG and verification

**Files:**
- Create: `basic_experiment/slurm/run_unary_diagnostics.sbatch`
- Create: `basic_experiment/slurm/submit_smoke_v3_dag.sh`
- Modify: `basic_experiment/README.md`

- [x] Add a SLURM script with account, CPU partition, terminal-state email, explicit resource limits, and diagnostic cost reporting.
- [x] Add a DAG submission script with calibration feeding both frozen smoke and correlated bank; unary diagnostics depend only on the merged smoke schedules. Document forbidden flows `held-out evaluation -> calibration`, `method results -> calibration`, and `calibration probes -> held-out gain estimates`.
- [x] Run `python -m pytest -q` and config validation.
- [x] Submit the DAG with new v3 run IDs, query every job, and report job IDs and log paths.
