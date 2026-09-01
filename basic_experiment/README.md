# SAPS Basic Experiment

Evaluation semantics: this benchmark uses **held-out reward instances with fixed-budget few-shot adaptation**, not zero-shot held-out instances. For a held-out reward instance, instance-specific \(b_{r,c}\) and \(F_{r,c}\) are fitted from terminal-label adaptation trajectories whose RNG domain is disjoint from final gain evaluation. `adaptation_terminal_labels` reports the method-level label budget, including every unique schedule geometry queried during P-SAPS shortlist selection.

CPU-only synthetic benchmark implementing the theory contract in
../math_theory/correlated_budgeted_feedback_frontier_report.md.

Install:

    python -m pip install -e '.[dev]'

Validate the smoke configuration:

    feedback-frontier validate-config --config configs/smoke.yaml

Run tests:

    python -m pytest -q

Run on a Torch CPU compute node (the output and error logs include the Slurm
job id):

    mkdir -p logs
    sbatch --time=02:00:00 slurm/run_experiment.sbatch \
      configs/smoke.yaml phase1-smoke-slurm-v1

For q=2 runs, analytic width weights permit instance sharding. Submit an array
and then an `afterok` merge job:

    mkdir -p logs
    ARRAY_JOB=$(sbatch --parsable --array=0-15 slurm/run_array.sbatch \
      configs/smoke.yaml phase1-smoke-v2 16)
    sbatch --dependency="afterok:${ARRAY_JOB}" slurm/merge_and_analyze.sbatch \
      phase1-smoke-v2 16

For q>2 runs, first build the independent development-only width calibration
bank, then pass its frozen manifest to every evaluation shard:

    CAL_JOB=$(sbatch --parsable slurm/run_calibration.sbatch \
      configs/smoke_calibration.yaml phase1-smoke-calibration-v1)
    ARRAY_JOB=$(sbatch --parsable --dependency="afterok:${CAL_JOB}" \
      --array=0-15%16 slurm/run_frozen_array.sbatch \
      configs/smoke.yaml phase1-smoke-v1 16 \
      outputs/phase1-smoke-calibration-v1/calibration_manifest.json)
    sbatch --dependency="afterok:${ARRAY_JOB}" \
      slurm/merge_and_analyze.sbatch phase1-smoke-v1 16

The calibration stage keeps the locked smoke grid unchanged and constructs
targeted balanced schedules for widths 1--5 under the registered
`width_calibration_bank` RNG namespace. It uses only development reward
instances and stops a cell only after 24 valid probes (up to 32 attempts),
while the hard acceptance threshold remains 20. It writes and hashes
`calibration_schedule_bank.jsonl`, `calibration_coverage.csv`,
`calibration_response_probes.parquet`, and `calibration_manifest.json`.
The manifest reports calibration model calls, terminal-label calls, and wall
time separately from held-out few-shot adaptation and gain evaluation.

The frozen information-flow contract forbids `held-out evaluation ->
calibration`, `method results -> calibration`, and `calibration probes ->
held-out gain estimates`. Evaluation shards receive only the frozen width
weights after manifest, artifact-hash, coverage, and development-ID checks.

## v3 correlated mechanism diagnostics

The locked 16-instance smoke remains unchanged. A separate 16-instance-per-seed
bank uses `configs/smoke_correlated_diagnostic.yaml`; every reward has chain and
balanced-tree Potts instances at coupling 0.7 in both development and held-out
splits. It uses its own development-only calibration artifact from
`configs/smoke_correlated_calibration.yaml` and an instance RNG key containing
`correlated_potts`, so it cannot reuse smoke reward instances accidentally.

M5 is reported as two records. `M5_exact` checks the exact product-unary theorem
control. `M5_fixed_budget` repeats the unchanged two-rollout controller with 16
controller and 16 terminal RNG realizations. Those repetitions measure
conditional algorithmic randomness; the manifest reports the unchanged number
of held-out reward instances and never counts repetitions as new instances.

Submit the complete CPU DAG with new v3 run IDs:

    bash slurm/submit_smoke_v3_dag.sh

The DAG runs two independent calibration jobs, the frozen smoke and correlated
arrays, both merges, the unary diagnostic, and the final smoke re-analysis. The
unary manifest separately reports model calls, terminal-label calls, wall time,
source hashes, artifact hashes, and diagnostic RNG domains.
