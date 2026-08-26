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

Do **not** shard q>2 runs. Held-out B-SAPS must use one width calibration frozen
from the complete development split before any held-out cell is evaluated. The
runner rejects q>2 instance shards until the array workflow has a separate,
immutable development-calibration artifact.
