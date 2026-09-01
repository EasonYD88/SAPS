#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/scratch/yd2915/SAPS/basic_experiment"
ACCOUNT="torch_pr_281_general"
PARTITION="cs"
MAIN_RUN_ID="phase1-smoke-targeted-v3"
CORRELATED_RUN_ID="phase1-smoke-correlated-v3"
MAIN_CALIBRATION_ID="phase1-smoke-calibration-bank-v3"
CORRELATED_CALIBRATION_ID="phase1-smoke-correlated-calibration-bank-v3"
SHARD_COUNT=16

# Frozen forbidden information flows:
# held-out evaluation -> calibration
# method results -> calibration
# calibration probes -> held-out gain estimates

cd "$PROJECT_ROOT"
mkdir -p logs

MAIN_CALIBRATION_JOB=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" \
  slurm/run_calibration.sbatch configs/smoke_calibration.yaml "$MAIN_CALIBRATION_ID")
CORRELATED_CALIBRATION_JOB=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" \
  slurm/run_calibration.sbatch configs/smoke_correlated_calibration.yaml \
  "$CORRELATED_CALIBRATION_ID")

MAIN_ARRAY_JOB=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" \
  --dependency="afterok:$MAIN_CALIBRATION_JOB" --array="0-15" \
  slurm/run_frozen_array.sbatch configs/smoke.yaml "$MAIN_RUN_ID" "$SHARD_COUNT" \
  "outputs/$MAIN_CALIBRATION_ID/calibration_manifest.json")
CORRELATED_ARRAY_JOB=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" \
  --dependency="afterok:$CORRELATED_CALIBRATION_JOB" --array="0-15" \
  slurm/run_frozen_array.sbatch configs/smoke_correlated_diagnostic.yaml \
  "$CORRELATED_RUN_ID" "$SHARD_COUNT" \
  "outputs/$CORRELATED_CALIBRATION_ID/calibration_manifest.json")

MAIN_MERGE_JOB=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" \
  --dependency="afterok:$MAIN_ARRAY_JOB" slurm/merge_and_analyze.sbatch \
  "$MAIN_RUN_ID" "$SHARD_COUNT")
CORRELATED_MERGE_JOB=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" \
  --dependency="afterok:$CORRELATED_ARRAY_JOB" slurm/merge_and_analyze.sbatch \
  "$CORRELATED_RUN_ID" "$SHARD_COUNT")

UNARY_DIAGNOSTIC_JOB=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" \
  --dependency="afterok:$MAIN_MERGE_JOB" slurm/run_unary_diagnostics.sbatch \
  configs/smoke.yaml "outputs/$MAIN_RUN_ID")
FINAL_ANALYSIS_JOB=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" \
  --dependency="afterok:$UNARY_DIAGNOSTIC_JOB" --job-name=saps-v3-final-analysis \
  --cpus-per-task=1 --mem=8G --time=02:00:00 \
  --output=logs/%x-%j.out --error=logs/%x-%j.err \
  --mail-user=yd2915@nyu.edu --mail-type=END,FAIL,TIME_LIMIT \
  --wrap="cd $PROJECT_ROOT; module purge; module load anaconda3/2025.06; export PYTHONPATH=$PROJECT_ROOT/src MPLBACKEND=Agg; python -c 'from feedback_frontier.cli import main; raise SystemExit(main())' analyze --run-dir outputs/$MAIN_RUN_ID")

printf '%s\n' \
  "main_calibration=$MAIN_CALIBRATION_JOB" \
  "correlated_calibration=$CORRELATED_CALIBRATION_JOB" \
  "main_array=$MAIN_ARRAY_JOB" \
  "correlated_array=$CORRELATED_ARRAY_JOB" \
  "main_merge=$MAIN_MERGE_JOB" \
  "correlated_merge=$CORRELATED_MERGE_JOB" \
  "unary_diagnostic=$UNARY_DIAGNOSTIC_JOB" \
  "final_analysis=$FINAL_ANALYSIS_JOB"
