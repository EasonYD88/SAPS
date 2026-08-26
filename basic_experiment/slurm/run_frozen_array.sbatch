#!/bin/bash
#SBATCH --job-name=saps-frozen
#SBATCH --account=torch_pr_281_general
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --mail-user=yd2915@nyu.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: sbatch --array=0-(N-1) run_frozen_array.sbatch CONFIG RUN_ID N CALIBRATION_MANIFEST" >&2
  exit 2
fi

PROJECT_ROOT="/scratch/yd2915/SAPS/basic_experiment"
CONFIG_PATH="$1"
RUN_ID="$2"
SHARD_COUNT="$3"
CALIBRATION_MANIFEST="$4"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
SHARD_RUN_ID="${RUN_ID}-shard-${SHARD_INDEX}-of-${SHARD_COUNT}"

module purge
module load anaconda3/2025.06

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "job_id=$SLURM_JOB_ID shard=$SHARD_INDEX/$SHARD_COUNT calibration=$CALIBRATION_MANIFEST"
python -c 'from feedback_frontier.cli import main; raise SystemExit(main())' \
  run --config "$CONFIG_PATH" --run-id "$SHARD_RUN_ID" \
  --shard-index "$SHARD_INDEX" --num-shards "$SHARD_COUNT" \
  --frozen-calibration "$CALIBRATION_MANIFEST"
echo "completed run_id=$SHARD_RUN_ID"
