#!/bin/bash
#SBATCH --job-name=saps-calibrate
#SBATCH --account=torch_pr_281_general
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-user=yd2915@nyu.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: sbatch run_calibration.sbatch CONFIG_PATH RUN_ID" >&2
  exit 2
fi

PROJECT_ROOT="/scratch/yd2915/SAPS/basic_experiment"
CONFIG_PATH="$1"
RUN_ID="$2"

module purge
module load anaconda3/2025.06

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "job_id=$SLURM_JOB_ID host=$(hostname) config=$CONFIG_PATH run_id=$RUN_ID"
python -c 'from feedback_frontier.cli import main; raise SystemExit(main())' \
  validate-config --config "$CONFIG_PATH"
python -c 'from feedback_frontier.cli import main; raise SystemExit(main())' \
  calibrate --config "$CONFIG_PATH" --run-id "$RUN_ID"
echo "completed calibration run_id=$RUN_ID"
