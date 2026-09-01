#!/bin/bash
#SBATCH --job-name=saps-unary-diagnostic
#SBATCH --account=torch_pr_281_general
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-user=yd2915@nyu.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: sbatch run_unary_diagnostics.sbatch CONFIG SOURCE_RUN_DIR" >&2
  exit 2
fi

PROJECT_ROOT="/scratch/yd2915/SAPS/basic_experiment"
CONFIG_PATH="$1"
SOURCE_RUN_DIR="$2"

module purge
module load anaconda3/2025.06

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python -c 'from feedback_frontier.cli import main; raise SystemExit(main())' \
  diagnose-unary --config "$CONFIG_PATH" --source-run-dir "$SOURCE_RUN_DIR" \
  --controller-replicates 16 --terminal-replicates 16
python -c 'import json,sys; p=json.load(open(sys.argv[1])); print(json.dumps(p["resources"], sort_keys=True))' \
  "$SOURCE_RUN_DIR/unary_null_diagnostic_manifest.json"
