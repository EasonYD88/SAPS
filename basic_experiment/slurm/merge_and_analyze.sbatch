#!/bin/bash
#SBATCH --job-name=saps-merge
#SBATCH --account=torch_pr_281_general
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-user=yd2915@nyu.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: sbatch --dependency=afterok:ARRAY_JOB merge_and_analyze.sbatch RUN_ID N" >&2
  exit 2
fi

PROJECT_ROOT="/scratch/yd2915/SAPS/basic_experiment"
RUN_ID="$1"
SHARD_COUNT="$2"

module purge
module load anaconda3/2025.06

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

SHARD_DIRS=()
for ((SHARD_INDEX=0; SHARD_INDEX<SHARD_COUNT; SHARD_INDEX++)); do
  SHARD_DIRS+=("outputs/${RUN_ID}-shard-${SHARD_INDEX}-of-${SHARD_COUNT}")
done

python -c 'from feedback_frontier.cli import main; raise SystemExit(main())' \
  merge --run-id "$RUN_ID" --shard-dir "${SHARD_DIRS[@]}"
python -c 'from feedback_frontier.cli import main; raise SystemExit(main())' \
  analyze --run-dir "outputs/$RUN_ID"
echo "merged and analyzed run_id=$RUN_ID"
