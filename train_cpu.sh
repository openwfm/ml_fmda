#!/bin/bash


#SBATCH --job-name=train
#SBATCH --partition=math-alderaan
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=142G

# Runs a single training job.
#
# Two modes:
# 1) Normal (non-array) job:
#      sbatch train_cpu.sh <config_path>
#    -> calls: python src/train.py <config_path>
#
# 2) SLURM array job (submitted with --array):
#      sbatch --array=0-9 train_cpu.sh <config_path>
#    -> uses SLURM_ARRAY_TASK_ID as the seed
#    -> calls: python src/train.py <config_path> <seed>

set -euo pipefail

# Expect exactly one user argument: config path.
# (Seed is not passed explicitly; arrays use SLURM_ARRAY_TASK_ID.)
if [ "$#" -ne 1 ]; then
    echo "Error: Expected exactly 1 arguments, but got $#."
    echo "Usage: $0 <target_dir>"
    exit 1
fi

TARGET_DIR="$1"
echo "Config path: $TARGET_DIR"

# Set up environment
eval "$(conda shell.bash hook)"
echo "Activating conda env: gpu_TEST"
conda activate gpu_TEST

# If this job is part of an array, SLURM will define SLURM_ARRAY_TASK_ID.
# Use it as the seed for reproducibility across replications.
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SEED="${SLURM_ARRAY_TASK_ID}"
    echo "Array task detected. Using seed: $SEED"
    echo "Running executable Python script: python src/train.py \"$TARGET_DIR\" \"$SEED\""
    python src/train.py "$TARGET_DIR" "$SEED"
else
    echo "Non-array job. No seed provided."
    echo "Running executable Python script: python src/train.py \"$TARGET_DIR\""
    python src/train.py "$TARGET_DIR"
fi
