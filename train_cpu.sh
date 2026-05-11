#!/bin/bash


#SBATCH --job-name=train
#SBATCH --partition=math-alderaan-short
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --ntasks=4
#SBATCH --mem=64G

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
    echo "Usage: $0 <config_path>"
    exit 1
fi

CONFIG_PATH="$1"
echo "Config path: $CONFIG_PATH"

# Set up environment
source ~/.bashrc
conda activate gpu_TEST

# If this job is part of an array, SLURM will define SLURM_ARRAY_TASK_ID.
# Use it as the seed for reproducibility across replications.
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SEED="${SLURM_ARRAY_TASK_ID}"
    echo "Array task detected. Using seed: $SEED"
    python src/train.py "$CONFIG_PATH" "$SEED"
else
    echo "Non-array job. No seed provided."
    python src/train.py "$CONFIG_PATH"
fi
