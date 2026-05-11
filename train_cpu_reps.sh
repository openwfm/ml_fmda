#!/bin/bash


#SBATCH --job-name=trep
#SBATCH --partition=math-alderaan-short
#SBATCH --output=logs/trep_%j.out
#SBATCH --ntasks=4
#SBATCH --mem=64G

# This script is a *submitter* (runs on the login node).
# It submits a SLURM job array where each task runs one replication.
#
# Usage:
#   bash train_reps.sh <config_path> <nreps>
#
# What it does:
#   - Submits: sbatch --array=0-(nreps-1) train_cpu.sh <config_path>
#   - Each array task sets SLURM_ARRAY_TASK_ID
#   - train_cpu.sh uses that as the seed and calls:
#       python src/train.py <config_path> <seed>

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Error: Expected exactly 2 arguments, but got $#."
    echo "Usage: $0 <config_path> <nreps>"
    exit 1
fi

CONFIG_PATH="$1"
NREPS="$2"

# Basic validation
if ! [[ "$NREPS" =~ ^[0-9]+$ ]]; then
    echo "Error: <nreps> must be a nonnegative integer, got: $NREPS"
    exit 1
fi

if [ "$NREPS" -lt 1 ]; then
    echo "Error: <nreps> must be >= 1, got: $NREPS"
    exit 1
fi

ARRAY_SPEC="0-$((NREPS-1))"
echo "Config path: $CONFIG_PATH"
echo "Replications (NREPS): $NREPS"
echo "Submitting array: $ARRAY_SPEC"


# Submit the array. Each task will run train_cpu.sh once.
# NOTE: train_cpu.sh contains the #SBATCH directives (partition, mem, etc.)
# NOTE: Output logs should be handled by train_cpu.sh 
sbatch --array="$ARRAY_SPEC" train_cpu.sh "$CONFIG_PATH"
