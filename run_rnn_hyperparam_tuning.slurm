#!/bin/bash
# Jonathon Hirschi, Spring 2025
#SBATCH --job-name=rnn_hyperparam_tuning
#SBATCH --partition=math-alderaan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=168:00:00
#SBATCH --output=logs/log_%x_%j.out

# Define arguments
DATA_DIR=$1
OUTPUT_DIR=$2

# Setup conda env
source ~/.bashrc
conda activate ml_fmda_models

# Call the shell script with arguments
echo "Running job: $SLURM_JOB_NAME"
echo "Calling run_rnn_hyperparam_tuning.sh from directory: $(pwd)"
./run_rnn_hyperparam_tuning.sh "$DATA_DIR" "$OUTPUT_DIR"

