#!/bin/bash
#SBATCH --job-name=tune_opt
#SBATCH --output=logs/rnn_hyperparam_%j.out
#SBATCH --ntasks=1
#SBATCH --partition=math-alderaan



# Script used to run the hyperparam tuning process. 
# Steps: 
# 1. Run rnn_hyperparam_setup.py to create a directory and test files 
#     for a model grid and optimization grid
# 2. Run run_rnn_hyperparam_model.sh with sbatch with array argument corresponding to
#     the number of model configurations created by step 1


# Setup input arg, directory where model and outputs live
if [ "$#" -ne 1 ]; then
    echo "Error: Expected exactly 1 arguments, but got $#."
    echo "Usage: $0 <model_directory> "
    exit 1
fi
MODEL_DIRECTORY="$1"

# Set up environment
source ~/.bashrc
conda activate ml_fmda_models



# Analyze Model Error, extract minimum error model
python src/rnn_hyperparam_eval.py "$MODEL_DIRECTORY"


# Run Optimization parameter tuning
# Extract number of opt configurations from setup
mkdir -p "$MODEL_DIRECTORY/opt_errors"
N_OPT=$(wc -l < "$MODEL_DIRECTORY/opt_grid.txt")
job2_id=$(sbatch --array=1-$N_OPT --mem=8G --output="$MODEL_DIRECTORY/logs/opt_%j_%a.out" run_rnn_hyperparam_opt.sh "$MODEL_DIRECTORY" |  awk '{print $NF}') 


# Wait for process to finish and run eval again
while squeue -j "$job2_id" &>/dev/null; do
    sleep 120  # Check every 2 minutes
done

python src/rnn_hyperparam_eval.py "$MODEL_DIRECTORY"
