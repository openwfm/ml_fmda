#!/bin/bash
#SBATCH --job-name=rnn_hyperparam
#SBATCH --output=logs/rnn_hyperparam_%a.out
#SBATCH --ntasks=1
#SBATCH --partition=math-alderaan-short



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
column -s, -t < "$MODEL_DIRECTORY/model_err_df.csv" | less -S


# Run Optimization parameter tuning
# Extract number of opt configurations from setup
mkdir -p "$MODEL_DIRECTORY/opt_errors"
N_OPT=$(wc -l < "$MODEL_DIRECTORY/opt_grid.txt")
sbatch --array=1-$N_OPT --output="$MODEL_DIRECTORY/logs/opt_%a.out" run_rnn_hyperparam_opt.sh "$MODEL_DIRECTORY" 

