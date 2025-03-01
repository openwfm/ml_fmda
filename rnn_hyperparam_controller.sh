#!/bin/bash
#SBATCH --job-name=tune_model
#SBATCH --output=logs/rnn_hyperparam_%j.out
#SBATCH --ntasks=1
#SBATCH --partition=math-alderaan-short



# Script used to run the hyperparam tuning process. 
# Steps: 
# 1. Run rnn_hyperparam_setup.py to create a directory and test files 
#     for a model grid and optimization grid
# 2. Run run_rnn_hyperparam_model.sh with sbatch with array argument corresponding to
#     the number of model configurations created by step 1


# Setup input arg, directory where model and outputs live
if [ "$#" -ne 2 ]; then
    echo "Error: Expected exactly 2 arguments, but got $#."
    echo "Usage: $0 <model_directory> <data_directory>"
    exit 1
fi
MODEL_DIRECTORY="$1"
DATA_DIRECTORY="$2"

# Set up environment
source ~/.bashrc
conda activate ml_fmda_models

# Run setup, specify <model_directory> and <data_directory>
mkdir -p "$MODEL_DIRECTORY"
mkdir -p "$MODEL_DIRECTORY/model_errors"
python src/rnn_hyperparam_setup.py "$MODEL_DIRECTORY" "$DATA_DIRECTORY"

# Run Model Architecture tuning
# Extract number of models from previous step to setup array
# Create log directory and send outputs there
mkdir -p "$MODEL_DIRECTORY/logs"
N_MODELS=$(wc -l < "$MODEL_DIRECTORY/model_grid.txt")
sbatch --array=1-$N_MODELS --output="$MODEL_DIRECTORY/logs/model_%a.out" run_rnn_hyperparam_model.sh "$MODEL_DIRECTORY"




# TODO: add code from controller 2 below, while waiting for sbatch above to finish

# Analyze Model Error, extract minimum error model
# python src/rnn_hyperparam_eval.py "$MODEL_DIRECTORY"



# Run Optimization parameter tuning
# Extract number of opt configurations from setup
#mkdir -p "$MODEL_DIRECTORY/opt_errors"
#N_OPT=$(wc -l < "$MODEL_DIRECTORY/opt_grid.txt")
# sbatch --array=1-$N_OPT --output="$MODEL_DIRECTORY/logs/opt_%a.out" run_rnn_hyperparam_opt.sh "$MODEL_DIRECTORY" 

