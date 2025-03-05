#!/bin/bash
#SBATCH --job-name=tune_model
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
if [ "$#" -ne 3 ]; then
    echo "Error: Expected exactly 2 arguments, but got $#."
    echo "Usage: $0 <model_directory> <data_directory> <config_file>"
    exit 1
fi
MODEL_DIRECTORY="$1"
DATA_DIRECTORY="$2"
CONFIG_FILE="$3"

# Set up environment
source ~/.bashrc
conda activate ml_fmda_models

# Run setup, specify <model_directory> and <data_directory>
echo "Running Hyperparam Tuning step 1, model architecture"
mkdir -p "$MODEL_DIRECTORY"
mkdir -p "$MODEL_DIRECTORY/model_errors"
python src/rnn_hyperparam_setup.py "$MODEL_DIRECTORY" "$DATA_DIRECTORY" "$CONFIG_FILE"

# Run Model Architecture tuning
# Extract number of models from previous step to setup array
# Create log directory and send outputs there
# Extract the jobid to use with waiting for the next things to run
mkdir -p "$MODEL_DIRECTORY/logs"
N_MODELS=$(wc -l < "$MODEL_DIRECTORY/model_grid.txt")
job1_id=$(sbatch --array=1-$N_MODELS --output="$MODEL_DIRECTORY/logs/model_%j_%a.out" run_rnn_hyperparam_model.sh "$MODEL_DIRECTORY" |  awk '{print $NF}')



# Run Part 2 of hyperparam tuning for optimization params
# Run shell file that evaluates minimum error model and runs optimization param grid
# Wait for previous job to finish
echo "Running Hyperparam Tuning step 2, optmization params"
sbatch --dependency=afterok:$job1_id rnn_hyperparam_controller2.sh $MODEL_DIRECTORY



