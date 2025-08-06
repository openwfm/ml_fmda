#!/bin/bash

#SBATCH --mem=8G

# Shell script to run a single iteration of model train and predict
# Used by shell file that assigns a CPU for each array task to allow for parallelization of hyperparam tuning
# Credit to user lorellis for structure of the code
# Re-grab the variable that was passed in the run_rnn_hyperparam_model.sh script



SLURM_TASK_ARRAY_ID=$1
MODEL_DIRECTORY=$2

# Set up environment
source ~/.bashrc
conda activate ml_fmda_models

# Pass array number to python script and run

python src/rnn_hyperparam_model.py $SLURM_ARRAY_TASK_ID $MODEL_DIRECTORY
