#!/bin/bash
#SBATCH --job-name=feval
#SBATCH --output=logs/forecast_eval_%j.out
#SBATCH --ntasks=2
#SBATCH --partition=math-alderaan
#SBATCH --mem=64G

# Shell script to combine model outputs and calculate errors for each replication in input model directory
# Credit to user lorellis for structure of the code
# Re-grab the variable that was passed in the run_rnn_hyperparam_model.sh script


MODEL_DIRECTORY="$1"

# Set up environment
source ~/.bashrc
conda activate ml_fmda_models

echo python src/forecast_eval.py $MODEL_DIRECTORY
python src/forecast_eval.py $MODEL_DIRECTORY


