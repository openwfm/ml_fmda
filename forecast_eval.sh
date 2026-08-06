#!/bin/bash
#SBATCH --job-name=feval
#SBATCH --output=logs/forecast_eval_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=math-alderaan
#SBATCH --mem=128G

# Shell script to combine model outputs and calculate errors for each replication in input model directory
# Credit to user lorellis for structure of the code
# Re-grab the variable that was passed in the run_rnn_hyperparam_model.sh script

if [ "$#" -ne 1 ]; then
    echo "Error: Expected exactly 1 arguments, but got $#."
    echo "Usage: $0 <model_directory>"
    exit 1
fi

MODEL_DIRECTORY="$1"

# Set up environment
eval "$(conda shell.bash hook)"
echo "Activating conda env: ml_fmda_models"
conda activate ml_fmda_models

echo python src/forecast_eval.py $MODEL_DIRECTORY
echo "Running executable Python script: python src/forecast_eval.py \"$MODEL_DIRECTORY\""
python src/forecast_eval.py $MODEL_DIRECTORY


