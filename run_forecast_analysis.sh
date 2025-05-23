#!/bin/bash


#SBATCH --job-name=fcast

#SBATCH --partition=math-alderaan

#SBATCH --ntasks=1

#SBATCH --mem=8G

# Credit to user lorellis for aspects of this code structure
# Pass the variable {SLURM_TASK_ARRAY_ID} to the test_slurm_array.sh script and run said script

if [ "$#" -ne 1 ]; then
    echo "Error: Expected exactly 1 argument, but got $#."
    echo "Usage: $0 <model_directory>"
    exit 1
fi

MODEL_DIRECTORY="$1"

echo "Model directory: $MODEL_DIRECTORY"

./forecast_analysis.sh "${SLURM_TASK_ARRAY_ID}" "$MODEL_DIRECTORY"
