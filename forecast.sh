#!/bin/bash


#SBATCH --job-name=train
#SBATCH --partition=math-alderaan-gpu-cuda12
#SBATCH --output=logs/train_%j.out
#SBATCH --ntasks=2
#SBATCH --mem=32G

# NOTE: different scripts than forecast analysis which is used to estimate forecast error with spatiotemporal CV
# this is intended to deploy a model operationally

if [ "$#" -ne 3 ]; then
    echo "Error: Expected exactly 2 arguments, but got $#."
    echo "Usage: $0 <model_directory> <config_path> <output_path>"
    exit 1
fi

MODEL_DIRECTORY="$1"
CONFIG_PATH="$2"
OUTPUT_PATH="$3"

echo "Model directory: $MODEL_DIRECTORY"
echo "Config path: $CONFIG_PATH"
echo "Output path: $OUTPUT_PATH"

# Set up environment
source ~/.bashrc
conda activate ml_gpu2
export PYTHONUNBUFFERED=1
python src/forecast.py "$MODEL_DIRECTORY" "$CONFIG_PATH" "$OUTPUT_PATH"
