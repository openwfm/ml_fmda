#!/bin/bash


#SBATCH --job-name=train
#SBATCH --partition=math-alderaan-gpu-cuda12
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/train_%j.out
#SBATCH --ntasks=2
#SBATCH --mem=32G

# Credit to user lorellis for aspects of this code structure

if [ "$#" -ne 2 ]; then
    echo "Error: Expected exactly 2 arguments, but got $#."
    echo "Usage: $0 <model_directory> <config_path>"
    exit 1
fi

MODEL_DIRECTORY="$1"
CONFIG_PATH="$2"

echo "Model directory: $MODEL_DIRECTORY"
echo "Config path: $CONFIG_PATH"

# Set up environment
source ~/.bashrc
conda activate gpu_TEST
export PYTHONUNBUFFERED=1
python src/train.py "$MODEL_DIRECTORY" "$CONFIG_PATH"
