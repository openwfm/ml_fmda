#!/bin/bash


#SBATCH --job-name=train
#SBATCH --partition=math-alderaan-gpu-cuda12
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/train_%j.out
#SBATCH --ntasks=4
#SBATCH --mem=64G

# Credit to user lorellis for aspects of this code structure

if [ "$#" -ne 1 ]; then
    echo "Error: Expected exactly 1 arguments, but got $#."
    echo "Usage: $0 <config_path>"
    exit 1
fi

CONFIG_PATH="$1"

echo "Model directory: $MODEL_DIRECTORY"
echo "Config path: $CONFIG_PATH"

# Set up environment
eval "$(conda shell.bash hook)"
conda activate gpu_TEST

python src/train.py "$CONFIG_PATH"
