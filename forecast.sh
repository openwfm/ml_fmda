#!/bin/bash

#SBATCH --job-name=fcast
#SBATCH --output=logs/forecast_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --partition=math-alderaan
#SBATCH --mem=64G

# NOTE: different scripts than forecast analysis which is used to estimate forecast error with spatiotemporal CV
# this is intended to deploy a model operationally

if [ "$#" -ne 1 ]; then
    echo "Error: Expected exactly 1 arguments, but got $#."
    echo "Usage: $0 <config_path>"
    exit 1
fi

CONFIG_PATH="$1"

echo "Config path: $CONFIG_PATH"

# Set up environment
eval "$(conda shell.bash hook)"
conda activate ml_gpu2

python src/forecast.py "$CONFIG_PATH" 
