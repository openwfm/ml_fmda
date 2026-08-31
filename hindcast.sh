#!/bin/bash

#SBATCH --job-name=hcast
#SBATCH --output=logs/hindcast_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --partition=math-alderaan
#SBATCH --mem=128G

# Process runs forecast mode on a historical period

if [ "$#" -ne 1 ]; then
    echo "Error: Expected exactly 1 arguments, but got $#."
    echo "Usage: $0 <config_path>"
    echo "Example: ./hindcast.sh etc/forecast_TEST.yaml"
    exit 1
fi

CONFIG_PATH="$1"

echo "Config path: $CONFIG_PATH"

# Set up environment
eval "$(conda shell.bash hook)"
echo "Activating conda env: ml_gpu2"
conda activate ml_gpu2

echo "python src/hindcast.py "$CONFIG_PATH""
python src/hindcast.py "$CONFIG_PATH"


