#!/bin/bash
#SBATCH --job-name=clim
#SBATCH --output=logs/climatology_%j.out
#SBATCH --ntasks=16
#SBATCH --partition=math-alderaan-short
#SBATCH --mem=32G


# Setup input arg, directory where model and outputs live
if [ "$#" -ne 1 ]; then
    echo "Error: Expected exactly 1 arguments, but got $#."
    echo "Usage: $0 <config_path>"
    exit 1
fi
CONFIG_PATH="$1"

# Set up environment, need environment with synopticpy
source ~/.bashrc
conda activate ml_fmda_data

# Run setup, specify <forecast_directory> and <data_directory>
python src/run_climatology.py $CONFIG_PATH

