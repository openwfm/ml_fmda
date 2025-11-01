#!/bin/bash


#SBATCH --job-name=fmda_data
#SBATCH --partition=math-alderaan
#SBATCH --output=logs/fmda_data_%j.out
#SBATCH --ntasks=2
#SBATCH --mem=16G

# Control script to build datasets for fuel moisture models
# Given time range and spatial domain,
# get all FMC from RAWS and join HRRR weather data from same period
# save to target directory
# NOTE: expect UTC times, see example
# NOTE: intended to use with GACC bounding boxes, see rtma_cycler in wrfxpy for coords, but it should work with any bbox
# NOTE: executed python script organizes output by days. Minimum saved data is one full day, so it will pad out if only an hour is requested

if [ "$#" -ne 1 ]; then
    echo "Error: Expected exactly 1 arguments, but got $#."
    echo "Usage: $0 <config_file>"
    echo "Example: $0 etc/forecast_analysis_TEST.yaml"
    exit 1
fi

CONFIG_FILE="$1"


# Set up environment
source ~/.bashrc
conda activate ml_fmda_data

export PYTHONUNBUFFERED=1
python -u src/ingest/get_fmda_data.py "$CONFIG_FILE"


