#!/bin/bash
#SBATCH --job-name=fcast_setup
#SBATCH --output=logs/forecast_analysis_%j.out
#SBATCH --ntasks=1
#SBATCH --partition=math-alderaan


# Script to run forecast analysis for RNN and baseline models
# Steps
# 1. Run forecast_analysis_setup.py to create directory and build needed data
# 2. Run forecast_analysis.sh with sbatch with array argument corresponding to
#       the number of forecast periods created in step 1



# Setup input arg, directory where model and outputs live
if [ "$#" -ne 2 ]; then
    echo "Error: Expected exactly 2 arguments, but got $#."
    echo "Usage: $0 <forecast_directory> <data_directory>"
    exit 1
fi
FORECAST_DIRECTORY="$1"
DATA_DIRECTORY="$2"

# Set up environment
source ~/.bashrc
conda activate ml_fmda_models

# Run setup, specify <forecast_directory> and <data_directory>
mkdir -p "$FORECAST_DIRECTORY"
mkdir -p "$FORECAST_DIRECTORY/forecast_periods"
mkdir -p "$FORECAST_DIRECTORY/logs"
python src/forecast_analysis_setup.py "$FORECAST_DIRECTORY" "$DATA_DIRECTORY"


# Run train/test for each forecast period
# Create slurm job array for each forecast period
N_PERIOD=$(jq '.forecast_periods' forecasts/fmc_forecast_test/analysis_info.json)
echo $N_PERIOD
# sbatch --array=1-$N_PERIODS --output="$FORECAST_DIRECTORY/logs/fperiod_%a_%j.out" run_forecast_period.sh "$FORECAST_DIRECTORY"


