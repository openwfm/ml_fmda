#!/bin/bash
#SBATCH --job-name=fsetup
#SBATCH --output=logs/forecast_analysis_%j.out
#SBATCH --ntasks=2
#SBATCH --partition=math-alderaan
#SBATCH --mem=32G

# Script to run forecast analysis for RNN and baseline models
# Steps
# 1. Run forecast_analysis_setup.py to create directory and build needed data
# 2. Run forecast_analysis.sh with sbatch with array argument corresponding to
#       the number of forecast periods created in step 1



# Setup input arg, directory where model and outputs live
if [ "$#" -ne 2 ]; then
    echo "Error: Expected exactly 2 arguments, but got $#."
    echo "Usage: $0 <forecast_directory> <config_path>"
    exit 1
fi
FORECAST_DIRECTORY="$1"
CONFIG_PATH="$2"

# Set up environment
source ~/.bashrc
conda activate ml_fmda_models

# Run setup, specify <forecast_directory> and <data_directory>
mkdir -p "$FORECAST_DIRECTORY"
mkdir -p "$FORECAST_DIRECTORY/forecast_outputs"
mkdir -p "$FORECAST_DIRECTORY/logs"
python src/forecast_analysis_setup.py "$FORECAST_DIRECTORY" "$CONFIG_PATH"


# Run train/test for each forecast replication
# Create slurm job array for each forecast replication
N_REPS=$(jq '.nreps' "$FORECAST_DIRECTORY/analysis_info.json")
echo "Creating slurm array jobs for $N_REPS replications with command:"
echo "sbatch --array=1-$N_REPS --mem=24G --output=\"$FORECAST_DIRECTORY/logs/frep_%j_%a.out\" run_forecast_analysis.sh \"$FORECAST_DIRECTORY\""
job_id=$(sbatch --parsable --array=1-$N_REPS --mem=32G --output="$FORECAST_DIRECTORY/logs/frep_%j_%a.out" run_forecast_analysis.sh "$FORECAST_DIRECTORY")

# Wait for job to finish and run eval
# NOT working as of June17, get error: sbatch: error: Batch job submission failed: No partition specified or system default partition
echo "Evaluating Forecast Accuracy after job $job_id"
sbatch --partition=math-alderaan-short --mem=64G  --dependency=afterok:$job_id forecast_eval.sh $MODEL_DIRECTORY

