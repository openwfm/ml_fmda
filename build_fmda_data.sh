#!/bin/bash


#SBATCH --job-name=fmda_data
#SBATCH --partition=math-alderaan
#SBATCH --output=logs/fmda_data_%j.out
#SBATCH --ntasks=1
#SBATCH --mem=16G

# Control script to build datasets for fuel moisture models
# Given time range and spatial domain,
# get all FMC from RAWS and join HRRR weather data from same period
# save to target directory
# NOTE: expect UTC times, see example
# NOTE: intended to use with GACC bounding boxes, see rtma_cycler in wrfxpy for coords, but it should work with any bbox
# NOTE: executed python script organizes output by days. Minimum saved data is one full day, so it will pad out if only an hour is requested

if [ "$#" -ne 4 ]; then
    echo "Error: Expected exactly 4 arguments, but got $#."
    echo "Usage: $0 <start_time_utc> <end_time_utc> <bbox> <target_directory>"
    echo "Example: $0 '2023-01-01T00:00:00Z' '2023-01-01T02:00:00Z' '[42,-124.6,49,-116.4]' data/nw_fmda"
    exit 1
fi

START_TIME="$1"
END_TIME="$2"
BBOX="$3"
DIR="$4"

echo "Building data for FMDA. Run configuration:"
echo "Start Time: $START_TIME"
echo "End Time: $END_TIME"
echo "Bounding Box: $BBOX"
echo "Target Data Directory: $DIR"


# Set up environment
source ~/.bashrc
conda activate ml_fmda_data

export PYTHONUNBUFFERED=1
python -u src/ingest/get_fmda_data.py "$START_TIME" "$END_TIME" "$BBOX" "$DIR"


