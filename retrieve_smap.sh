#!/bin/bash


#SBATCH --job-name=smap
#SBATCH --partition=math-alderaan
#SBATCH --output=logs/smap_%j.out
#SBATCH --ntasks=2
#SBATCH --mem=32G

# Control script to retrieve, format, and stash HRRR data, used to stash HRRR data to speed up building training and forecast data
# Given time range, save to stash directory
# Stash directory set in global paths configuration file etc/paths.yaml
# NOTE: expect UTC times, see example
# NOTE: intended to use with GACC bounding boxes, see rtma_cycler in wrfxpy for coords, but it should work with any bbox
# NOTE: executed python script organizes output by days. Minimum saved data is one full day, so it will pad out if only an hour is requested

if [ "$#" -ne 2 ]; then
    echo "Error: Expected exactly 2 arguments, but got $#."
    echo "Usage: $0 <start_time_utc> <end_time_utc>"
    echo "Example: $0 '2023-01-01' '2023-01-02'"
    exit 1
fi

START_TIME="$1"
END_TIME="$2"

echo "Retrieving SMAP data from earthdata:"
echo "Start Time: $START_TIME"
echo "End Time: $END_TIME"


# Set up environment
source ~/.bashrc

export PYTHONUNBUFFERED=1
python -u src/ingest/get_smap_data.py "$START_TIME" "$END_TIME" 


