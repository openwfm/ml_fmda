#!/bin/bash

# Shell file used to convert landfire data to HRRR projection  
# and extract landfire data at HRRR grid. Intended to join to HRRR data for training/forecasting
# Usage: ./src/setup_lf_elevation.sh /path/to/dir/LF2020_Elev_220_CONUS

# Check if argument is given
if [ -z "$1" ]; then
  echo "Usage: $0 <LF_DIR>"
  exit 1
fi

LF_DIR="$1"
echo "LF_DIR is: $LF_DIR"

# Set up environment
source ~/.bashrc
conda activate ml_fmda_data


# Construct derived paths
LF_PATH="$LF_DIR/Tif/LC20_Elev_220.tif"       # Path to actual TIF file of elevation
HPROJ_PATH="$LF_DIR/hrrr_projection.prj"      # Path to proj file from HRRR grib2, created from gribfile_projection in a herbie object
HREF_PATH="$LF_DIR/hrrr_reference.tif"        # TIF file with HRRR grid
REPROJ_PATH="$LF_DIR/lf_elev_reprojected.tif" # Destination path for LF data reprojected at HRRR grid
OUT_PATH="$LF_DIR/lf_elev_at_hrrr.tif"        # Destination path for elevation at HRRR grid

# Reproject LandFire data using projection info from HRRR
gdalwarp -t_srs "$HPROJ_PATH" -r near "$LF_PATH" "$REPROJ_PATH" 

# Get nearest values from LF to HRRR grid
gdalwarp -r near \
         -te $(gdalinfo -json "$HREF_PATH" | jq '.cornerCoordinates | .lowerLeft[0], .lowerLeft[1], .upperRight[0], .upperRight[1]') \
         -tr $(gdalinfo -json "$HREF_PATH" | jq '.geoTransform | .[1], .[5]') \
         -t_srs "$HPROJ_PATH" \
         "$REPROJ_PATH" \
         "$OUT_PATH"



