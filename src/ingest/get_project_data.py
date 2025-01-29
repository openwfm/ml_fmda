# Executable process to retrieve data for FMC model testing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Given a target forecast start time, 1 year plus 48 hours collected
# the forecast period is 48 hours into the future from that time
# the training period is 1 year prior to that time
# 
# User will input a start and end time to this process, this will define a forecasting testing period
# Based on the input times, we break it into 48 hour periods and get the training and forecasting periods associated with each time
# For associated paper of this project, start and end times will cover all of 2024
# So starting on Jan 1 2024 00:00, we want training data from Jan 1 2023 through this time, and we will forecast 48hrs through Jan 2 2024 23:00
# That would be 1 testing period. Then the target time gets shifted 48 hours into the future 
# This way, models will be tested at their ability to forecast 48 hours into the future, tested on all of 2024

# So this code is intended to retrieve enough data to run that analysis from RAWS, HRRR, and other sources


import pandas as pd
import herbie
from herbie import FastHerbie
from datetime import datetime
import numpy as np
import os
import os.path as osp
import sys
from dateutil.relativedelta import relativedelta
import pickle
import ast

# Set up project paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## We do this so the module can be imported from different locations
CURRENT_DIR = osp.abspath(__file__)
while osp.basename(CURRENT_DIR) != "ml_fmda":
    CURRENT_DIR = osp.dirname(CURRENT_DIR)
PROJECT_ROOT = CURRENT_DIR
CODE_DIR = osp.join(PROJECT_ROOT, "src")
sys.path.append(CODE_DIR)
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import time_range
import ingest.RAWS as rr
import ingest.HRRR as ih


def retrieve_fmda_data(start, end, bbox, raws_source = "stash", save_path = None):
    """
    Retrieve data for FMC models.

    Parameters
    ----------

    
    raws_source : str
        One of "stash" or "api".

    save_filename: str
        Optional filename string, will be saved in the "data" directory relative to PROJECT_ROOT

    Returns
    ----------
    raws_dict : dict
        Nested dictionary with top level key corresponding to a RAWS and subkeys for RAWS, atmospheric data (HRRR), geographic info, etc
    """
    
    # Handle RAWS Source
    if raws_source == "stash":
        raws_stash_path = rr.raws_meta["raws_stash_path"]
        assert osp.exists(raws_stash_path), f"Config raws stash path not found: {raws_stash_path}"
        build_raws_dict = rr.build_raws_dict_stash
    elif raws_source == "api":
        build_raws_dict = rr.build_raws_dict_api
    else:
        raise ValueError(f"Input raws_source: {raws_source} not recognized")

    # Retrieve Data
    raws_dict = build_raws_dict(start, end, bbox)
    hrrr_ds = ih.retrieve_hrrr_train(start, end, bbox)

    # Handle HRRR data
    hrrr_pts = ih.subset_hrrr2raws(hrrr_ds, raws_dict)
    hrrr_pts = ih.rename_ds(hrrr_pts)
    assert np.all(hrrr_pts.point_stid.to_numpy() == np.array([*raws_dict.keys()])), "Not all RAWS STID in raws_dict found in hrrr data"

    # Merge Dictionaries
    for st in raws_dict:
        # Comfirm times match. For HRRR data it should be the date_time which accounts for forecast hour
        raws_timesi = raws_timesi = raws_dict[st]["times"]
        assert np.all(raws_timesi == hrrr_pts.date_time.to_numpy()), "Times in RAWS dict don't match HRRR data date_time"
    
        # Extract dataframe of predictors, save in HRRR subdictionary
        df = hrrr_pts.where(hrrr_pts.point_stid == st, drop=True).to_dataframe()
        df.reset_index('point', drop=True, inplace=True)
        raws_dict[st]["HRRR"] = df

    # Save if a filename specified to data directory relative to PROJECT_ROOT
    if save_path is not None:
        print(f"Saving data to {save_path}")
        with open(save_path, 'wb') as handle:
            pickle.dump(raws_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return raws_dict

def parse_bbox(box_str):
    try:
        # Use ast.literal_eval to safely parse the string representation
        # This will only evaluate literals and avoids security risks associated with eval
        box = ast.literal_eval(box_str)
        # Check if the parsed box is a list and has four elements
        if isinstance(box, list) and len(box) == 4:
            return box
        else:
            raise ValueError("Invalid bounding box format")
    except (SyntaxError, ValueError) as e:
        print("Error parsing bounding box:", e)
        sys.exit(-1)
        return None

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print(f"Invalid arguments. {len(sys.argv)} was given but 4 expected")
        print(('Usage: %s <esmf_from_utc> <esmf_to_utc> <bbox> <output_dir>' % sys.argv[0]))
        print("Example: python src/ingest/get_project_data.py '2024-01-01T00:00:00Z' '2025-01-01T00:00:00Z' '[37,-111,46,-95]' data/rocky_fmda")
        print("bbox format should match rtma_cycler: [latmin, lonmin, latmax, lonmax]")
        print("Times should match format: 2023-06-01T00:00:00Z")
        sys.exit(-1)

    print()
    print("~"*75)
    print("Setting up 48 hour forecast periods...")

    
    start = sys.argv[1]
    end = sys.argv[2]
    bbox = parse_bbox(sys.argv[3])
    output_dir = sys.argv[4]

    if not osp.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.mkdir(output_dir)

    # Define Forecast start times
    forecast_periods = time_range(
        start = start,
        end = end,
        freq = "2d"
    )
    print(f"Number of forecast periods: {len(forecast_periods)}")
    print(f"Earliest forecast start: {forecast_periods.min()}")
    print(f"Latest forecast start: {forecast_periods.max()}")

    # Get earliest and latest times
    # Earliest time will be start of training period for earliest forecast start time
    # Latest time will be end of test forecast period for latest forecast start time
    print()
    print("Defining CV time periods based on earliest and latest forecast times")
    t0 = forecast_periods.min() - relativedelta(hours=8760)
    t1 = forecast_periods.max() + relativedelta(hours=48)
    days = time_range(t0, t1, freq="1d")
    
    # Retrieve Data
    # For earliest and latest days in previous, retrieve one day of data at a time for computational limits
    # Organize files in Month directories and whole days for pkl files
    # Check if data exists and continue if not. Should allow for rerunning on crash or easily adding time periods
    # Note, default behavior of getting complete day of final period, even if not needed for analysis. Just makes code cleaner but will run a little longer
    print()
    print("~"*75)
    for t in days:
        print("~"*50)
        print(f"Processing data in day {t}")
        ym = t.strftime("%Y%m")
        d = t.strftime("%d")
        start_t = t
        end_t = t.replace(hour=23, minute=0, second=0, microsecond=0) # Add 24 hours to start time
        ym_dir = osp.join(output_dir, ym)
        os.makedirs(ym_dir, exist_ok=True)
        filepath = osp.join(ym, f"fmda_{ym}.pkl")
        if not osp.exists(filepath):
            print(f"Retrieving FMDA data from {start_t} to {end_t}")
            retrieve_fmda_data(start_t, end_t, bbox, save_path = filepath)
        else:
            print(f"Data for day {t} already exists in {output_dir}/{ym_dir}, skipping to next period")

    
    
