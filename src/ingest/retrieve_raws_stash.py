# Set of functions and executable to retrieve and manipulate RAWS data from a stash
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Stash created and maintained by Angel Farguell and broader OpenWFM community. Just ask for access

import sys
import synoptic
import numpy as np
import pandas as pd
import pickle
import json
import os.path as osp
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
from pathlib import Path

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
from utils import Dict, str2time, read_yml, read_pkl, time_range
import ingest.retrieve_raws_api as rr

# Read RAWS Metadata
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
raws_meta = read_yml(osp.join(CONFIG_DIR, "variable_metadata", "raws_metadata.yaml"))

# Update stash path. We do this here so it works if module called from different locations
raws_meta.update({'raws_stash_path': osp.join(PROJECT_ROOT, raws_meta['raws_stash_path'])})


def get_file_paths(times):
    """
    Get file paths for RAWS stash from given start and end dates.

    Arguments
        times: 1d numpy array of datetime
    """  

    # Create list of file paths based on needed hours
    paths = [
        osp.join(
            raws_meta["raws_stash_path"],        
            str(time.year),     
            time.strftime('%j'),    #  Julian day of year, 001-366
            f"{str(time.year)}{time.strftime('%j')}{time.strftime('%H')}.pkl" )# Join with hour of day, 00-23
        for time in times
    ]
    
    return paths


def build_raws_dict(config, rename=True, verbose = True):
    """
    Wrapper function that applies the module functions. Given config dictionary, it returns a formatted dictionary of RAWS data
    """

    # Extract config info
    bbox = config.bbox
    start = config.start_time
    end = config.end_time
    print(f"Start Date of RAWS retrieval: {start}")
    print(f"End Date of retrieval: {end}")
    print(f"Spatial Domain: {bbox}")  


    # Get station metadata within bbox and time period
    bbox_reordered = [bbox[1], bbox[0], bbox[3], bbox[2]] # Synoptic uses different bbox order
    start_dt = str2time(start)
    end_dt = str2time(end)
    sts = rr.get_stations(bbox_reordered)

    # Based on time arguments, get list of needed file paths in stash
    # Offset by 1 hr for data retrieval, so interpolation has endpoints
    times = time_range(start_dt, end_dt, start_offset=True)
    paths = get_file_paths(times)

    # Create return dictionary with static info and other metadata
    raws_dict = {
        st: {
            "RAWS": [],
            "units": {"fm": "%", "elev": "m"},
            "loc": rr.get_static(sts, st),
            "misc": "Data collected from RAWS stash."
            } 
        for st in sts["stid"]}
    
    
    # Loop through file paths and extract info from need STID
    for path in paths:
        dat = read_pkl(path)
        for st in sts["stid"]:
            # Filter the data for the current station and append
            filtered = dat[dat['STID'] == st]
            if not filtered.empty:
                raws_dict[st]["RAWS"].append(filtered)

    # Combine the lists of DataFrames for each station into a single DataFrame, rename, and interpolate
    for st in raws_dict:
        if raws_dict[st]["RAWS"]:  # Check if the list is not empty
            raws_dict[st]["RAWS"] = pd.concat(raws_dict[st]["RAWS"], ignore_index=True)
            # Add a few static vars
            raws_dict[st]["RAWS"]["lat"] = raws_dict[st]["loc"]["lat"]
            raws_dict[st]["RAWS"]["lon"] = raws_dict[st]["loc"]["lon"]
            raws_dict[st]["RAWS"]["elev"] = raws_dict[st]["loc"]["elev"]      
        else:
            raws_dict[st]["RAWS"] = pd.DataFrame()  # Set an empty DataFrame if no data was found
        if rename:
            raws_dict[st]["RAWS"].rename(columns=raws_meta["rename_stash"], inplace=True)

    # Remove Stations with missing data
    for st in list(raws_dict.keys()):
        if raws_dict[st]["RAWS"].shape[0] == 0:
            print(f"No data found for station {st}, removing")
            raws_dict.pop(st)
    print(f"Retrieved data for {len(raws_dict.keys())} stations")
    
    # Interpolate
    # No start time offset here 
    # Hard coded static and time columns
    times = time_range(start_dt, end_dt, start_offset=False)
    for st in raws_dict:
        d = rr.time_intp_df(raws_dict[st]["RAWS"], times, static_cols = ["stid", "lat", "lon", "elev"], time_cols = ["fm"])
        d = pd.DataFrame(d, columns = d.columns)
        raws_dict[st]["RAWS"] = d

    return raws_dict



# Executable Code 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':

    # Handle arguments
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ## expects a config file with times and bbox, option argument saves output
    ## Config bbox format should match format in wrfxpy rtma_cycler: [latmin, lonmin, latmax, lonmax]
    if len(sys.argv) not in {2, 3}: 
        print(f"Invalid arguments. {len(sys.argv)} was given but 2 or 3 expected")
        print(('Usage: %s <config_file> <optional_output_file>' % sys.argv[0]))
        print("Example: python src/ingest/retrieve_raws_stash.py etc/training_data_config.json data/raws.pkl")
        sys.exit(-1)

    
    # Handle config
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    conf_file = sys.argv[1]
    with open(conf_file, "r") as json_file:
        config = json.load(json_file)   
        config = Dict(config)
    print(json.dumps(config, indent=4))

    bbox = config.bbox
    start = config.start_time
    end = config.end_time
    print(f"Start Date of RAWS retrieval: {start}")
    print(f"End Date of retrieval: {end}")
    print(f"Spatial Domain: {bbox}")


    # Build Data Dictionary
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    raws_dict = build_raws_dict(config)
      

    # Write output if path provided
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    

    if len(sys.argv) == 3:
        output_path = sys.argv[2]
        print(f"Writing output to {output_path}")
        with open(output_path, 'wb') as file:
            pickle.dump(raws_dict, file)
