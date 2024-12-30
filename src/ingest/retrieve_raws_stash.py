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
from utils import str2time, read_yml
# from utils import read_yml, Dict, time_intp, str2time
# from data_funcs import rename_dict

# Read RAWS Metadata
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
raws_meta = read_yml(osp.join(CONFIG_DIR, "variable_metadata", "raws_metadata.yaml"))

# Update stash path. We do this here so it works if module called from different locations
raws_meta.update({'raws_stash_path': osp.join(PROJECT_ROOT, raws_meta['raws_stash_path'])})


def get_file_paths(start, end):
    """
    Get file paths for RAWS stash from given start and end dates.
    """  
    
    # Get array of hourly times
    times = pd.date_range(start-relativedelta(hours=1), end, freq="h")
    times = times.to_pydatetime()

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
