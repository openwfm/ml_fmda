# Executable process to retrieve HRRR data, format, and save to stash
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# User inputs a start and end time to this process
# HRRR metadata files in etc/ extract information needed for FMDA


import pandas as pd
import herbie
from herbie import FastHerbie
from datetime import datetime
import numpy as np
import os
import os.path as osp
import sys
from dateutil.relativedelta import relativedelta

# Set up project paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CURRENT_DIR = osp.dirname(osp.normpath(osp.abspath(__file__)))
PROJECT_ROOT = osp.dirname(osp.dirname(osp.normpath(CURRENT_DIR)))
sys.path.append(osp.join(PROJECT_ROOT, "src"))
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import time_range, str2time, Dict, read_yml
import ingest.HRRR as ih

project_paths = Dict(read_yml(osp.join(CONFIG_DIR, "paths.yaml")))
hrrr_meta = ih.hrrr_meta

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Invalid arguments. {len(sys.argv)} was given but 2 expected")
        print(('Usage: %s <esmf_from_utc> <esmf_to_utc>' % sys.argv[0]))
        print("Example: python src/ingest/get_hrrr_data.py '2023-06-01T00:00:00Z' '2023-06-02T00:00:00Z'")
        print("Times should match format: '2023-06-01T00:00:00Z'")
        sys.exit(-1)

    start = str2time(sys.argv[1])
    end = str2time(sys.argv[2])
    hrrr_stash_path = project_paths["hrrr_stash_path"]
    
    print(f"Retrieving HRRR data for FMDA stash")
    print(f"    Start Time: {start}")
    print(f"    End Time: {end}")
    print(f"    Stash Path: {hrrr_stash_path}")

    # Break time range up into days and loop 
    # Since code retrieves whole days by design, drop last day as it is redundant with the previous
    days = time_range(start, end, freq="1d")
    if len(days) > 1:
        days = days[:-1]
    print("~"*75)
    for t in days:
        print("~"*50)
        print(f"Processing data in day {t}")
        ym = t.strftime("%Y%m")
        d = t.strftime("%d")
        start_t = t.replace(hour=0, minute=0, second=0, microsecond=0) # remove hours to grab full day
        end_t = t.replace(hour=23, minute=0, second=0, microsecond=0) # Add 24 hours to given time    
        hrrr_ds = ih.retrieve_hrrr(start, end, save_to_stash=True, read_to_memory=False)

