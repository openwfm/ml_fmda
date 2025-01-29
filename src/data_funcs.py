# Set of Functions to process and format fuel moisture model inputs
# These functions are specific to the particulars of the input data, and may not be generally applicable
# Generally applicable functions should be in utils.py
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from datetime import datetime
import numpy as np
import os.path as osp
import sys
import pickle
import pandas as pd
import random
import copy

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
from utils import read_pkl, time_range
import reproducibility
import ingest.RAWS as rr
import ingest.HRRR as ih


# Data Retrieval Wrappers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def subdicts_identical(d1, d2, subdict_keys = ["units", "loc", "misc"]):
    """
    Helper function to merge retrieved data dictionaries. Checks that subdicts for metadata are the same
    """
    return all(d1.get(k) == d2.get(k) for k in subdict_keys)


def extend_fmda_dicts(d1, d2, subdict_keys=["RAWS", "HRRR", "times"]):
    assert subdicts_identical(d1, d2), "Metadata subdicts not the same"
    merged_dict = {k: d1[k] for k in ["units", "loc", "misc"]} # copy metadata

    for key in subdict_keys:
        if key in ["RAWS", "HRRR"]:  # DataFrames
            merged_dict[key] = (
                pd.concat([d1[key], d2[key]])
                .drop_duplicates(subset="date_time")
                .sort_values("date_time")
                .reset_index(drop=True)
            )
        elif key == "times":  # NumPy datetime array
            merged_dict[key] = np.unique(np.concatenate([d1[key], d2[key]]))

    return merged_dict

def combine_fmda_files(input_file_paths, verbose=True):
    """
    Read a list of files retrieved with retrieve_fmda_data and combine data at common stations based on time
    """
    # Read all
    dicts = [read_pkl(path) for path in input_file_paths]
    # Initialize combined dictionary as first dict, then loop over others and merge
    combined_dict = dicts[0]
    for i in range(1, len(dicts)):
        di = dicts[i]
        for st in di:
            if st not in combined_dict.keys():
                combined_dict[st] = di[st]
            else:
                combined_dict[st] = extend_fmda_dicts(combined_dict[st], di[st])

    return combined_dict


# Cross Validation Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Helper function to filter dataframe on time
def filter_df(df, filter_col, ts):
    return df[df[filter_col].isin(ts)]








