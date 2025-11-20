# Set of functions to manipulate SMAP data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SMAP data retrieved with src/ingest/get_smap_data.py, 
# raw L3 9km enhanced data. This module turns that into 
# data for training

import h5py
import pandas as pd
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
from utils import read_yml, Dict, time_intp, str2time, print_dict_summary, rename_dict, time_range


# Read Metadata
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
smap_meta = read_yml(osp.join(CONFIG_DIR, "variable_metadata", "smap_metadata.yaml"))
project_paths = read_yml(osp.join(CONFIG_DIR, "paths.yaml"))
smap_stash_path = project_paths["smap_stash_path"]

# Module Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def get_smap_files(start, end, directory=smap_stash_path):
    """
    Given target times, get a list of files of SMAP data.
    See etc/variable_metadata/smap_metadata.yaml for info on SMAP product used
    """
    start = str2time(start)
    end = str2time(end)
    days = time_range(start, end, freq="1d")
    files=[]
    for di in days:
        yr = di.year
        mdy = di.strftime("%Y%m%d")
        dir_temp = osp.join(directory, str(yr))
        all_files = os.listdir(dir_temp)
        matches = [
            os.path.join(dir_temp, fj)
            for fj in all_files
            if mdy in fj and fj.endswith(".h5")
        ]       
        if len(matches) == 0:
            raise ValueError("No matching SMAP files found for the requested date: {di}.")

        elif len(matches) > 1:
            print(f"Warning: multiple matches found ({len(matches)}). Using the first: {matches[0]}")
            files.append(matches[0])

        else:  # exactly one
            files.append(matches[0])

    return files


def read_smap_file(fname, bbox = None):
    """
    Read h5 file contents, extract needed fields, rename according to namelist in metadata
    If bbox provided, filter with subset_bbox
    """
    nlist = smap_meta["namelist"]
    var_names = [*nlist.keys()]
    var_names_pm = [name + "_pm" for name in var_names]
    def read_3d(grp, vnames=var_names):
        sample = grp[vnames[0]][:]
        ny, nx = sample.shape
        dat = np.zeros((len(vnames), ny, nx), dtype=sample.dtype)
        for i, name in enumerate(vnames):
            dat[i] = grp[name][:]
        return dat

    with h5py.File(fname, "r") as fi:
        # print(list(f.keys()))              
        # meta = f["Metadata"]
        dat_am = read_3d(fi["Soil_Moisture_Retrieval_Data_AM"])        
        dat_pm = read_3d(fi["Soil_Moisture_Retrieval_Data_PM"], vnames=var_names_pm)
    
    if bbox is not None:
        dat_am = subset_smap(dat_am, bbox)
        dat_pm = subset_smap(dat_pm, bbox)
    
    return dat_am, dat_pm


def read_smap_files(files, bbox = None):
    """
    Read file list and concat, if bbox provided, call subset_smap on each
    """
    out = [read_smap_file(f, bbox) for f in files]
    dams, dpms = zip(*out)
    dams, dpms = zip(*out)
    dams, dpms = np.stack(dams, axis=0), np.stack(dpms, axis=0)
    print(f"AM data shape: {dams.shape},    PM data shape: {dpms.shape}")
    print(f"    Number of days requested: {len(files)}")
    print(f"    Number of variables extracted from SMAP: {len(smap_meta['namelist'])}")
    if bbox is not None:
        print(f"    Subsetting to boudning box (w, s, e, n): {bbox}")
    return dams, dpms


def subset_smap(dat, bbox, MISS=-9999.0):
    """
    Spatially Subset by bounding box
    NOTE: as of Nov 18 2025 this isn't working as expected, odd behavior with -9999
    """
    dat[dat == -9999.0] = np.nan
    var_names = list(smap_meta["namelist"].keys())
    west, south, east, north = bbox
    lon_idx = var_names.index("longitude")
    lat_idx = var_names.index("latitude")
    lon = dat[lon_idx]
    lat = dat[lat_idx]
    mask = (
        (lon >= west) & (lon <= east) &
        (lat >= south) & (lat <= north)
    )
    # find bounding rows/cols (tight cropping)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()

    # subset every variable automatically
    dat_sub = dat[:, row_min:row_max+1, col_min:col_max+1]
    print("Original shape:", dat.shape)
    print("Subset shape:", dat_sub.shape)    
    return dat_sub

def check_missing_coords(x_slice, fill=-9999.0):
    """
    x_slice : ndarray of shape (5, H, W)
              order = [lon, lat, sm, sm_err, sm_dca]

    Returns:
        any_problem (bool)
        count_problem (int)
    """
    lon = x_slice[0]
    lat = x_slice[1]
    sm  = x_slice[2]

    valid_sm = sm != fill
    missing_coord = (lon == fill) | (lat == fill)
    problem_mask = valid_sm & missing_coord

    return problem_mask.any(), int(problem_mask.sum())

def fill_timeseries(dat):
    

if __name__ == '__main__':

    # Test
    t0 = "2023-01-01T00:00:00Z"
    t1 = "2023-01-03T23:00:00Z"
    days = time_range(t0, t1, freq="1d")
    files = get_smap_files(t0, t1)
    bbox = (-111, 37, -95, 46)
    dam, dpm = read_smap_files(files, bbox = None)
    breakpoint()
    #dam2, dpm2 = read_smap_files(files, bbox = bbox)
    

    old_names = list(smap_meta["namelist"].keys())
    new_names = list(smap_meta["namelist"].values())
    
    
    
