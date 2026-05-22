# Set of functions and executable to retrieve and manipulate SMAP Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# As of Jan 7 2025, building code for L4 SMAP data version 8
# earthaccess python package used to retrieve data granules
# Goal is to get daily SMAP (9km res) on the HRRR grid (3km res)
# Executable module is get_smap_data.py, uses ~/.netrc config file, see documentation to set up

import pandas as pd
import earthaccess
import h5py
import herbie
from herbie import FastHerbie
from datetime import datetime
import numpy as np
import os
import os.path as osp
import sys
import xarray as xr

# Set up project paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CURRENT_DIR = osp.dirname(osp.normpath(osp.abspath(__file__)))
PROJECT_ROOT = osp.dirname(osp.dirname(osp.normpath(CURRENT_DIR)))
sys.path.append(osp.join(PROJECT_ROOT, "src"))
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import read_yml, Dict

# Read SMAP Metadata
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
smap_meta = read_yml(osp.join(CONFIG_DIR, "variable_metadata", "smap_metadata.yaml"))
project_paths = read_yml(osp.join(CONFIG_DIR, "paths.yaml"))

def get_granules(start, end, config, verbose=True):
    """
    Search EarthAccess for SMAP granules over a given time range.

    Parameters
    ----------
    start : str or datetime
        Start of the temporal search window.
    end : str or datetime
        End of the temporal search window.
    config : dict
        Dictionary containing at least 'product' and 'version'.
    verbose : bool, optional
        If True, print search diagnostics.

    Returns
    -------
    list
        List of EarthAccess granule metadata objects.
    """
    if verbose:
        print(f"Retrieving SMAP data from EarthAccess")
        print(f"    Start Time: {start}")
        print(f"    End Time: {end}")
        print(f"    Product: {config['product']}")
        print(f"    Version: {config['version']}")

    time_range = (start, end)
    granules = earthaccess.search_data(
        short_name=config["product"],
        version=config["version"],
        temporal=time_range
    )
    if verbose:
        print(f"Found {len(granules)} granules")    
    return granules

def get_smap_spatial(fname, verbose=True):
    """
    Parameters
    ----------
    fname : str
        SMAP granule file returned by earthaccess, eg via get_granules function in this module.
        
    Returns
    -------
    proj_info : dict
        Projection and grid metadata extracted from the SMAP granule.
    x : ndarray
        X-coordinate values for the EASE2 grid.
    y : ndarray
        Y-coordinate values for the EASE2 grid.    
    """
    
    proj_info = {}
    coords = {}
    with h5py.File(fname, "r") as f:
        dset = f["EASE2_global_projection"]
        for key, val in dset.attrs.items():
            if isinstance(val, bytes):
                val = val.decode()
            proj_info[key] = val
        coords["x"] = f["x"][:]
        coords["y"] = f["y"][:]        

    return proj_info, coords




if __name__ == '__main__':

    print("Imports successful, no executable code")
