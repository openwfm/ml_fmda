# Set of functions to manipulate CYGNSS soil moisture dat data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Target is L3 9km subdaily. This module turns that into 
# data for training

import h5py
import pandas as pd
from datetime import datetime
import numpy as np
import os
import os.path as osp
import sys
from dateutil.relativedelta import relativedelta
import earthaccess
import xarray as xr

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
cygnss_meta = read_yml(osp.join(CONFIG_DIR, "variable_metadata", "cygnss_metadata.yaml"))
project_paths = read_yml(osp.join(CONFIG_DIR, "paths.yaml"))

# Module Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def retrieve_cygnss_results(t0, t1, product = cygnss_meta["product"], resolution="9km"):
    """
    Retrieve CYGNSS granule metadata for a specified time range and resolution from earthaccess.

    Parameters
    ----------
    t0 : datetime.datetime
        Start of the time window for the query.
    t1 : datetime.datetime
        End of the time window for the query.
    product : str, optional
        CYGNSS product short name to query. Defaults to the value in `cygnss_meta["product"]`.
    resolution : str, optional
        Spatial resolution identifier to filter granules (e.g., "9km" or "36km").
        Defaults to "9km".

    Returns
    -------
    list
        A list of granule metadata dictionaries returned by `earthaccess.search_data`,
        filtered to include only granules matching the requested resolution.
    """    
    
    # Login
    if not osp.exists(osp.expanduser("~/.netrc")):
        print(f"Auth file ~/.netrc does not exist, create with user and pass")
        sys.exit(1)
    earthaccess.login()
    
    print(f"Retrieving CYGNSS product: {product}")
    print(f"    Start time: {t0}")
    print(f"    End time: {t1}")
    results = earthaccess.search_data(
        short_name=product,
        temporal=(t0.strftime("%Y%m%d"), t1.strftime("%Y%m%d")),
        bounding_box=(-135.0, -38.15, 164.0, 38.15),
        cloud_hosted=True
    )

    filenames = [r["umm"]["GranuleUR"] for r in results]
    keep = [resolution in fn for fn in filenames]

    return [res for res, k in zip(results, keep) if k]

if __name__ == '__main__':

    # Check login
    if not osp.exists(osp.expanduser("~/.netrc")):
        print(f"Auth file ~/.netrc does not exist, create with user and pass")
        sys.exit(1)    

    # Earthdata credentials
    auth = earthaccess.login()
    session = earthaccess.get_requests_https_session()

    # Test
    t0 = str2time("2024-01-01T00:00:00Z")
    t1 = str2time("2024-01-03T23:00:00Z")
    

    # Results links
    results = retrieve_cygnss_results(t0, t1)
    files = earthaccess.open(results)
    ds = xr.open_mfdataset(files, combine="by_coords", data_vars="all")
    ds = ds[["SM_daily", "SIGMA_daily"]]
    breakpoint()


