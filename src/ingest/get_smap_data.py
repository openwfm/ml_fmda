# Executable process to retrieve SMAP data, format, and save to stash
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# User inputs a start and end time to this process
# Retrieval config as of Nov 17 2025
#     Level 3, 9km enhanced
#     Earth data product SPL3SMP_E
#     Getting all of Conus
# Need to have auth file ~/.netrc

import pandas as pd
import numpy as np
import earthaccess
import netCDF4
import h5py
import xarray as xr
import os
import os.path as osp
import sys
import time
import random

# Set up project paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CURRENT_DIR = osp.dirname(osp.normpath(osp.abspath(__file__)))
PROJECT_ROOT = osp.dirname(osp.dirname(osp.normpath(CURRENT_DIR)))
sys.path.append(osp.join(PROJECT_ROOT, "src"))
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import Dict, read_yml

project_paths = Dict(read_yml(osp.join(CONFIG_DIR, "paths.yaml")))

config = Dict(read_yml(osp.join(CONFIG_DIR, "variable_metadata", "smap_metadata.yaml")))

# Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def granules_to_xarray(granules, variables=["sm_surface"]):
    # Filter to only h5 files 
    # .qa files come in for 2026 retrievals
    
    files = [f for f in earthaccess.open(granules)
         if getattr(f, "path", "").endswith(".h5")] 

    datasets = []
    print(f"Data variables requested: {variables}")

    for file in files:
        with h5py.File(file, "r") as f:
            g = f["Geophysical_Data"]

            data_vars = {}
            for var in variables:
                data_vars[var] = (
                    ("time", "y", "x"),
                    g[var][:][None, ...],
                )
            ds = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "time": np.atleast_1d(f["time"][()]),
                    "y": f["y"][:],
                    "x": f["x"][:],
                    "cell_lat": (("y", "x"), f["cell_lat"][:]),
                    "cell_lon": (("y", "x"), f["cell_lon"][:]),
                },
            )
            datasets.append(ds)
    ds = xr.concat(datasets, dim="time").sortby("time")
    # Convert fill values to NaN
    for var in ds.data_vars:
        ds[var] = ds[var].where(ds[var] != -9999)    
    # Fix time, calc from seconds offset
    units = "seconds since 2000-01-01 11:58:55.816"
    ds = ds.assign_coords(
        time=np.array(
            netCDF4.num2date(ds.time.values, units=units),
            dtype="datetime64[ns]",
        )
    )

    # Drop extra min/sec/ns
    ds = ds.assign_coords(
        time=ds.time.dt.round("3h")
    )
    return ds


#smap_config = {
#    'product': "SPL3SMP_E",   # daily, 9 km enhanced soil moisture
#    'bbox': (-125, 24.94, -66.93, 49.6) # CONUS, [south_lat, west_lon, north_lat, east_lon]
#}
bbox = (-125, 24.94, -66.93, 49.6) # CONUS, [south_lat, west_lon, north_lat, east_lon]

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Invalid arguments. {len(sys.argv)} was given but 2 expected")
        print(('Usage: %s <esmf_from_utc> <esmf_to_utc>' % sys.argv[0]))
        print("Example: python src/ingest/get_smap_data.py '2023-01-01' '2023-01-02'")
        print("Times should match format: YYYY-MM-DD")
        sys.exit(1)

    # User args
    start = sys.argv[1]
    end = sys.argv[2]
    
    # Config args
    product = config.product
    if product == "SPL4SMGP": level = "4"
    elif product == "SPL3SMP_E": level = "3"
    smap_stash_path = osp.join(project_paths["smap_stash_path"], f"L{level}")
    variables_list = config.get("variables", ["sm_surface"])
    
    if not osp.exists(smap_stash_path):
        print(f"Stash directory doesn't exist: {smap_stash_path}")
        print(f"Update path in `etc/paths.yaml` and manually create directory")
        sys.exit(1)
    
    print(f"Retrieving SMAP L4 surface soil moisture data for FMDA stash")
    print(f"    Start Time: {start}")
    print(f"    End Time: {end}")
    print(f"    Stash Path: {smap_stash_path}")

    # Login
    if not osp.exists(osp.expanduser("~/.netrc")):
        print(f"Auth file ~/.netrc does not exist, create with user and pass")
        sys.exit(1)
    earthaccess.login()

    # Break time range into day
    days = pd.date_range(start, end, freq="D")
    for day in days:
        print("~"*50)
        print(f"Processing day: {day}")
        year = str(pd.Timestamp(day).year)
        filename = f"smap_L4_{day.strftime('%Y%m%d')}.nc"
        if osp.exists(osp.join(smap_stash_path, year, filename)):
            print(f"SMAP data for current day exits at {osp.join(smap_stash_path, filename)}")
            continue        
        delay = random.uniform(1, 3)
        print(f"Waiting {delay:.1f} s before requesting the next day's data...")
        time.sleep(delay)        
        granules = earthaccess.search_data(
            short_name=config["product"],
            temporal=(
                day.strftime("%Y-%m-%dT00:00:00Z"),
                (day + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"),
            )
        )
        print(f"Found {len(granules)} granules")
        ds = granules_to_xarray(granules, variables = variables_list)
        # Save in YYYY subdirectory
        os.makedirs(osp.join(smap_stash_path, year), exist_ok=True)
        print(f"Writing sm_surface data to {osp.join(smap_stash_path, year, filename)}")
        encoding = {
            var: {"zlib": True, "complevel": 4}
            for var in ds.data_vars
        }        
        ds.to_netcdf(
            osp.join(smap_stash_path, year, filename),
            format="NETCDF4",
            engine="netcdf4",
            encoding=encoding,
        )
    
    
    
