# Set of Functions to process and format fuel moisture model inputs
# These functions are specific to the particulars of the input data, and may not be generally applicable
# Generally applicable functions should be in utils.py
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from datetime import datetime
import numpy as np
import os
import os.path as osp
import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from .utils import Dict, read_yml, read_pkl, time_range, str2time, is_consecutive_hours, time_intp
from . import reproducibility



def scale_3d(X, scaler, fit=False):
    """
    Apply an sklearn scaler to 3d numpy arrays
    
    Parameters:
    -----------
    X : ndarray of shape (n_locs, timesteps, features)
    scaler : fitted scaler with .transform method
    fit : bool, optional
        If True, fit the scaler on X before transforming. Default is False.    

    Returns:
    --------
    X_scaled : ndarray of same shape as X    
    """
    n_locs, timesteps, features = X.shape
    X_flat = X.reshape(-1, features)
    if fit:
        scaler.fit(X_flat)
    X_scaled_flat = scaler.transform(X_flat, copy=False)
    X_scaled = X_scaled_flat.reshape(n_locs, timesteps, features)

    return X_scaled


# Feature Engineering Utilities
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def calc_hod(t):
    return t.strftime("%H")

def calc_doy(t):
    return t.strftime("%j")

# Cyclical time encoding
## Example Source: https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html?utm_source=chatgpt.com
def calc_hod_trig(hod):
    """
    Convert hour of day (0-23) to cyclic sine and cosine features.
    Args:
        hod: Scalar or array-like hour of day.
    Returns:
        Tuple (hod_sin, hod_cos).
    """
    hod_sin = np.sin(2 * np.pi * hod / 24)
    hod_cos = np.cos(2 * np.pi * hod / 24)    
    return hod_sin, hod_cos

def calc_doy_trig(doy):
    """
    Convert day of year (1-365) to cyclic sine and cosine features.
    Args:
        doy: Scalar or array-like day of year.
    Returns:
        Tuple (doy_sin, doy_cos).
    """
    doy_sin = np.sin(2 * np.pi * (doy - 1) / 365)
    doy_cos = np.cos(2 * np.pi * (doy - 1) / 365)
    return doy_sin, doy_cos


def add_terrain(ds, terrain):
    """
    Helper to join static terrain data from a raster to a time series raster
    """
    import xarray as xr
    
    # Check that lon/lat coordinates match
    if not (np.mean(terrain.longitude == ds.lon).values == 1) and (np.mean(terrain.latitude == ds.lat).values==1):
        print("Mismatch lon/lat coordinates between HRRR terrain and weather")
    # Join elevation, need to get times to line up
    #ds["time"] = ds["valid_time"]
    terrain = terrain.drop_vars(["step", "valid_time", "surface"])
    terrain = terrain.broadcast_like(ds)
    
    terrain = terrain.drop_vars(["step", "valid_time", "surface"], errors="ignore")
    ds = xr.merge([ds, terrain])
    ds["elev"] = ds["orog"]
    ds["lsm"] = terrain["lsm"]
    return ds


### Tools for Joining SMAP data
from scipy.spatial import cKDTree
import numpy as np


def add_smap_grid_indices(ml_dict, sm):
    """
    Add nearest SMAP grid indices to each station in-place.

    Args:
        ml_dict: Dictionary of station data. Each station must contain
            station["loc"]["lat"] and station["loc"]["lon"].
        sm: xarray Dataset containing 2D variables
            sm.cell_lat and sm.cell_lon.

    Returns:
        None
    """
    import xarray as xr

    # Build KDTree from SMAP grid
    lat = sm.cell_lat.values
    lon = sm.cell_lon.values
    tree = cKDTree(np.column_stack((lat.ravel(), lon.ravel())))

    # Gather station coordinates
    stids = list(ml_dict.keys())
    coords = np.array([
        (
            ml_dict[st]["loc"]["lat"],
            ml_dict[st]["loc"]["lon"],
        )
        for st in stids
    ])

    # Find nearest SMAP grid cells
    _, inds = tree.query(coords)
    yind, xind = np.unravel_index(inds, lat.shape)

    # Store indices in-place
    for st, y, x in zip(stids, yind, xind):
        ml_dict[st]["loc"]["smap_grid_y"] = int(y)
        ml_dict[st]["loc"]["smap_grid_x"] = int(x)


def add_smap(ml_dict, files):
    """
    Add SMAP soil moisture to each station's data.

    Assumes add_smap_grid_indices() has already been called.
    """
    import xarray as xr

    for fi in files:
        with xr.open_dataset(fi) as sm:
            sm_times = pd.to_datetime(sm.time.values, utc=True)
            sm_surface = sm["sm_surface"].values
            for station in ml_dict.values():
                times = station["times"]
                y = station["loc"]["smap_grid_y"]
                x = station["loc"]["smap_grid_x"]
                for i, (t0, t1) in enumerate(zip(sm_times[:-1], sm_times[1:])):
                    mask = (times >= t0) & (times < t1)
                    if not np.any(mask):
                        continue
                    station["data"].loc[mask, "sm_surface"] = sm_surface[i, y, x]


def ds_to_numpy(ds, features_list):
    arr = (
        ds[features_list]
        .to_array("feature")
        .transpose("y", "x", "time", "feature")
        .values
    )

    ny, nx, nt, nf = arr.shape

    return arr.reshape(ny * nx, nt, nf)

def preds_to_ds(preds, ds):
    """Convert (loc, time) predictions to (time, y, x) Dataset."""
    import xarray as xr
    ny = ds.sizes["y"]
    nx = ds.sizes["x"]
    nt = ds.sizes["time"]

    fm10 = preds.reshape(ny, nx, nt).transpose(2, 0, 1)
    return xr.Dataset(
        {"fm10": (("time", "y", "x"), fm10)},
        coords={
            "time": ds["time"].values,
            "y": ds["y"].values,
            "x": ds["x"].values,
            "gribfile_projection": ds["gribfile_projection"],
        },
    )



