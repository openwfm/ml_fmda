# Script used to forecast with a trained RNN on HRRR grid
# Intended for operational use, not for forecast analysis which
# has it's own set of scripts

import sys
import pickle
import os.path as osp
import os
from dateutil.relativedelta import relativedelta
import json
import pandas as pd
import numpy as np
import yaml
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import xarray as xr
import shutil

# Set up project paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CURRENT_DIR = osp.dirname(osp.normpath(osp.abspath(__file__)))
PROJECT_ROOT = osp.dirname(osp.normpath(CURRENT_DIR))
sys.path.append(osp.join(PROJECT_ROOT, "src"))
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import read_yml, read_pkl, Dict, str2time, time_range
import data_funcs
import reproducibility
from models.moisture_rnn import RNN_Flexible, RNNData, scale_3d
import ingest.HRRR as ih

# Config and Params
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
paths = Dict(read_yml(osp.join(CONFIG_DIR, "paths.yaml")))

# Module Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



if __name__ == '__main__':

    if len(sys.argv) != 4:
        print(f"Invalid arguments. {len(sys.argv)} was given but 4 expected")
        print(('Usage: %s <train_dir> <config_path> <output_dir>' % sys.argv[0]))
        print("<train_dir> is where trained model sent. <config_path> is path to yaml file setting up time frame and other analysis parameters")
        print("Example: python src/forecast.py models/train_test etc/forecast_TEST.yaml", "forecasts/TEST")
        sys.exit(-1)

    # Get input args
    t_dir = sys.argv[1]
    conf_path = sys.argv[2]
    forecast_dir = sys.argv[3]
    os.makedirs(forecast_dir, exist_ok=True)

    # Extract config details
    conf = Dict(read_yml(conf_path))
    fstart = str2time(conf.f_start)
    fend = str2time(conf.f_end)
    out_dir = paths.forecast_output
    hrrr_dir = paths.hrrr_stash_path
    params = Dict(read_yml(osp.join(t_dir, "params.yaml")))
    # bbox

    # Read trained model
    rnn = tf.keras.models.load_model(osp.join(t_dir, 'rnn.keras'))

    print("~"*75)
    print(f"Forecasting with RNN from {fstart} to {fend}")
    print(f"Saving gridded forecasts to {out_dir}")
    print()

    print(f"    Loading HRRR data from stash {hrrr_dir}")
    ds = ih.retrieve_hrrr(fstart, fend)
    terrain = xr.open_dataset(osp.join(paths.landfire_elev_dir, "hrrr_terrain.nc"))
    # Check that lon/lat coordinates match
    if not (np.mean(terrain.longitude == ds.longitude).values == 1) and (np.mean(terrain.latitude == ds.latitude).values==1):
        print("Mismatch lon/lat coordinates between HRRR terrain and weather")
    # Join elevation, need to get times to line up
    ds["time"] = ds["valid_time"]
    terrain = terrain.drop_vars(["step", "valid_time", "surface"])
    terrain = terrain.broadcast_like(ds)
    ds = ih.rename_ds(ds)
    ds['lon'] = xr.where(ds.lon > 180, ds.lon - 360, ds.lon) # Fix lat/lon to match RAWS format
    ds = xr.merge([ds, terrain])
    #ds2 = ih.rename_ds(ds2)
    ds["elev"] = ds["orog"]# TODO: update to LF elevation

    # ds2.to_netcdf(osp.join(forecast_dir, "hrrr_ds.nc"))
    #elev = xr.open_dataset(osp.join(paths.landfire_elev_dir, "lf_elevation_hrrrgrid.tif"))

    # Format input dataframe for RNN predict
    # Subset to features list used by rnn, some features are data_vars in xarray but some are coords
    # TODO: to_dataframe is slow, test to_numpy
    features_list = params.features_list
    print(f"    Subsetting HRRR data to features: {features_list}")
    ds2 = ds[features_list]
    coord_features = [name for name in features_list if name in ds2.coords] # Features from list that exist in xarray coordinates rather than data_vars
    ds2 = ds2.reset_coords(coord_features, drop=False)
    assert len(ds2.data_vars) == len(features_list), f"Missing features from list, {features_list=}, data_vars= {(list(ds2.data_vars))}"
    ds_stacked = ds2[features_list].stack(loc=("y", "x"))
    ds_transposed = ds_stacked.transpose("loc", "time", ...)
    X = ds_transposed.to_array().transpose("loc", "time", "variable").values

    times = time_range(fstart, fend)
    assert X.shape == (ds.x.shape[0] * ds.y.shape[0], len(times), len(features_list)), f"Unexpected X array shape: {X.shape=}, expected={(ds.x.shape[0] * ds.y.shape[0], len(times), len(features_list))}"

    # Run prediction with RNN
    # NOTE: batch size in predict is only a memory constraint and not related to batch_size used in training. 
    # We want to make batch_size as large as possible while avoiding memory constraints
    try:
        preds = rnn.predict(X, batch_size=1024, verbose=1)
    except (MemoryError, tf.errors.ResourceExhaustedError) as e:
        print("Batch size 1024 failed due to memory limits. Falling back to batch size 32.")
        preds = rnn.predict(X, batch_size=32, verbose=1)

    # Reshape preds and assign to an xarray object for save
    preds = preds.squeeze() # NOTE: this only works with 1d prediction. If ever go to 2-d, break up preds and add each separately
    pred_da = xr.DataArray(
        preds,
        dims=("loc", "time"),
        coords={
            "loc": ds_transposed["loc"],
            "time": ds_transposed["time"]
        },
        name="predicted"
    )
    pred_da = pred_da.unstack("loc")  # dims: (y, x, time)
    ds["fm_preds"] = pred_da.transpose("time", "y", "x")
    ds["lsm"] = terrain.lsm
    # Write out
    print(f"Writing predictions to netcdf: {osp.join(forecast_dir, 'fm_preds_hrrr.nc')}")
    ds.to_netcdf(osp.join(forecast_dir, "fm_preds_hrrr.nc"))
    
