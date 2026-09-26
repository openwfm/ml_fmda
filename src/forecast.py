# Script used to generate forecast with a trained RNN on HRRR grid
# Intended for operational use, not for forecast analysis which
# has it's own set of scripts
# HRRR 48h forecast, starting from f03 so 45 forecast window
# NOTE: the process src/hindcast.py is set up to run forecast model on a historical period using HRRR f03 as "analysis" data. 

import sys
import pickle
import os.path as osp
import os
from datetime import datetime, timedelta, timezone
import json
import pandas as pd
import numpy as np
import yaml
import tensorflow as tf
#import xarray as xr
import warnings
from joblib import dump, load

# Set up project paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CURRENT_DIR = osp.dirname(osp.normpath(osp.abspath(__file__)))
PROJECT_ROOT = osp.dirname(osp.normpath(CURRENT_DIR))
sys.path.append(osp.join(PROJECT_ROOT, "src"))
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import read_yml, read_pkl, Dict, str2time, time_range, save_yaml
from data_funcs import add_terrain
import reproducibility
#from models.moisture_rnn import RNN_Flexible, RNNData, scale_3d
#from ingest.HRRR import rename_ds, retrieve_hrrr, retrieve_hrrr_fcst

# Config and Params
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
paths = Dict(read_yml(osp.join(CONFIG_DIR, "paths.yaml")))

# Module Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print(f"Invalid arguments. {len(sys.argv)} was given but 2 expected")
        print(('Usage: %s <config_path>' % sys.argv[0]))
        print("Example: python src/forecast.py etc/forecast_TEST.yaml")
        sys.exit(-1)
    
    # Get input args
    conf_path = sys.argv[1]
    breakpoint() 
    # Extract config details, save to outdir
    conf = Dict(read_yml(conf_path))
    outdir = conf["forecast_dir"]
    hrrr_dir = paths.hrrr_stash_path
    t_dir = conf["target_model_dir"]
    os.makedirs(outdir, exist_ok=True)
    save_yaml(dict(conf), outdir, "config.yaml")

    # Get RNN params file from target model directory, save copy to outdir
    params = Dict(read_yml(osp.join(t_dir, "params.yaml")))
    save_yaml(dict(params), outdir, "params.yaml")
    
    # Get times, default to now and (48-3) hour forecast 
    ## Start time defaults to now, rounds down to nearest whole hour
    ## `now` rounds down to nearest whole hour
    ## Time logic: check whether fstart + fcst hours is in future or not
    ## relative to now(). 
    ## For hours in the past, get HRRR f03 as analysis time. Latency buffer of 3hrs
    ## For hours in the future, logic to get overlapping fcast times,
    ## based on 48 hour extension every 6 hrs
    now = datetime.now(timezone.utc).replace(tzinfo=None, minute=0, second=0, microsecond=0)
    now = str2time("2026-07-17 12:00:00+00:00").replace(tzinfo=None, minute=0, second=0, microsecond=0)## DEBUG STEP
    fstart = str2time(conf.get("f_start", now)).replace(tzinfo=None, minute=0, second=0, microsecond=0)
    fcst_hours = conf.get("fcst_hours", 45)
    fend = fstart + timedelta(hours=fcst_hours)
    spinup_hours = conf.get("spinup", 0)
    spinup_start = fstart - timedelta(hours=spinup_hours)


    # Build model from input model directory
    rnn = OperationalRNNPredictor.from_weights(params, osp.join(t_dir, "rnn.weights.h5"))
    rnn.save_weights(osp.join(outdir, "rnn.weights.h5"))
    scaler = load(osp.join(t_dir, "scaler.joblib"))
    dump(scaler, osp.join(outdir, "scaler.joblib"))
    features_list = params.features_list
    if len(features_list) != len(set(features_list)):
        raise ValueError("features_list contains duplicate features")


    # Static Data, elevation, land-sea-mask
    if osp.exist(osp.join(paths.landfire_elev_dir, "hrrr_terrain.nc"))
        terrain = xr.open_dataset(osp.join(paths.landfire_elev_dir, "hrrr_terrain.nc"))
        lsm = terrain["lsm"]
    else:
        print("No HRRR terrain stash found, attempting retrieval")
        raise NotImplementedError("")

    # Get HRRR data, check stash and retrieve if missing
    # Default to save to stash f03 model, treated as analysis data
    print("~"*75)
    print(f"Forecasting with RNN from {fstart} to {fend}")
    print(f"Saving gridded forecasts to {outdir}")
    print()

    if spinup_hours>0:
        ds0 = retrieve_hrrr(spinup_start, f_start) 
    ds = retrieve_hrrr_fcst(fstart, fend)
    ds = rename_ds(ds)
    ds = add_terrain(ds, terrain)
    ds = ds.assign_coords(time=ds.valid_time).drop_vars("valid_time")

    print(f"Subsetting HRRR data to features: {features_list}")
    ds = ds[features_list]
    #coord_features = [name for name in features_list if name in ds.coords] # Features from list that exist in xarray coordinates rather than data_vars
    #ds = ds.reset_coords(coord_features, drop=False)
    #ds["lon"] = ((ds["lon"] + 180) % 360) - 180 # fix longitude convention
    assert set(ds.data_vars) == set(features_list), f"Feature mismatch: expected {features_list}, got {list(ds.data_vars)}"
    print(f"Converting xarray to numpy object")
    X_gridded = ds_to_numpy(ds, features_list)
    assert X_gridded.shape == (ds.x.shape[0] * ds.y.shape[0], len(times), len(features_list)), f"Unexpected X array shape: {X_gridded.shape=}, expected={(ds.x.shape[0] * ds.y.shape[0], len(times), len(features_list))}"

    # Scale Data
    X = scale_3d(X_gridded, scaler)

    breakpoint()
