# Script used to generate CONUS forecast with a trained RNN on HRRR grid
# Intended to run on historical period as hindcast, so it can be run quickly
# with stashed data

import sys
import pickle
import os.path as osp
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import json
import pandas as pd
import numpy as np
import yaml
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import xarray as xr
import shutil
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
from data_funcs import add_terrain, ds_to_numpy, preds_to_ds, calc_hod_trig, calc_doy_trig
import reproducibility
from models.moisture_rnn import RNN_Flexible,OperationalRNNPredictor, scale_3d
import ingest.HRRR as ih

# Config and Params
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
paths = Dict(read_yml(osp.join(CONFIG_DIR, "paths.yaml")))

# Module Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def predict_auto_batch(model,
                       X,
                       batch_sizes=(16384, 8192, 4096, 2048, 1024, 512, 256, 128, 32),
                       verbose=1, reset_state=True):
    """
    Predict using the largest batch size that fits in memory.

    NOTE: at this step for non-stateful model, batch size in predict is just a performance issue. The bigger the faster
    """
    last_exception = None

    for bs in batch_sizes:
        try:
            if verbose:
                print(f"Trying predict batch_size={bs}")
            preds = model.predict_cycle(X, batch_size=bs, verbose=verbose, reset_state=reset_state)
            #preds = model.predict(X, batch_size=bs, verbose=verbose)
            if verbose:
                print(f"Success with batch_size={bs}")
            return preds
        except (MemoryError, tf.errors.ResourceExhaustedError) as e:
            last_exception = e
            if verbose:
                print(f"Failed with batch_size={bs}")

    raise RuntimeError(
        "All batch sizes failed during prediction."
    ) from last_exception

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print(f"Invalid arguments. {len(sys.argv)} was given but 2 expected")
        print(('Usage: %s <config_path>' % sys.argv[0]))
        print("Example: python src/hindcast.py etc/hindcast_TEST.yaml")
        sys.exit(-1)

    # Get input args
    conf_path = sys.argv[1]
    
    # Extract config details, save to outdir
    # Save to timestamped subdirectory
    conf = Dict(read_yml(conf_path))
    fstart = str2time(conf.f_start).replace(tzinfo=None, minute=0, second=0, microsecond=0)
    fend = str2time(conf.f_end).replace(tzinfo=None, minute=0, second=0, microsecond=0)
    outdir = conf.forecast_dir
    outdir = osp.join(outdir, f"{fstart:%Y%m%dT%H%M}_{fend:%Y%m%dT%H%M}")
    t_dir = conf.target_model_dir
    os.makedirs(outdir, exist_ok=True)
    save_yaml(dict(conf), outdir, "config.yaml")
    hrrr_dir = paths.hrrr_stash_path
    params = Dict(read_yml(osp.join(t_dir, "params.yaml")))
    save_yaml(dict(params), outdir, "params.yaml")
    # bbox

    # Static Data, elevation, land-sea-mask
    terrain = xr.open_dataset(osp.join(paths.landfire_elev_dir, "hrrr_terrain.nc"))
    lsm = terrain["lsm"]

    # Read trained model weights
    rnn = OperationalRNNPredictor.from_weights(params, osp.join(t_dir, "rnn.weights.h5"))
    rnn.save_weights(osp.join(outdir, "rnn.weights.h5"))
    scaler = load(osp.join(t_dir, "scaler.joblib"))
    dump(scaler, osp.join(outdir, "scaler.joblib"))


    # Get HRRR data, check stash and retrieve if missing
    # Default to save to stash f03 model, treated as analysis data
    print("~"*75)
    print(f"Forecasting with RNN from {fstart} to {fend}")
    print(f"Saving gridded forecasts to {outdir}")
    print(f"Loading HRRR data from stash {hrrr_dir}")
    print()

    # Cycle over days of data, first period use None initial state,
    # then save moving forward
    # TODO: handle fend falling within a full cycle at end
    cycle_length = conf.get("cycle_length", 12)  # number of hours to group together for cyclical prediction
    cycles = time_range(fstart, fend, freq=f"{int(cycle_length)}h") 
    print(f"Cycle length: {cycle_length}")
    features_list = params.features_list
    if len(features_list) != len(set(features_list)):
        raise ValueError("features_list contains duplicate features")
    for i, cycle in enumerate(cycles):
        print(f"    Processing cycle start: {cycle}, {i} out of {len(cycles)}")
        cycle_start = cycle
        cycle_end = cycle + pd.Timedelta(cycle_length-1, unit="hours")
        times = time_range(cycle_start, cycle_end, freq="1h")
        ds = ih.retrieve_hrrr(cycle_start, cycle_end)
        ds = ih.rename_ds(ds)
        ds = add_terrain(ds, terrain)
        # Set valid time, f03 shifted, as dimension
        ds = ds.assign_coords(time=ds.valid_time).drop_vars("valid_time")

        # Calculate Derived Features
        if "lograin" in features_list:
            ds["lograin"] = np.log1p(ds["rain"])
        ds["hod_sin"], ds["hod_cos"] = calc_hod_trig(ds["hod"])
        ds["doy_sin"], ds["doy_cos"] = calc_doy_trig(ds["doy"])

        # Format as input array to RNN, (nbatch, ntime, nfeat) 
        print(f"    Subsetting HRRR data to features: {features_list}")
        ds = ds[features_list]
        coord_features = [name for name in features_list if name in ds.coords] # Features from list that exist in xarray coordinates rather than data_vars
        ds = ds.reset_coords(coord_features, drop=False)
        ds["lon"] = ((ds["lon"] + 180) % 360) - 180 # fix longitude convention
        assert set(ds.data_vars) == set(features_list), f"Feature mismatch: expected {features_list}, got {list(ds.data_vars)}"
        print(f"    Converting xarray to numpy object")
        X_gridded = ds_to_numpy(ds, features_list)
        assert X_gridded.shape == (ds.x.shape[0] * ds.y.shape[0], len(times), len(features_list)), f"Unexpected X array shape: {X_gridded.shape=}, expected={(ds.x.shape[0] * ds.y.shape[0], len(times), len(features_list))}"

        # Scale Data
        X = scale_3d(X_gridded, scaler)

        # Predict, try large batch sizes for memory. only reset states on initial cycle, 
        # then reuse internally stored states
        # batch size in predict is only a memory constraint and not related to training.
        preds = predict_auto_batch(rnn, X, reset_state=(i==0))
        assert preds.shape[-1] == 1, f"Expected one output feature, got shape {preds.shape}"
        preds = preds.squeeze(axis=-1)
        cycle_ds = preds_to_ds(preds, ds)
        cycle_ds["fm10"] = cycle_ds["fm10"].where(lsm == 1)
        cycle_ds = xr.merge([ds, cycle_ds])
        cycle_ds["lsm"] = lsm


        # Write out cycle
        file_name = (f"fm_preds_"f"{cycle_start:%Y%m%d_%H}_"f"{cycle_end:%Y%m%d_%H}.nc")
        file_path = osp.join(outdir, file_name)
        print(f"Writing predictions to netcdf: {file_path}")
        cycle_ds.to_netcdf(file_path)
    
