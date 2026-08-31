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
from data_funcs import add_terrain
import reproducibility
from models.moisture_rnn import RNN_Flexible,OperationalRNNPredictor
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
            #preds = model.predict_cycle(X, batch_size=bs, verbose=verbose, reset_state=reset_state)
            preds = model.predict(X, batch_size=bs, verbose=verbose)
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

    # Read trained model
    rnn = tf.keras.models.load_model(osp.join(t_dir, 'rnn.keras'))
    rnn.save_weights(osp.join(outdir, "rnn.weights.h5"))
    
    scaler = load(osp.join(t_dir, "scaler.joblib"))
    dump(scaler, osp.join(outdir, "scaler.joblib"))

    rnn2 = OperationalRNNPredictor.from_weights(params, osp.join(t_dir, "rnn.weights.h5"))
    # Get HRRR data, check stash and retrieve if missing
    # Default to save to stash f03 model, treated as analysis data
    print("~"*75)
    print(f"Forecasting with RNN from {fstart} to {fend}")
    print(f"Saving gridded forecasts to {outdir}")
    print()

    print(f"    Loading HRRR data from stash {hrrr_dir}")
    ds = ih.retrieve_hrrr(fstart, fend)

    # Static HRRR data, join to timeseries rasters
    terrain = xr.open_dataset(osp.join(paths.landfire_elev_dir, "hrrr_terrain.nc"))
    ds = ih.rename_ds(ds)
    ds = add_terrain(ds, terrain)
    
    # Set valid time, f03 shifted, as dimension
    ds = ds.assign_coords(time=ds.valid_time).drop_vars("valid_time")

    #elev = xr.open_dataset(osp.join(paths.landfire_elev_dir, "lf_elevation_hrrrgrid.tif"))

    # Format input dataframe for RNN predict
    # Subset to features list used by rnn, some features are data_vars in xarray but some are coords
    features_list = params.features_list
    if "lograin" in features_list:
        ds["lograin"] = np.log1p(ds["rain"])
    
    print(f"    Subsetting HRRR data to features: {features_list}")
    ds = ds[features_list]
    coord_features = [name for name in features_list if name in ds.coords] # Features from list that exist in xarray coordinates rather than data_vars
    ds = ds.reset_coords(coord_features, drop=False)
    assert len(ds.data_vars) == len(features_list), f"Missing features from list, {features_list=}, data_vars= {(list(ds.data_vars))}"
    ds_stacked = ds[features_list].stack(loc=("y", "x"))
    ds_transposed = ds_stacked.transpose("loc", "time", ...)
    X_gridded = ds_transposed.to_array().transpose("loc", "time", "variable").values

    times = time_range(fstart, fend)
    assert X_gridded.shape == (ds.x.shape[0] * ds.y.shape[0], len(times), len(features_list)), f"Unexpected X array shape: {X.shape=}, expected={(ds.x.shape[0] * ds.y.shape[0], len(times), len(features_list))}"

    # Run prediction with RNN
    # NOTE: batch size in predict is only a memory constraint and not related to batch_size used in training. 
    # We want to make batch_size as large as possible while avoiding memory constraints
    
    ## Scale Data
    # Reshape to 2d table to apply scaler, flatten (xy) dimensions
    X_flat = X_gridded.reshape(-1, X_gridded.shape[-1])
    X_scaled = scaler.transform(X_flat)
    nbatch, ntimes, nfeatures = X_gridded.shape
    assert X_scaled.shape[0] == nbatch * ntimes
    assert X_scaled.shape[1] == nfeatures
    X = X_scaled.reshape(nbatch, ntimes, nfeatures)
  
    # Predict, try large batch sizes for memory
    preds = predict_auto_batch(rnn, X)

    # Reshape preds and assign to an xarray object for save
    preds = preds.squeeze() # NOTE: this only works with 1d prediction. If ever go to 2-d, break up preds and add each separately
   
    breakpoint()
    preds2 = rnn2.predict_cycle(
            X, 
            reset_state=True,     # predicting from fresh start
            initial_states=None,  # start from naive, zeros by default
            return_states=True   # keep to use for cyclical
            )
    
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
    ds["fm10"] = pred_da.transpose("time", "y", "x")
    ds["lsm"] = terrain.lsm
    
    # Write out
    print(f"Writing predictions to netcdf: {osp.join(outdir, 'fm_preds_hrrr.nc')}")
    ds.to_netcdf(osp.join(outdir, f"fm_rnn.nc"))
    
