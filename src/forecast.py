# Script used to generate CONUS forecast with a trained RNN on HRRR grid
# Intended for operational use, not for forecast analysis which
# has it's own set of scripts

import sys
import pickle
import os.path as osp
import os
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
import json
import pandas as pd
import numpy as np
import yaml
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import xarray as xr
import shutil
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
from models.moisture_rnn import RNN_Flexible, RNNData, scale_3d
import ingest.HRRR as ih

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
    if fstart > now: 
        warnings.warn(f"Forecast start greater than now, check time input. {fstart=},   {now=}")
        raise NotImplementedError("Future forecast start times are not currently supported.")

    if fstart.hour not in (0, 6, 12, 18):
        raise NotImplementedError("Forecast start must correspond to an extended HRRR cycle (00, 06, 12, or 18 UTC).")

    print(f"{fstart=}"); print(f"{fcst_hours=}"); print(f"{fend=}")
    if fend < now:
        print(f"Forecast period all analysis times, using f03 HRRR for all hours")
        astart = fstart
        aend = fend
        fstart = None
        fend = None
    else:
        astart = fstart
        aend = now-timedelta(hours=1) # assume f03 from 3 hours in past exists, safe buffer
        fstart = now
        if fend > (now+timedelta(hours=(48-3))):
            fend = now + timedelta(hours=48-3)
            warnings.warn(f"Forecast end outside HRRR 45hr forecast window. Trimming fend to {fend}")
        print(f"Analysis hours: {astart} to {aend}"); print(f"Forecast hours: {fstart} to {fend}")

    
    # TODO: bbox?

    # Read trained model
    if osp.isfile(osp.join(t_dir, 'rnn.keras')):
        rnn = tf.keras.models.load_model(osp.join(t_dir, 'rnn.keras'))
        scaler = load(osp.join(t_dir, "scaler.joblib"))
    elif osp.isfile(osp.join(t_dir, "median_seed.csv")):
        med = pd.read_csv(osp.join(t_dir, 'median_seed.csv'))
        print(f"Reading model from median fitting accuracy seed: seed_{med['seed'][0]}")
        rnn = tf.keras.models.load_model(osp.join(t_dir, f"seed_{med['seed'][0]}", 'rnn.keras'))
        scaler = load(osp.join(t_dir, f"seed_{med['seed'][0]}", "scaler.joblib"))
    else:
        raise RuntimeError(f"Required model file not found in {t_dir}")

    print("~"*75)
    print(f"Forecasting with RNN from {fstart} to {fend}")
    print(f"Saving gridded forecasts to {outdir}")
    print()

    # Handle Weather Inputs
    ## Split forecast period into analysis and forecast, use associated retrieval
    ## By default, retrieve_hrrr saves f03 to stash. Config can turn off and just save to memory
    print(f"    Loading HRRR data, Herbie API tool and/or stash {hrrr_dir}")
    ds = ih.retrieve_hrrr(astart, aend, save_to_stash=conf.get("save_to_stash",True))
    
    if fstart is not None: 
        dsf = ih.retrieve_hrrr_fcst(fstart, fend, features_list=params["features_list"])
    else:
        dsf = None


    ## Static Data
    terrain = xr.open_dataset(osp.join(paths.landfire_elev_dir, "hrrr_terrain.nc"))
   
    # Analysis Data
    ds = ih.rename_ds(ds)
    ds = add_terrain(ds, terrain)

    # Forecast Data
    dsf = ih.rename_ds(dsf)
    dsf = add_terrain(dsf, terrain)
    #dsf['lon'] = xr.where(dsf.lon > 180, dsf.lon - 360, dsf.lon) # Fix lat/lon to match RAWS format
    #elev = xr.open_dataset(osp.join(paths.landfire_elev_dir, "lf_elevation_hrrrgrid.tif"))

    # Format input dataframe for RNN predict
    # Subset to features list used by rnn, some features are data_vars in xarray but some are coords
    features_list = params.features_list
    if "lograin" in features_list:
        ds["lograin"] = np.log1p(ds["rain"])
        dsf["lograin"] = np.log1p(dsf["rain"])
    
    print(f"    Subsetting HRRR data to features: {features_list}")
    ds = ds[features_list] 
    dsf = dsf[features_list]
    
    coord_features = [name for name in features_list if name in ds.coords] # Features from list that exist in xarray coordinates rather than data_vars
    ds = ds.reset_coords(coord_features, drop=False)
    dsf = dsf.reset_coords(coord_features, drop=False)
    
    assert len(ds.data_vars) == len(features_list), f"Missing features from list, {features_list=}, data_vars= {(list(ds.data_vars))}"
    assert len(dsf.data_vars) == len(features_list), f"Missing features from list, {features_list=}, data_vars= {(list(dsf.data_vars))}"
    
    # Reshape to 2d input
    ds_stacked = ds[features_list].stack(loc=("y", "x"))
    dsf_stacked = dsf[features_list].stack(loc=("y", "x"))
    
    ds_transposed = ds_stacked.transpose("loc", "time", ...)
    dsf_transposed = dsf_stacked.transpose("loc", "time", ...)

    X_gridded = ds_transposed.to_array().transpose("loc", "time", "variable").values
    Xf_gridded = dsf_transposed.to_array().transpose("loc", "time", "variable").values


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
    print(f"Writing predictions to netcdf: {osp.join(outdir, 'fm_preds_hrrr.nc')}")
    ds.to_netcdf(osp.join(outdir, "fm_preds_hrrr.nc"))
    
