# Script used to set up training data. Use before actual training called since when using reps we don't want to recreate the main data pool
# Intended for operational use, not for forecast analysis which
# has it's own set of scripts
# NOTE: the fastest GPU training relies on non-deterministic code, to make the code deterministic and reproducibile you can import the module reproducibility.py, but that will slow down training

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
import time
from joblib import dump, load
from itertools import groupby
from pathlib import Path

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
#import reproducibility

# Config and Params
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
params = read_yml(osp.join(CONFIG_DIR, "params_models.yaml"), subkey="rnn")
project_paths = Dict(read_yml(osp.join(CONFIG_DIR, "paths.yaml")))


def calc_smap_files(days):
    return [osp.join(project_paths.smap_stash_path, "L4", day.strftime("%Y"), f"smap_L4_{day.strftime('%Y%m%d')}.nc") for day in days]

if __name__ == '__main__':


    # Optional seed argument. When calling this script with slurm arrays, the array numbers get passed in as random seed
    # If seed passed, import reproduciblity for deterministic ops and set seed
    # If no second argument, run without deterministic ops
    if len(sys.argv) != 2:
        print(f"Invalid arguments. {len(sys.argv)} was given but 2 expected")
        print(('Usage: %s <config_path> [seed]' % sys.argv[0]))
        print("<config_path> is path to yaml file setting up time frame and other analysis parameters")
        print("Example: python src/train_setup.py etc/train_config_TEST.yaml")
        sys.exit(1)

    # Get input conf, construct target directory, and write configs
    conf_path = sys.argv[1]
    conf = read_yml(conf_path)
    # Overwrite any model parameters with run-specific config
    params.update(conf)

    # Setup output directory.
    data_dir = conf["data_dir"]
    region = conf["region"]
    tstart = str2time(conf['train_start'])
    tend = str2time(conf['train_end'])
    tstring =  f"{tstart.strftime('%Y%m%d')}-{tend.strftime('%Y%m%d')}" # time parameters string for naming model directory
    t_dir = osp.join(conf['target_model_dir'], f"{region}_{tstring}")    
    print("Creating directory {t_dir}", file=sys.stderr)
    print(t_dir) # NOTE: this is captured by shell file and passed to a later process, all other print statements need to specify file=sys.stderr to avoid messing it up
    os.makedirs(t_dir, exist_ok=True) 
    os.makedirs(osp.join(t_dir, "logs"), exist_ok=True)
    with open(osp.join(t_dir, "train_config.yaml"), 'w') as f:
        yaml.dump(conf, f, default_flow_style=False, sort_keys=False)
    with open(osp.join(t_dir, "params.yaml"), 'w') as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)
    
    days = time_range(tstart, tend, freq="1d")
    print("~"*75, file=sys.stderr)
    print(f"Training RNN from {tstart} to {tend}", file=sys.stderr)
    print(f"Saving trained model to {t_dir}", file=sys.stderr)
    
    # Build / Read training data dictionary
    # NOTE: stashed data organized in days, so read the full days that bracket input train times
    print(f"    Building Training Data", file=sys.stderr)
    # Read and Format Data into monthly files, get set up for train and test.
    
    file_paths = [f"{data_dir}/{dt.strftime('%Y%m')}/fmda_{dt.strftime('%Y%m%d')}.pkl" for dt in days]
    print("~"*75, file=sys.stderr)
    monthly_file_paths = [
        list(group)
        for _, group in groupby(file_paths, key=lambda p: Path(p).parent.name)
    ]
    if osp.exists(conf['valid_path']):
        print(f"Using labeled valid data file: {conf['valid_path']}", file=sys.stderr)
        df_valid = pd.read_csv(osp.join(PROJECT_ROOT, conf['valid_path']))
    else:
        print(f"No labeled valid data found at {conf['valid_path']}, proceeding with no filtering of bad RAWS", file=sys.stderr)
        df_valid = None

    # TEST STEP August 5: filter to specific GACCs
    conf = Dict(conf)
    if "gaccs" in conf:
        gacc =  Dict(json.load(open('/data001/projects/hirschij/github/wrfxpy/etc/fmda_cycler_all.json')))
        gacc_regions = gacc["regions"]
        regions = {
            region_name: {
                "code": region["code"],
                "bbox": region["bbox"],
            }
            for region_name, region in gacc["regions"].items()
            if region["code"] in conf.gaccs
        }
        print(f"Filtering to regions: {[*regions.keys()]}", file=sys.stderr)
    else:
        regions = None 


    for paths in monthly_file_paths:
        month = Path(paths[0]).parent.name
        mpath = osp.join(PROJECT_ROOT, t_dir, "ml_data")
        output_file = Path(mpath) / f"ml_data_{month}.pkl"
        if osp.exists(output_file):
            print(f"Skipping {month}: output file already exists: {output_file}", file=sys.stderr)
            continue

        print(f"Processing {month}...", file=sys.stderr)
        os.makedirs(mpath, exist_ok=True)
        data = data_funcs.combine_fmda_files(paths)

        # Filter GACCs, NOTE this is not an efficient way to do things, just test code. Proper implementation would require changing data_funcs modules
        if regions is not None:
            for stid in list(data):
                lat = data[stid]["loc"]["lat"]
                lon = data[stid]["loc"]["lon"]
                keep = False
                for region in regions.values():
                    min_lat, min_lon, max_lat, max_lon = region["bbox"]
                    if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                        keep = True
                        break
                if not keep:
                    del data[stid]

        ml_dict = data_funcs.build_ml_data(data, verbose=False)

        if df_valid is not None:
            ml_dict = data_funcs.remove_invalid_data(ml_dict, df_valid)

        # Add any derived features
        data_funcs.add_derived_features(ml_dict)

        # Add SMAP if in features list
        if "sm_surface" in conf.features_list:
            import xarray as xr
            print(f"Adding SMAP data from {project_paths['smap_stash_path']}", file=sys.stderr)
            files = calc_smap_files(days)
            print(f"Days of SMAP data needed: {len(files)}", file=sys.stderr)
            assert np.all([osp.exists(fi) for fi in files]), f"Missing SMAP files, exiting"
            with xr.open_dataset(files[0]) as sm:
                data_funcs.add_smap_grid_indices(ml_dict, sm)
            data_funcs.add_smap(ml_dict, files)

        print(f"Writing data to {output_file}", file=sys.stderr)
        with open(output_file, "wb") as f:
            pickle.dump(ml_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        del data
        del ml_dict



    print(f"Training setup complete at directory: {t_dir}", file=sys.stderr)

