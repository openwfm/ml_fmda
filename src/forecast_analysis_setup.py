# Script used to setup analysis of forecast error of models over a time period
# Creates formatted data for use in the various forecast periods

import sys
import pickle
import os.path as osp
import os
from dateutil.relativedelta import relativedelta
import json
import pandas as pd
import numpy as np
import yaml
import xarray as xr
from pathlib import Path
from itertools import groupby

# Set up project paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CURRENT_DIR = osp.dirname(osp.normpath(osp.abspath(__file__)))
PROJECT_ROOT = osp.dirname(osp.normpath(CURRENT_DIR))
sys.path.append(osp.join(PROJECT_ROOT, "src"))
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import read_yml, Dict, str2time, time_range
import data_funcs
import reproducibility


# Config and Params
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
params_models = read_yml(osp.join(CONFIG_DIR, "params_models.yaml"))
project_paths = Dict(read_yml(osp.join(CONFIG_DIR, "paths.yaml")))

# Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def calc_smap_files(days):
    return [osp.join(project_paths.smap_stash_path, "L4", day.strftime("%Y"), f"smap_L4_{day.strftime('%Y%m%d')}.nc") for day in days]

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print(f"Invalid arguments. {len(sys.argv)} was given but 2 expected")
        print(('Usage: %s <target_directory> <config_path>' % sys.argv[0]))
        print("<config_path> is path to yaml file setting up time frame and other analysis parameters")
        print("Example: python src/forecast_analysis_setup_CHUNKS.py forecast_analysis/TEST etc/forecast_analysis_TEST.yaml")
        sys.exit(-1)

    # Get input args
    f_dir = sys.argv[1]
    conf_path = sys.argv[2]
    fconf = read_yml(conf_path)
    os.makedirs(osp.join(f_dir, 'forecast_outputs'), exist_ok=True)
    
    # Write copy of forecast config file to forecast directory
    # Do this so multiple tests can be run with different input config files
    with open(osp.join(f_dir, "forecast_config.yaml"), 'w') as f:
        yaml.dump(fconf, f, default_flow_style=False, sort_keys=False)
    fconf = Dict(fconf)
    data_dir = fconf["data_dir"]
    # Update hard coded RNN params with any run-specific changes
    params_models["rnn"].update(fconf)
    
    # Write copy of model params config file to forecast directory
    with open(osp.join(f_dir, "params_models.yaml"), 'w') as f:
        yaml.dump(params_models, f, default_flow_style=False, sort_keys=False)  

    # Write simplified copy of  config file as json, so shell files can use jq. (TODO: test with yq and no jsons)
    info = {
        'forecast_start': fconf.f_start,
        'forecast_end': fconf.f_end,
        'forecast_hours': fconf.forecast_hours,
        'train_start': fconf.train_start,
        'train_end': fconf.train_end,
        'nreps': fconf.n_reps,
        'data_input_dir': fconf.data_dir,
        'baselines': fconf.baselines
    }
    info_file = osp.join(f_dir, 'analysis_info.json')
    with open(info_file, "w") as json_file:
        json.dump(info, json_file)

    # Set up ML data used in train and test
    fstart = str2time(fconf.f_start)
    fend = str2time(fconf.f_end)
    tstart = str2time(fconf.train_start)
    tend = str2time(fconf.train_end)
    print("~"*75)
    print(f"Running Forecast Analysis from {fstart} to {fend}")
    print(f"Training from {tstart} to {tend}")
    print(f"Baseline methods: {fconf.baselines}")
    print()
    tdays = time_range(tstart, tend, freq="1d")
    fdays = time_range(fstart, fend, freq="1d")
    days = np.concat((tdays, fdays))    
    print(f"Days of Data Needed: {days.shape[0]}")
    print(f"Earliest Day of Data: {days.min()}")
    print(f"Latest Day of Data: {days.max()}")
    file_paths = [f"{data_dir}/{dt.strftime('%Y%m')}/fmda_{dt.strftime('%Y%m%d')}.pkl" for dt in days]
    all_exist = all(osp.exists(path) for path in file_paths)
    # For now, hard exit if not all data exists. Maybe relax in the future
    if not all_exist:
        print(f"Not all needed file paths exist for target analysis. Exiting...")
        missing_paths = [path for path in file_paths if not osp.exists(path)]
        print("Missing files:")
        for path in missing_paths:
            print(path)
        sys.exit(-1)
    else:
        print(f"All Needed Data exists in {data_dir}, proceeding...")
    
    # Read and Format Data, get set up for train and test
    print("~"*75) 
    monthly_file_paths = [
        list(group)
        for _, group in groupby(file_paths, key=lambda p: Path(p).parent.name)
    ]
    if osp.exists(fconf.valid_path):
        print(f"Using labeled valid data file: {fconf.valid_path}")
        df_valid = pd.read_csv(osp.join(PROJECT_ROOT, fconf.valid_path))
    else:
        print(f"No labeled valid data found at {fconf.valid_path}, proceeding with no filtering of bad RAWS")
        df_valid = None
    for paths in monthly_file_paths:
        month = Path(paths[0]).parent.name
        mpath = osp.join(PROJECT_ROOT, f_dir, "ml_data")
        output_file = Path(mpath) / f"ml_data_{month}.pkl"
        if osp.exists(output_file):
            print(f"Skipping {month}: output file already exists: {output_file}")
            continue

        print(f"Processing {month}...")
        os.makedirs(mpath, exist_ok=True)
        data = data_funcs.combine_fmda_files(paths)
        ml_dict = data_funcs.build_ml_data(data, verbose=False)

        if df_valid is not None:
            ml_dict = data_funcs.remove_invalid_data(ml_dict, df_valid)

        # Add any derived features
        data_funcs.add_derived_features(ml_dict)
        
        # Add SMAP if in features list
        if "sm_surface" in fconf.features_list:
            print(f"Adding SMAP data from {project_paths['smap_stash_path']}")
            files = calc_smap_files(days)
            print(f"Days of SMAP data needed: {len(files)}")
            assert np.all([osp.exists(fi) for fi in files]), f"Missing SMAP files, exiting"
            with xr.open_dataset(files[0]) as sm:
                data_funcs.add_smap_grid_indices(ml_dict, sm)
            data_funcs.add_smap(ml_dict, files)


        print(f"Writing data to {output_file}")
        with open(output_file, "wb") as f:
            pickle.dump(ml_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        del data
        del ml_dict


    print(f"Forecast analysis setup complete at directory: {f_dir}")

