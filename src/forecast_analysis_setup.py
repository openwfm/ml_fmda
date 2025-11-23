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

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print(f"Invalid arguments. {len(sys.argv)} was given but 3 expected")
        print(('Usage: %s <forecast_dir> <config_path>' % sys.argv[0]))
        print("<forecast_dir> is where outputs from the forecasts are sent.  <config_path> is path to yaml file setting up time frame and other analysis parameters")
        print("Example: python src/forecast_analysis_setup.py forecasts/fmc_forecast_test etc/forecast_analysis_TEST.yaml")
        sys.exit(-1)

    # Get input args
    f_dir = sys.argv[1]
    conf_path = sys.argv[2]
    os.makedirs(osp.join(f_dir, 'forecast_outputs'), exist_ok=True)
    
    # Check if already run, allows for easy rerun of process
    if osp.exists(osp.join(f_dir, 'ml_data.pkl')) and osp.exists(osp.join(f_dir, 'analysis_info.json')):
        print(f"Forecast analysis setup already run at {f_dir}, exiting")
        sys.exit(0)

    # Set up forecast directory and config
    os.makedirs(f_dir, exist_ok=True)
    fconf = read_yml(conf_path)
    # Write copy of forecast config file to forecast directory
    # Do this so multiple tests can be run with different input config files
    with open(osp.join(f_dir, "forecast_config.yaml"), 'w') as f:
        yaml.dump(fconf, f, default_flow_style=False, sort_keys=False)
    fconf = Dict(fconf)
    data_dir = fconf["data_dir"]
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
    data = data_funcs.combine_fmda_files(file_paths, atm_source=fconf.atm_source)
    ml_dict = data_funcs.build_ml_data(data, verbose=False, atm_source=fconf.atm_source)
    if fconf.atm_source == "RAWS":
        # Limit to stations with all sensor variables in features list
        print(f"Filtering stations to ones with sensors for features list: {fconf.features_list}")
        cond = {st: all(s in ml_dict[st]["data"].columns for s in fconf.features_list) for st in ml_dict}
        kept = [st for st, ok in cond.items() if ok]
        removed = [st for st, ok in cond.items() if not ok]
        print(f"    {len(kept)} stations kept.")
        print(f"    {len(removed)} stations removed: {', '.join(removed)}")
        ml_dict = {st: v for st, v in ml_dict.items() if cond[st]}
    df_valid = pd.read_csv(osp.join(PROJECT_ROOT, fconf.valid_path))
    ml_dict = data_funcs.remove_invalid_data(ml_dict, df_valid)
    data_file =  osp.join(PROJECT_ROOT, f_dir, "ml_data.pkl")
    with open(data_file, 'wb') as f:
        pickle.dump(ml_dict, f)





