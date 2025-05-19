# Script used to setup analysis of forecast error of models over a time period
# Creates formatted data for use in the various forecast periods

import sys
import pickle
import os.path as osp
import os
from dateutil.relativedelta import relativedelta
import json
import pandas as pd
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
from models.moisture_static import XGB
from models.moisture_ode import ODE_FMC
from models.moisture_rnn import RNN_Flexible, RNNData


# Config and Params
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


params_data = Dict(read_yml(osp.join(CONFIG_DIR, "params_data.yaml")))
params_models = Dict(read_yml(osp.join(CONFIG_DIR, "params_models.yaml")))
n_features = len(params_data['features_list'])


if __name__ == '__main__':

    if len(sys.argv) != 4:
        print(f"Invalid arguments. {len(sys.argv)} was given but 4 expected")
        print(('Usage: %s <forecast_dir> <data_dir>' '<config_path>' % sys.argv[0]))
        print("<forecast_dir> is where outputs from the forecasts are sent. <data_dir> is where data for analysis lives. <config_path> is path to yaml file setting up time frame and other analysis parameters")
        print("Example: python src/forecast_analysis_setup.py forecasts/fmc_forecast_test data/rocky_fmda etc/forecast_analysis.yaml")
        sys.exit(-1)

    # Get input args
    f_dir = sys.argv[1]
    data_dir = sys.argv[2]
    conf_path = sys.argv[3]
    forecast_config = read_yml(conf_path)
    os.makedirs(f_dir, exist_ok=True)
    # Write copy of forecast config file to forecast directory, do this so multiple tests can be run with different input config files
    with open(osp.join(f_dir, "forecast_config.yaml"), 'w') as f:
        yaml.dump(forecast_config, f, default_flow_style=False)
    forecast_config = Dict(forecast_config)

    # Check if already run, allows for easy rerun of process
    if osp.exists(osp.join(f_dir, 'ml_data.pkl')) and osp.exists(osp.join(f_dir, 'analysis_info.json')):
        print(f"Forecast analysis setup already run at {f_dir}, exiting")
        sys.exit(0)

    # Get analysis run configuration
    fstart = str2time(forecast_config.start_time)
    fend = str2time(forecast_config.end_time)
    FORECAST_HOURS = params_data.forecast_hours
    TRAIN_HOURS = params_data.train_hours

    # Handle Forecast Periods
    # Define Forecast start times, 48hr spacing
    forecast_periods = time_range(
        start = fstart,
        end = fend,
        freq = "2d"
    )

    print("~"*75)
    print(f"Running Forecast Analysis from {fstart} to {fend}")
    print(f"Analysis Params: ")
    print(f"    {FORECAST_HOURS=}")
    print(f"    {TRAIN_HOURS=}")
    print(f"Total Forecast Periods: {forecast_periods.shape[0]}")
    print(f"Earliest forecast start: {forecast_periods.min()}")
    print(f"Latest forecast start: {forecast_periods.max()}")


    # Get FMDA Data Paths based on CV Scheme
    # 48hr test periods, 1 year train (incl 48 hour validation)
    # Assuming structure of pickle files for 1 day of some bbox
    # Format: {data_dir}/YYYYMM/fmda_YYYYMMDD.pkl

    print("~"*75)
    print("Defining CV time periods based on earliest and latest forecast times")
    t0 = forecast_periods.min() - relativedelta(hours=TRAIN_HOURS)
    t1 = forecast_periods.max() + relativedelta(hours=FORECAST_HOURS-1)  # subtract one for count from zero
    days = time_range(t0, t1, freq="1d")
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
    data = data_funcs.combine_fmda_files(file_paths)
    ml_dict = data_funcs.build_ml_data(data, hours=params_data.hours, max_linear_time = params_data.max_linear_time, verbose=False)
    df_valid = pd.read_csv(osp.join(PROJECT_ROOT, params_data.valid_path))
    ml_dict = data_funcs.remove_invalid_data(ml_dict, df_valid)
    data_file =  osp.join(PROJECT_ROOT, f_dir, "ml_data.pkl")
    with open(data_file, 'wb') as f:
        pickle.dump(ml_dict, f)

    # Write output file storing configuration
    info = {
        'forecast_periods': forecast_periods.shape[0],
        'earliest_forecast_period': forecast_periods.min().strftime('%Y%m%d_%H'),
        'latest_forecast_period': forecast_periods.max().strftime('%Y%m%d_%H'),
        'data_input_dir': data_dir
    }    

    info_file = osp.join(f_dir, 'analysis_info.json')
    with open(info_file, "w") as json_file:
        json.dump(info, json_file) 


