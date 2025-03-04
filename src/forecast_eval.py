# Script to combine results from different models and analyze results
# Files organized by forecast period, read those in and summarize


import numpy as np
import sys
import os
import os.path as osp
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error
import re

# Set up project paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## We do this so the module can be imported from different locations
CURRENT_DIR = osp.abspath(__file__)
while osp.basename(CURRENT_DIR) != "ml_fmda":
    CURRENT_DIR = osp.dirname(CURRENT_DIR)
PROJECT_ROOT = CURRENT_DIR
CODE_DIR = osp.join(PROJECT_ROOT, "src")
sys.path.append(CODE_DIR)
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")
DATA_DIR = osp.join(PROJECT_ROOT, "data")

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import Dict, read_pkl, read_yml, str2time, time_range
from models.moisture_rnn import model_grid, optimization_grid, RNNData, RNN_Flexible
import data_funcs

forecast_config = Dict(read_yml(osp.join(CONFIG_DIR, "forecast_analysis.yaml")))

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print(f"Invalid arguments. {len(sys.argv)} was given but 2 expected")
        print(('Usage: %s <model_dir>' % sys.argv[0]))
        print("Example: python src/forecast_eval.py forecasts/fmc_forecast_test/")
        sys.exit(-1)

    f_dir = sys.argv[1]
    # Check output exists
    if osp.exists(osp.join(f_dir, 'forecast_errs.csv')) and osp.exists(osp.join(f_dir, 'forecast_summary.csv')):
        print(f"Output already exists at {f_dir}, exiting")
        sys.exit(0)

    # Read analysis config, used for QC checks that input actually matches target
    # Get analysis run configuration
    fstart = str2time(forecast_config.start_time)
    fend = str2time(forecast_config.end_time)
    # Define Forecast start times, 48hr spacing
    forecast_periods = time_range(
        start = fstart,
        end = fend,
        freq = "2d"
    )    



    # Set up files and read
    files = os.listdir(osp.join(f_dir, 'forecast_periods'))
    files = sorted(files, key=lambda x: int(re.search(r'_(\d+)\.pkl', x).group(1))) # Sort by task number, shouldn't be necessary but for clarity
    results = [read_pkl(osp.join(f_dir, 'forecast_periods', f)) for f in files]
    assert len(results) == len(forecast_periods), "Mismatch number of results files {len(results)} vs target forecast periods {len(forecast_periods)}"
    # Compare Models
    # Run some checks on time and location, combine results into a df
    ode_errs = []
    xgb_errs = []
    rnn_errs = []
    for i, fperiod in enumerate(results):
        stids = fperiod['stids']
        times = fperiod['times']
        times.sort()
        # Check times match, num stations matches
        assert pd.Timestamp(forecast_periods[i]) == times[0], "Time array from ML output dict doesn't match target file time"
        for mod in ['RNN']:
            assert len(fperiod[mod]['loc_rmse']) == len(stids), "Mismatch between number of stations and number of RMSE per station"
        ode_errs.append(fperiod['ODE']['rmse'])
        xgb_errs.append(fperiod['XGB']['rmse'])
        rnn_errs.append(fperiod['RNN']['rmse'])

    df = pd.DataFrame({
        'ODE': ode_errs,
        'XGB': xgb_errs,
        'RNN': rnn_errs,
    })
    df.index = forecast_periods
    print('~'*75)
    print(f"Writing Forecast Errors table to: {osp.join(f_dir, 'forecast_errs.csv')}")
    df.to_csv(osp.join(f_dir, 'forecast_errs.csv')) 
    # Mean Error for Model
    means = df.mean(axis=0)
    stds = df.std(axis=0)

    print(f"Writing Forecast Error Summary table to: {osp.join(f_dir, 'forecast_summary.csv')}")
    overall_errs_df = pd.DataFrame({"Mean RMSE": means, "(Std)": stds})
    overall_errs_df.to_csv(osp.join(f_dir, 'forecast_summary.csv'))
    print(overall_errs_df)


