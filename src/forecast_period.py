# Script to run a single iteration of model forecasting given a forecast period
# Runs models, including ODE, XGBoost, and RNN on given forecast period. Save to input directory
# Forecast periods are assigned out with slurm array

import os.path as osp
import sys
import ast
import numpy as np
import pickle
import gc
import h5py

# Set up project paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CURRENT_DIR = osp.dirname(osp.normpath(osp.abspath(__file__)))
PROJECT_ROOT = osp.dirname(osp.normpath(CURRENT_DIR))
sys.path.append(osp.join(PROJECT_ROOT, "src"))
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import Dict, read_pkl, read_yml, str2time, time_range
from models.moisture_rnn import model_grid, optimization_grid, RNNData, RNN_Flexible
import data_funcs
import reproducibility
from models.moisture_static import XGB
from models.moisture_ode import ODE_FMC
from models.moisture_rnn import RNN_Flexible, RNNData


# Config and metadata files
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

params_data = Dict(read_yml(osp.join(CONFIG_DIR, "params_data.yaml")))
params_models = Dict(read_yml(osp.join(CONFIG_DIR, "params_models.yaml")))
features_list = params_data['features_list']

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Invalid arguments. {len(sys.argv)} was given but 3 expected")
        print(f"Usage: {sys.argv[0]} <task_id> <directory>")
        print("Example: python forecast_period.py 5 forecasts/fmc_forecast_test")
        sys.exit(-1)

    # Get model architecture from slurm task array
    task_id = int(sys.argv[1])
    f_dir = sys.argv[2]
    print(f"Running task {task_id}")    

    # Read forecast config file, it is created during setup py script
    forecast_config = Dict(read_yml(osp.join(f_dir, "forecast_config.yaml")))

    # Check if output already exists, exit if so.
    # Allows for running multiple times if process stops for any reason
    # Requires manual deletion of old files if you want to rerun
    out_dir = osp.join(f_dir, 'forecast_periods')
    out_file = osp.join(out_dir, f"fperiod_output_{task_id}.h5")    

    if osp.exists(out_file):
        print(f"Output for task {task_id} already exists at: {out_file}, exiting")
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

    ft = forecast_periods[task_id-1]
    print("~"*75)
    print(f"Running Forecast Analysis for period {ft}")
    print(f"Forecast config file: {osp.join(CONFIG_DIR, 'forecast_analysis.yaml')}")
    print(f"Analysis Params: ")
    print(f"    {FORECAST_HOURS=}")
    print(f"    {TRAIN_HOURS=}")

    # Get needed data
    # Split train/val/test, use task_id for repro seed
    ml_data = read_pkl(osp.join(f_dir, 'ml_data.pkl'))
    reproducibility.set_seed(task_id)
    train, val, test = data_funcs.cv_data_wrap(ml_data, ft, train_hours=TRAIN_HOURS,forecast_hours=FORECAST_HOURS)

    # Run Models
    # ODE
    print('~'*75)
    print("Running ODE")
    params = params_models['ode']
    te_sts = [*test.keys()]
    test_times = test[te_sts[0]]["times"]
    ode_data = data_funcs.get_ode_data(ml_data, te_sts, test_times)
    ode = ODE_FMC(params=params)
    m_ode, errs_ode = ode.run_model(ode_data, hours=72, h2=24)
    # Clear up space
    del ode_data
    gc.collect()

    ## Static XGBoost
    print('~'*75)
    print("Running XGB")
    params = params_models['xgb']
    dat = data_funcs.StaticMLData(train, val, test, features_list = features_list)
    dat.scale_data()
    xgb_model = XGB(params=params)
    xgb_model.fit(dat.X_train, dat.y_train)
    m_xgb = xgb_model.predict(dat.X_test)
    # Turn into 3d array using location labels in data, NOTE: this assumes equal number of predictions per location
    ## Check test locs are the same
    _, idx = np.unique(dat.test_locs, return_index=True)
    assert np.mean(dat.test_locs[np.sort(idx)] == te_sts) == 1
    m_xgb = m_xgb.reshape(len(te_sts), FORECAST_HOURS, 1)

    # Clear up space
    del dat
    gc.collect()

    # RNN
    print('~'*75)
    print('Running RNN')
    params = params_models['rnn']
    params.update({'timesteps': None}) # Allows for flexible sequence length
    dat = RNNData(train, val, test, method="random", timesteps=FORECAST_HOURS, random_state=None, features_list = features_list)
    dat.scale_data()
    rnn = RNN_Flexible(n_features=dat.n_features,params=params)
    rnn.fit(dat.X_train, dat.y_train,
            validation_data=(dat.X_val, dat.y_val),
            batch_size = params["batch_size"],
            epochs = params["epochs"],
            verbose_fit = True,
            plot_history=False
           )
    m_rnn = rnn.predict(dat.X_test)
    # Save 3d array of observed data for output
    y_test = dat.y_test

    # Clear up space
    del dat
    gc.collect()

    # Write output for forecast period
    output = {
        'times': [t.strftime('%Y-%m-%d %H:%M:%S') for t in test_times],
        'stids': te_sts,
        'fmc_observed': y_test,
        "ODE": m_ode, 
        "XGB": m_xgb, 
        "RNN": m_rnn
    }
    print(f"Writing Output: {out_file}")
    # Write as h5 object, pickles are too unstable and version specific
    with h5py.File(out_file, 'w') as h5f:
        for key, value in output.items():
            if np.isscalar(value):
                h5f.attrs[key] = value
            else:
                h5f.create_dataset(key, data=value)


