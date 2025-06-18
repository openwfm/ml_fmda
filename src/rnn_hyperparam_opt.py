# Script to run a single set of optimization params as part of a hyperparameter tuning
# a shell file calls this script and assigns a single CPU, allows for parallelization of hyperparam tuning
# Inputs: task ID that corresponds to a model configuration, directory where the model config lives
# Model config file should be `opt_grid.txt`
# Script will create a dictionary of model errors and write output as a pickle file

import os
import os.path as osp
import sys
import ast
import numpy as np
import pickle
import pandas as pd
from dateutil.relativedelta import relativedelta

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

params_rnn = Dict(read_yml(osp.join(CONFIG_DIR, "params_models.yaml"), subkey="rnn"))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Invalid arguments. {len(sys.argv)} was given but 3 expected")
        print(f"Usage: {sys.argv[0]} <task_id> <directory>")
        print(f"Example: python src/rnn_hyperparam_opt.py 2 model/rnn_hyperparam_tuning_test")
        sys.exit(-1)

    task_id = int(sys.argv[1])
    model_dir = sys.argv[2]
    hconf = Dict(read_yml(osp.join(model_dir, "hyperparam_config.yaml")))

    features_list = hconf.features_list
    tstart = str2time(hconf.train_start)
    tend = str2time(hconf.train_end)
    fstart = str2time(hconf.f_start)
    fend = str2time(hconf.f_end)
    fhours = int(hconf.forecast_hours)

    # Check if output already exists, exit if so.
    # Allows for running multiple times if process stops for any reason
    # Requires manual deletion of old files if you want to rerun
    out_file = osp.join(model_dir, 'opt_outputs', f"opt_{task_id}.h5")
    if osp.exists(out_file):
        print(f"Output for task {task_id} already exists at: {out_file}, exiting")
        sys.exit(0)
    else:
        print(f"Running task {task_id}")    
    print("~"*75)
    print(f"Running a set of optimization params for Hyperparameter Selection")
    print(f"    Train Start: {tstart}")
    print(f"    Train End: {tend}")
    print(f"    Forecast Start: {fstart}")
    print(f"    Foreceast End: {fend}")

    # Get model architecture from previous tuning step
    model_path = osp.join(model_dir, "Final_Architecture.txt")
    breakpoint()
    with open(model_path, "r") as file:
        model = file.readlines()
    model = ast.literal_eval(model[0])
    print(f"Model Architecture: {model}")
    params_rnn.update(model)

    # Read opt grid and get task_id row
    file_path = osp.join(model_dir, "opt_grid.txt")
    with open(file_path, "r") as file:
        opt = file.readlines()

    # Ensure task_id is within bounds
    if not (0 <= task_id-1 < len(opt)):
        raise IndexError(f"task_id {task_id} is out of range. File has {len(opt)} lines.")
    row_i = opt[task_id-1].strip() # NOTE: -1 since slurm array counts from 1, python from 0
    # Parse to dict
    try:
        opt_i = ast.literal_eval(row_i)
        if not isinstance(opt_i, dict):
            raise ValueError("Parsed row is not a dictionary")
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Failed to parse row {task_id-1} as a dictionary: {e}")
    # Update RNN params with optimization parameters
    params_rnn.update(opt_i)

    # Set up data
    ml_data = read_pkl(osp.join(model_dir, "ml_data.pkl"))
    reproducibility.set_seed(task_id)
    train, val, test = data_funcs.cv_data_wrap(ml_data, fstart, fend, tstart, tend, val_hours=hconf.val_hours, test_frac = hconf.space_test_frac, random_state=task_id, all_test_times=False)
    # Define Forecast start times, 48hr spacing
    forecast_periods = time_range(
        start = fstart,
        end = fend,
        freq = f"{fhours}h"
    )

    # Train model once, reuse for forecast
    print('~'*75)
    print('Training RNN')
    params = params_rnn
    dat = RNNData(train, val, test=None, method="random", timesteps=fhours, random_state=None, features_list = features_list)
    dat.scale_data()
    rnn = RNN_Flexible(n_features=dat.n_features,params=params)
    rnn.fit(dat.X_train, dat.y_train,
            validation_data=(dat.X_val, dat.y_val),
            batch_size = params["batch_size"],
            epochs = params["epochs"],
            verbose_fit = True,
            plot_history=False
           )


    # Loop over forecast periods and predict
    # Looping ensures re-initialization of initial state
    column_types = {
        'preds': np.float64,
        'stid': str,
        'date_time': str,
        'fm': np.float64
    } # Used to construct output dataframes    
    rnn_output=pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in column_types.items()}) # initialize empty dataframe
    te_sts = [*test.keys()] # list of test stations
    for ft in forecast_periods:
        ts = time_range(ft, ft+relativedelta(hours = hconf.forecast_hours-1))
        # Extract needed times, remove stations with missing data
        test2 = data_funcs.get_sts_and_times(test, te_sts, ts)
        test2 = {k: v for k, v in test2.items() if v["data"].shape[0] == ts.shape[0]}
        # Small chance of no data for all stations sampled for test set within given period. 
        # NOTE: we get around this by running many replications, systematically searching for 
        # data availability is too inefficient 
        if len(test2) > 1:
            X_test = dat._combine_data(test2, features_list)
            sts = dat._combine_data(test2, ['stid'])
            y_test = dat._combine_data(test2, ['fm'])
            assert (X_test.shape[0] == len(test2)) and (X_test.shape[1]==ts.shape[0]) and (X_test.shape[0:2]==y_test.shape[0:2])
            # Run predictiona and format for output
            m_rnn = rnn.predict(X_test)
            df_temp = pd.DataFrame({'preds': m_rnn.flatten(), 'stid': sts.flatten(), 'date_time':np.tile(ts, m_rnn.shape[0]).astype(str), 'fm': y_test.flatten()})
            rnn_output = pd.concat([rnn_output, df_temp], ignore_index=True)

    # Write output
    os.makedirs(osp.join(model_dir, 'opt_outputs'), exist_ok=True)
    rnn_output.to_hdf(out_file, key="rnn", mode="w")













