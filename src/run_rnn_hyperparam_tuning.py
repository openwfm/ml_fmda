# Run restricted grid search hyperparameter selection
# NOTE: In this script we are forecasting "validation data",
# in the sense that it is data not used for fitting model params
# but used for tuning hyperparameters. Programatically, it ends uip
# being the "test" data, but it is validation data with respect to 
# the main project analysis. None of the data used for the main project
# test error should be used in this step

# Two step tuning:
#    1) Tune model architecture: num layers, num units
#    2) Tune optimization parameters: batch_size, learning_rate

import numpy as np
from sklearn.model_selection import ParameterGrid
from itertools import product
import sys
import os
import os.path as osp
from dateutil.relativedelta import relativedelta
import pickle
import pandas as pd


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
import reproducibility

# Read Config Files
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

params_data = Dict(read_yml(osp.join(CONFIG_DIR, "params_data.yaml")))
params_rnn = Dict(read_yml(osp.join(CONFIG_DIR, "params_models.yaml"), subkey="rnn"))
hyper_params = Dict(read_yml(osp.join(CONFIG_DIR, "rnn_hyperparam_tuning_config.yaml")))


# code will loop through forecast period, then loop through model architecture 
# as determined by hyperparam tuning config file
# Outputs organized by Forecast Period start on top level key, then model configs as subkeys

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print(f"Invalid arguments. {len(sys.argv)} was given but 3 expected")
        print(('Usage: %s <fmda_dir_path> <output_dir>' % sys.argv[0]))
        print("Example: python src/run_rnn_hyperparam_tuning.py  'data/rocky_fmda' outputs/rnn_hyperparam_tuning")
        sys.exit(-1)

    

    # Paths
    data_dir = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    # Param Grids    
    model_params_grid = model_grid(hyper_params['model_architecture'])
    opt_grid = optimization_grid(hyper_params['optimization'])
    
    
    forecast_periods = hyper_params["times"]["forecast_start_times"]
    forecast_periods = np.array([str2time(t) for t in forecast_periods])
    train_hours = hyper_params["times"]["train_hours"]
    forecast_hours = hyper_params["times"]["forecast_hours"]

    print("~"*75)
    print(f"Running Hyperparameter Selection on {len(forecast_periods)} dates")
    print(f"    Forecast Times: {forecast_periods}")
    print(f"Analysis Time Params: ")
    print(f"    {forecast_hours=}")
    print(f"    {train_hours=}")    
    print(f"Using Hyperparam grid from {osp.join(CONFIG_DIR, 'rnn_hyperparam_tuning_config.yaml')}")
    print(f"Using other RNN params from {osp.join(CONFIG_DIR, 'params_models.yaml')}")

    # Setup Data
    t0 = forecast_periods.min() - relativedelta(hours=train_hours)
    t1 = forecast_periods.max() + relativedelta(hours=forecast_hours)
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

    reproducibility.set_seed(123)
    data = data_funcs.combine_fmda_files(file_paths)
    ml_dict = data_funcs.build_ml_data(data, hours=params_data.hours, 
                                   max_linear_time = params_data.max_linear_time, 
                                   verbose=False)
    print(f"Total Stations with Data in Time Period: {len(ml_dict)}")

    # Run Model Architecture Tuning
    print("~"*75)
    print(f"Running Model Architecture Hyperparameter Selection with param grids: {hyper_params['model_architecture']}")

    print("~"*75)
    for ft in forecast_periods:
        out_file = osp.join(out_dir, f"model_{ft.strftime('%Y%m%d_%H')}.pkl")
        if osp.exists(out_file):
            print("Forecast output already exists, skipping to next period")
        else:
            print(f"Running model architecture selection for forecast time {ft}") 
            print("Defining CV time periods based on time params")
            train, val, test = data_funcs.cv_data_wrap(ml_dict, ft, train_hours=train_hours,forecast_hours=forecast_hours)
            # Make RNN Data, reused by different models
            dat = RNNData(train, val, test, timesteps=48, method="random")
            dat.scale_data()

            # Setup output file for the forecast period
            err_dict_output = {}

            for i in range(0, len(model_params_grid)):
                print("~"*75)
                print(i)
                # Setup params
                model_i = model_params_grid[i]
                print(f"Running model configuration: {model_i}")
                params_rnn.update(model_i)
                
                # Run Train and Predict
                rnn = RNN_Flexible(n_features = dat.n_features, params = params_rnn)
                rnn.fit(dat.X_train, dat.y_train, 
                        validation_data=(dat.X_val, dat.y_val),
                        batch_size = params_rnn["batch_size"],
                        # epochs = params_rnn["epochs"],
                        epochs = 3,
                        verbose_fit = True, plot_history=False
                       )
                errs = rnn.test_eval(dat.X_test, dat.y_test)
                print(errs)

                # Save to output file
                err_dict_output[i] = {
                    'model': model_i,
                    'errs': errs
                }

            # Write output for forecast period
            print(f"Writing Output: {out_file}")
            with open(out_file, 'wb') as handle:
                pickle.dump(err_dict_output, handle, protocol=pickle.HIGHEST_PROTOCOL)      

    # Combine Errors and Evaluate Model Architecture
    files = [f for f in os.listdir(out_dir) if f.startswith('model_')]
    assert len(files) == len(forecast_periods), f"Mismatch number of output files ({len(files)}) and forecast periods ({len(forecast_periods)})"
    err_dicts = [read_pkl(osp.join(out_dir, f)) for f in files]
    data = [{ key: file_dict[key]['errs']['rmse'] for key in file_dict } for file_dict in err_dicts]
    df = pd.DataFrame(data)
    mean_errs = df.mean(axis = 0)
    min_err_index = mean_errs.argmin()
    min_err_model = err_dicts[0][min_err_index] # NOTE: use first element of data files, since architectures should be the same
    print("~"*75)
    print(f"Minimum Overall RMSE by Model: {mean_errs.min()}")
    print(f"Model Architecture with Min Err: {min_err_model['model']}")

    # Fix model architecture run optimization grid
    print("~"*75)
    print(f"Running Optimization Hyperparam Tuning with {opt_grid}")
    params_rnn.update(min_err_model["model"])
    for ft in forecast_periods:
        out_file = osp.join(out_dir, f"opt_{ft.strftime('%Y%m%d_%H')}.pkl")
        if osp.exists(out_file):
            print("Forecast output already exists, skipping to next period")
        else:
            print(f"Running model architecture selection for forecast time {ft}") 
            print("Defining CV time periods based on time params")
            train, val, test = data_funcs.cv_data_wrap(ml_dict, ft, train_hours=train_hours,forecast_hours=forecast_hours)
            # Make RNN Data, reused by different models
            dat = RNNData(train, val, test, timesteps=48, method="random")
            dat.scale_data()

            # Setup output file for the forecast period
            err_dict_output = {}            
            for i in range(0, len(opt_grid)):
                print("~"*75)
                print(i)
                # Setup params
                opt_i = opt_grid[i]
                print(f"Running optimization hyperparam configuration: {opt_i}")
                params_rnn.update(opt_i)
                
                # Run Train and Predict
                rnn = RNN_Flexible(n_features = dat.n_features, params = params_rnn)
                rnn.fit(dat.X_train, dat.y_train, 
                        validation_data=(dat.X_val, dat.y_val),
                        batch_size = params_rnn["batch_size"],
                        # epochs = params_rnn["epochs"],
                        epochs = 3,
                        verbose_fit = True, plot_history=False
                       )
                errs = rnn.test_eval(dat.X_test, dat.y_test)
                print(errs)

                # Save to output file
                err_dict_output[i] = {
                    'opt': opt_i,
                    'errs': errs
                }

            # Write output for forecast period
            print(f"Writing Output: {out_file}")
            with open(out_file, 'wb') as handle:
                pickle.dump(err_dict_output, handle, protocol=pickle.HIGHEST_PROTOCOL)  

    # Combine Errors and Evaluate Optimization Params
    files = [f for f in os.listdir(out_dir) if f.startswith('opt_')]
    assert len(files) == len(forecast_periods), f"Mismatch number of output files ({len(files)}) and forecast periods ({len(forecast_periods)})"
    err_dicts = [read_pkl(osp.join(out_dir, f)) for f in files]
    data = [{ key: file_dict[key]['errs']['rmse'] for key in file_dict } for file_dict in err_dicts]
    df = pd.DataFrame(data)
    mean_errs = df.mean(axis = 0)
    min_err_index = mean_errs.argmin()
    min_err_opt = err_dicts[0][min_err_index] # NOTE: use first element of data files, since architectures should be the same
    print("~"*75)
    print(f"Minimum Overall RMSE by Optimization Hyperparams: {mean_errs.min()}")
    print(f"Optimization Hyperparams with Min Err: {min_err_opt['opt']}")

    # Write file of tuned params
    out_file = osp.join(out_dir, "Final Hyperparams.txt")
    with open(out_file, "w") as f:
        f.write(str(min_err_model["model"]) + "\n" + str(min_err_opt["opt"]))

    