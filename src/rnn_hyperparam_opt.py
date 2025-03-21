# Script to run a single set of optimization params as part of a hyperparameter tuning
# a shell file calls this script and assigns a single CPU, allows for parallelization of hyperparam tuning
# Inputs: task ID that corresponds to a model configuration, directory where the model config lives
# Model config file should be `opt_grid.txt`
# Script will create a dictionary of model errors and write output as a pickle file

import os.path as osp
import sys
import ast
import numpy as np
import pickle

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

params_data = Dict(read_yml(osp.join(CONFIG_DIR, "params_data.yaml")))
params_rnn = Dict(read_yml(osp.join(CONFIG_DIR, "params_models.yaml"), subkey="rnn"))

features_list = params_data['features_list']

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Invalid arguments. {len(sys.argv)} was given but 3 expected")
        print(f"Usage: {sys.argv[0]} <task_id> <directory>")
        print(f"Example: python src/rnn_hyperparam_opt.py 2 model/rnn_hyperparam_tuning_test")
        print("NOTE: count from zero with the task_id")
        sys.exit(-1)

    task_id = int(sys.argv[1])
    model_dir = sys.argv[2]
    hyper_params = Dict(read_yml(osp.join(model_dir, "hyperparam_config.yaml")))
    print(f"Running task {task_id}")


    # Initial setup
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

    # Set up data and paths
    ml_data = read_pkl(osp.join(model_dir, "ml_data.pkl"))
    out_file = osp.join(model_dir, 'opt_errors', f"opt_{task_id}.pkl")
    err_dict_output = {
            'id': int(task_id),
            'file_name':out_file,
            'opt': opt_i,
            'errs': {}
             }

    # Read in Final tuned model architecture
    file_path = osp.join(model_dir, "Final_Hyperparams.txt")
    with open(file_path, "r") as file:
        model = file.readlines()[0] # model architecture params should be first line of output file
    # Parse to dict
    try:
        model = ast.literal_eval(model)
        if not isinstance(opt_i, dict):
            raise ValueError("Parsed row is not a dictionary")
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Failed to parse Final_Hyperparams.txt  as a dictionary: {e}")


    # Run model with target params over forecast periods
    params_rnn.update(model)
    params_rnn.update(opt_i)
    reproducibility.set_seed(task_id)
    for ft in forecast_periods:
        print('~'*75)
        print(f"Running forecast period {ft}")
        # Make data for period
        train, val, test = data_funcs.cv_data_wrap(ml_data, ft, train_hours=train_hours,forecast_hours=forecast_hours)
        dat = RNNData(train, val, test, timesteps=48, method="random", features_list=features_list)
        dat.scale_data()

        # Train and predict
        rnn = RNN_Flexible(n_features = dat.n_features, params = params_rnn)
        rnn.fit(dat.X_train, dat.y_train, 
                validation_data=(dat.X_val, dat.y_val),
                batch_size = params_rnn["batch_size"],
                epochs = params_rnn["epochs"],
                verbose_fit = True, plot_history=False
               )   
        errs = rnn.test_eval(dat.X_test, dat.y_test)
        print(errs)        
        # Save to output file
        err_dict_output['errs'][ft.strftime('%Y%m%d_%H')] = errs




    # Write output for opt
    print(f"Writing Output: {out_file}")
    with open(out_file, 'wb') as handle:
        pickle.dump(err_dict_output, handle, protocol=pickle.HIGHEST_PROTOCOL)




