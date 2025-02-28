# Script to setup directory and configuration text files for hyperarameter tuning
# And creates RNNData necessary for analysis
# Based on model hyperparameter config file in etc/, create a directory and text files
# that correspond to the model grid and optimization grid
# The files model_grid.txt and opt_grid.txt have a line with a dictionary-style format on each line that can 
# be passed into a params dictionary for use in model training
# Text files are used in executables to set up number of CPU tasks
# 2 stage tuning: tune model architecture (num layers, units, layer type) THEN tune optimization params (batch size, learning rate)
# Other params fixed, see etc/params_models.yaml for more hyperparams and 
# moisture_rnn/model_grid for info on model architecture constraints

import sys
import os
import os.path as osp
import numpy as np
from dateutil.relativedelta import relativedelta
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
from models.moisture_rnn import model_grid, optimization_grid
import reproducibility
import data_funcs

params_data = Dict(read_yml(osp.join(CONFIG_DIR, "params_data.yaml")))
params_rnn = Dict(read_yml(osp.join(CONFIG_DIR, "params_models.yaml"), subkey="rnn"))
hyper_params = Dict(read_yml(osp.join(CONFIG_DIR, "rnn_hyperparam_tuning_config.yaml")))

features_list = ['Ed', 'Ew', 'elev', 'wind', 'solar', 'grid_x', 'grid_y', 'rain']


def write_txt(lst, outpath):
    """
    Write text file with list input, output text file number of lines should match list length
    """
    with open(outpath, 'w') as f:
        f.write('\n'.join(map(str, lst))+'\n')

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print(f"Invalid arguments. {len(sys.argv)} was given but 3 expected")
        print(('Usage: %s <model_dir> <data_dir>' % sys.argv[0]))
        print("<model_dir> is where outputs from the hyperparam tuning are sent. <data_dir> is where data for analysis lives")
        print("Example: python src/rnn_hyperparam_setup.py outputs/rnn_hyperparam_tuning_test")
        sys.exit(-1)

    # Get input args
    output_dir = sys.argv[1]
    data_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    
    # Time setup
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
                                   verbose=False, save_path = osp.join(output_dir, 'ml_data.pkl'))
    print(f"Total Stations with Data in Time Period: {len(ml_dict)}")


    # Param Grids    
    model_params_grid = model_grid(hyper_params['model_architecture'])
    opt_grid = optimization_grid(hyper_params['optimization'])

    print(f"Model Architecture Hyperparam Grid: {len(model_params_grid)} models")
    # print(model_params_grid)
    print(f"Optimization Hyperpam Grid: {len(opt_grid)} param combos")
    # print(opt_grid)

    # Write out
    write_txt(model_params_grid, osp.join(output_dir, "model_grid.txt"))
    write_txt(opt_grid, osp.join(output_dir, "opt_grid.txt"))

    



