# Evaluate Models from hyperparam tuning

# Two step tuning:
#    1) Tune model architecture: num layers, num units
#    2) Tune optimization parameters: batch_size, learning_rate

# Same script intended to be run twice, after each of steps 1) and 2) above
# Logical checks to see if output files exist in the script

import numpy as np
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

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print(f"Invalid arguments. {len(sys.argv)} was given but 2 expected")
        print(('Usage: %s <model_dir>' % sys.argv[0]))
        print("Example: python src/rnn_hyperparam_eval.py models/rnn_hyperparam_tuning_test")
        sys.exit(-1)

    model_dir = sys.argv[1]
    err_dir = osp.join(model_dir, 'model_errors')

    # Read model grid and opt grid
    # Read model grid and get task_id row
    file_path = osp.join(model_dir, "model_grid.txt")
    with open(file_path, "r") as file:
        models = file.readlines()
    # Read model grid and get task_id row
    file_path = osp.join(model_dir, "opt_grid.txt")
    with open(file_path, "r") as file:
        opts = file.readlines()        

    # Get Model Architecture if not already run
    out_file = osp.join(model_dir, "Final_Hyperparams.txt")
    if not osp.exists(out_file):
        print("Getting model architecture with min err")

        # Get minimum error model architecture 
        # Read files, extract rmse for each time period, write csv output, calc mean err, extract model from min err
        files = [f for f in os.listdir(err_dir) if f.startswith('model_')]
        # arrange by model number (shouldn't be necessary, but doing for clarity)
        # files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        assert len(files) == len(models), "Number of model error files {len(files)} not equal to number of model architectures in model_grid.txt {len(models)}"
        err_list = [read_pkl(osp.join(err_dir, f)) for f in files]
        rmse_dict = {
            f"model_{model['id']}": {date: values["rmse"] for date, values in model["errs"].items()}
            for model in err_list
        }        
        df = pd.DataFrame(rmse_dict).sort_index()
        df.to_csv(osp.join(model_dir, 'model_err_df.csv'))
        mean_errs = df.mean(axis = 0)
        min_err_index = mean_errs.argmin()
        min_err_model = err_list[min_err_index]['model'] 
        min_err_model.update({'id': err_list[min_err_index]['id']})
        print(f"Model Architecture Summary:")
        print(f"    Minimum Error Model: {min_err_index}")
        print(f"    Minimum Error: {mean_errs.min()}")

        # Write file of architecture
        with open(out_file, "w") as f:
            f.write(str(min_err_model) + "\n")
    
    # Get Optimization params with min error, but check if outputs have been created first
    else:
        err_dir = osp.join(model_dir, 'opt_errors')
        files = [f for f in os.listdir(err_dir) if f.startswith('opt_')]
        # Arrange by number (shouldn't be necessary, but for clarity)
        files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        if len(files) < len(opts):
            print("Optmization param search not run yet, Exiting script")
            sys.exit(0)
        elif len(files) > len(opts):
            print("More output files for optimziation params than configurations found in opt_grid.txt, exiting")
            sys.exit(1)
        else:
            print("Getting optmization params with min err")
            # Get minimum error optmization params
            # Read files, extract rmse for each time period, write csv output, calc mean err, extract opt from min err
            # arrange by opt number
            err_list = [read_pkl(osp.join(err_dir, f)) for f in files]            
            rmse_dict = {
                f"opt_{opt['id']}": {date: values["rmse"] for date, values in opt["errs"].items()}
                for opt in err_list
            }

            df = pd.DataFrame(rmse_dict).sort_index()
            df.to_csv(osp.join(model_dir, 'opt_err_df.csv'))
            mean_errs = df.mean(axis = 0)
            min_err_index = mean_errs.argmin()
            min_err_opt = err_list[min_err_index]['opt']
            min_err_opt.update({'id': err_list[min_err_index]['id']})
            print(f"Optimization Params Summary:")
            print(f"    Minimum Error Model: {min_err_index}")
            print(f"    Minimum Error: {mean_errs.min()}")

            # Write file of opt params, append to existing file
            with open(out_file, "a") as f:
                f.write(str(min_err_opt) + "\n")

