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
import re
import ast
import yaml

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

# Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def read_hdf_list(file_list, key):
    """
    Given a list of hdf files and a dataset key, read all with pandas and merge
    
    NOTE: assumes certain naming structure for file names and  the line that adds the column "rep"
    """
    data = [pd.read_hdf(f, key=key).assign(model=int(re.search(r'_(\d+)\.h5$', f).group(1))) for f in files]
    data = pd.concat(data, ignore_index=True)
    return data

def calc_errs(df, pred_col="preds", true_col="fm"):
    """
    Adds residual, absolute error, and squared error columns to a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing prediction and true value columns.
    - pred_col (str): Name of the column with predicted values. Default is "preds".
    - true_col (str): Name of the column with true/observed values. Default is "fm".

    Returns:
    - pd.DataFrame: The same DataFrame with new error columns added:
        'residual', 'abs_error', 'squared_error'
    """
    residual = df[true_col] - df[pred_col]
    df["residual"] = residual
    df["abs_error"] = residual.abs()
    df["squared_error"] = residual ** 2
    return df

# Dictionary used to calculate error summary stats, based on given grouping
agg_named = {
    'mean_error': pd.NamedAgg(column='residual', aggfunc='mean'),
    'median_error': pd.NamedAgg(column='residual', aggfunc='median'),
    'mean_absolute_error': pd.NamedAgg(column='abs_error', aggfunc='mean'),
    'mean_squared_error': pd.NamedAgg(column='squared_error', aggfunc='mean'),
    'n_predictions': pd.NamedAgg(column='residual', aggfunc='count')
}

# Executed Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print(f"Invalid arguments. {len(sys.argv)} was given but 2 expected")
        print(('Usage: %s <model_dir>' % sys.argv[0]))
        print("Example: python src/rnn_hyperparam_eval.py models/rnn_hyperparam_tuning_test")
        sys.exit(-1)

    model_dir = sys.argv[1]

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
    out_file = osp.join(model_dir, "Final_Architecture.txt")
    
    if not osp.exists(out_file):
        print("Getting model architecture with min err")
        # Read output files for model architecture tune run
        ## Get all files in outputs
        files = [osp.join(model_dir, 'model_outputs', fname) for fname in os.listdir(osp.join(model_dir, 'model_outputs'))]
        files = sorted(files, key=lambda x: int(re.search(r'_(\d+)\.h5$', x).group(1))) # Sort by task number, shouldn't be necessary but for clarity        
        df = read_hdf_list(files, key="rnn")

        # Calculate overall model error, write output then select minimum error architecture
        errs = calc_errs(df)    
        summary = errs.groupby(["model"]).agg(**agg_named).reset_index() 
        summary.to_csv(osp.join(model_dir, "model_errors.csv")) 
   
        print(f"Extracting Minimum Error")
        ind = int(summary.mean_squared_error.argmin())
        print(f"    Min Model MSE: {summary.mean_squared_error.min()}")
        print(f"    Model ID Number: {ind}")
        model_dict = ast.literal_eval(models[ind])
        model_dict.update({'id': ind})
        print(f"    Model Architecture: {model_dict}")
        # Write file of architecture
        with open(out_file, "w") as f:
            f.write(str(model_dict) + "\n")
    else:
        print(f"Getting Optimization params with min err")
        # Read output files for optmization tune run
        ## Get all files in outputs
        files = [osp.join(model_dir, 'opt_outputs', fname) for fname in os.listdir(osp.join(model_dir, 'opt_outputs'))]
        files = sorted(files, key=lambda x: int(re.search(r'_(\d+)\.h5$', x).group(1))) # Sort by task number, shouldn't be necessary but for clarity        
        df = read_hdf_list(files, key="rnn")
        # Calculate overall model error, write output then select minimum error architecture
        errs = calc_errs(df)
        summary = errs.groupby(["model"]).agg(**agg_named).reset_index()
        summary.to_csv(osp.join(model_dir, "opt_errors.csv"))

        print(f"Extracting Minimum Error")
        ind = int(summary.mean_squared_error.argmin())
        print(f"    Min Model MSE: {summary.mean_squared_error.min()}")
        print(f"    Opt ID Number: {ind}")
        opt_dict = ast.literal_eval(opts[ind])
        opt_dict.update({'id': ind})
        print(f"    Opt Params: {opt_dict}")
        # Append output file with optimization params
        with open(out_file, "a") as f:
            f.write(str(opt_dict) + "\n")



