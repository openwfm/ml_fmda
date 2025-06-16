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
CURRENT_DIR = osp.dirname(osp.normpath(osp.abspath(__file__)))
PROJECT_ROOT = osp.dirname(osp.normpath(CURRENT_DIR))
sys.path.append(osp.join(PROJECT_ROOT, "src"))
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import Dict, read_pkl, read_yml, str2time, time_range
from models.moisture_rnn import model_grid, optimization_grid, RNNData, RNN_Flexible
import data_funcs

fconf = Dict(read_yml(osp.join(CONFIG_DIR, "forecast_config.yaml")))

# Module Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def read_hdf_list(file_list, key):
    """
    Given a list of hdf files and a dataset key, read all with pandas and merge
    
    NOTE: assumes certain naming structure for file names and  the line that adds the column "rep"
    """
    data = [pd.read_hdf(osp.join(f_dir, "forecast_outputs", f), key=key).assign(rep=int(re.search(r'_(\d+)\.h5$', f).group(1))) for f in files]
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

def overall_errs(df):
    """
    Returns a summary of error metrics from a DataFrame containing
    residual, abs_error, and squared_error columns. Error is aggregated over
    all times and locations

    Metrics:
    - Mean Error (bias)
    - Median Error
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - Number of predictions
    """
    summary = {
        "mean_error": df["residual"].mean(),
        "median_error": df["residual"].median(),
        "mean_absolute_error": df["abs_error"].mean(),
        "mean_squared_error": df["squared_error"].mean(),
        "n_predictions": df["residual"].count()
    }
    return pd.DataFrame([summary])


def stid_errs(df, stid_col="stid"):
    """
    Returns a summary table of error metrics for each unique location (STID), so averaged over all times

    Parameters:
    - df (pd.DataFrame): DataFrame with error columns and an STID column.
    - stid_col (str): Name of the column with station/location IDs.

    Returns:
    - pd.DataFrame: One row per STID with aggregated error metrics.
    """
    grouped = df.groupby(stid_col).agg(
        mean_error=("residual", "mean"),
        median_error=("residual", "median"),
        mean_absolute_error=("abs_error", "mean"),
        mean_squared_error=("squared_error", "mean"),
        n_predictions=("residual", "count")
    ).reset_index()

    return grouped

def hod_errs(df, t_col="date_time"):
    """
    Returns a summary table of error metrics for each hour of the day (0-23), so aggregated over STID and day of year

    Parameters:
    - df (pd.DataFrame): DataFrame with error columns and an STID column.
    - t_col (str): Name of the column with station/location IDs.

    Returns:
    - pd.DataFrame: One row per STID with aggregated error metrics.
    """
    if type(df[t_col][0]) is str:
        # Convert to dt object
        df[t_col] = str2time(df[t_col].tolist())
    df["hod"] = df[t_col].dt.hour
    grouped = df.groupby("hod").agg(
        mean_error=("residual", "mean"),
        median_error=("residual", "median"),
        mean_absolute_error=("abs_error", "mean"),
        mean_squared_error=("squared_error", "mean"),
        n_predictions=("residual", "count")
    ).reset_index()

    return grouped

def h_errs(df, t_col="date_time"):
    """
    Returns a summary table of error metrics for each hour AND day, so averaged over STID only.

    Parameters:
    - df (pd.DataFrame): DataFrame with error columns and an STID column.
    - stid_col (str): Name of the column with station/location IDs.

    Returns:
    - pd.DataFrame: One row per STID with aggregated error metrics.
    """
    if type(df[t_col]) is str:
        # Convert to dt object
        df[t_col] = str2time(df[t_col].tolist())
    grouped = df.groupby(t_col).agg(
        mean_error=("residual", "mean"),
        median_error=("residual", "median"),
        mean_absolute_error=("abs_error", "mean"),
        mean_squared_error=("squared_error", "mean"),
        n_predictions=("residual", "count")
    ).sort_values(t_col).reset_index()

    return grouped

# Executed Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print(f"Invalid arguments. {len(sys.argv)} was given but 2 expected")
        print(('Usage: %s <model_dir>' % sys.argv[0]))
        print("Example: python src/forecast_eval.py forecasts/fmc_forecast_test/")
        sys.exit(-1)

    f_dir = sys.argv[1]
    
    
    ## Check output exists
    "..."

    # Read output files for analysis run
    ## Get all files in outputs
    files = os.listdir(osp.join(f_dir, 'forecast_outputs'))
    files = sorted(files, key=lambda x: int(re.search(r'_(\d+)\.h5$', x).group(1))) # Sort by task number, shouldn't be necessary but for clarity
    ## Read and combine into dataframe, add indicator column for replication number from file name
    rnn = read_hdf_list(files, key="rnn")
    ode = read_hdf_list(files, key="ode")
    xgb = read_hdf_list(files, key="xgb")

    # Evaluate Accuracy
    ## First calculate errors (residuals)
    rnn = calc_errs(rnn)
    ode = calc_errs(ode)
    xgb = calc_errs(xgb)


    ## Overall Error, averaged over every hour and location
    rnn_errs = overall_errs(rnn); rnn_errs["Model"] = "RNN"
    ode_errs = overall_errs(ode); ode_errs["Model"] = "ODE"
    xgb_errs = overall_errs(xgb); xgb_errs["Model"] = "XGB"
    summary_table = pd.concat([rnn_errs, ode_errs, xgb_errs], ignore_index=True)

    ## Error by Station, averaged over all times
    rnn_errs = stid_errs(rnn); rnn_errs["Model"] = "RNN"
    ode_errs = stid_errs(ode); ode_errs["Model"] = "ODE"
    xgb_errs = stid_errs(xgb); xgb_errs["Model"] = "XGB"

    breakpoint()
    ## Error by hour of the day, averaged over all stations and days
    rnn_errs = hod_errs(rnn); rnn_errs["Model"] = "RNN"
    ode_errs = hod_errs(ode); ode_errs["Model"] = "ODE"
    xgb_errs = hod_errs(xgb); xgb_errs["Model"] = "XGB"

    ## Error by hour & day, averaged over all stations
    rnn_errs = h_errs(rnn); rnn_errs["Model"] = "RNN"
    ode_errs = h_errs(ode); ode_errs["Model"] = "ODE"
    xgb_errs = h_errs(xgb); xgb_errs["Model"] = "XGB"





    ## Climatology, extract needed periods and stations, then calculate MSE
    ## Climatology forecasts wont exist for certain stations, eg new stations without long climate history
    ## Test stations chosen by ones with data availability, so there should be minimal missing data for raws in the forecast periods
    clim_file = fconf.climatology_file
    clim = read_pkl(osp.join(PROJECT_ROOT, clim_file))
    raws_file = fconf.raws_file
    raws = read_pkl(osp.join(PROJECT_ROOT, raws_file))
    assert all(pd.Timestamp(dt) in clim.columns for dt in forecast_outputs), "Climatology missing some target forecast periods, can't make comparison"
    for i in range(0, len(results)):
        dat = results[i]
        clim_i = clim.loc[clim.index.isin(dat['stids']), clim.columns.isin(dat['times'])]
        clim_i = clim_i.reindex(dat['stids'])
        raws_i = data_funcs.get_sts_and_times(raws, dat['stids'], dat['times'], data_dict = 'RAWS')
        raws_i = pd.DataFrame({k: v["RAWS"]["fm"] for k, v in raws_i.items()}).T
        raws_i.columns = dat['times']
        raws_i = raws_i.reindex(dat['stids'])
        raws_i = raws_i.astype(np.float64) # ensure type match
        assert clim_i.shape[1] == raws_i.shape[1], "Column mismatch between raws and climatology for forecast period {i}, {forecast_period[i]}"
        sts = clim_i.index.intersection(raws_i.index) # get common row indices, corresponding to stations, these should be the same in most cases except brand new stations without clim history
        clim_i = clim_i.loc[sts]
        raws_i = raws_i.loc[sts]
        diff_i = raws_i.sub(clim_i) # calc residual
        diff_i = diff_i ** 2         # square difference
        # Add to results dict
        results[i]['CLIMATOLOGY'] = {
            'mse': diff_i.mean().mean(),
            'loc_mse': diff_i.mean(axis=1).to_numpy()
        }


