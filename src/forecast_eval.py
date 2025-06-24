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
import data_funcs

# Module Code
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


def summary_table(df, group_vars, bound_vars="rep"):
    """
    """
    bias = (
        df.groupby(group_vars, sort=False)["residual"]
        .mean()
        .reset_index(name="bias")
    )
    mse = (
        df.groupby(group_vars, sort=False)["squared_error"]
        .mean()
        .reset_index(name="mse")
    )
    rep_metrics = pd.merge(bias, mse, on=group_vars)
    group_vars.remove(bound_vars)
    summary_stats = rep_metrics.groupby(group_vars, sort=False)[["bias", "mse"]].agg(["mean", "std"]).reset_index() 
    if not type(bound_vars) is list:
        bound_vars = [bound_vars]
    summary_stats.columns = group_vars + ["bias_mean", "bias_std", "mse_mean", "mse_std"]
    #summary_stats["Bias"] = summary_stats["bias_mean"].astype('float64').round(2).astype(str) + " +/- " + summary_stats["bias_std"].round(2).astype(str)
    #summary_stats["MSE"] = summary_stats["mse_mean"].astype('float64').round(2).astype(str) + " +/- " + summary_stats["mse_std"].round(2).astype(str)
    #summary_formatted = summary_stats[["Model", "Bias", "MSE"]]
    #return summary_formatted
    return summary_stats

# Dictionary used to calculate error summary stats, based on given grouping
agg_named = {
    'mean_error': pd.NamedAgg(column='residual', aggfunc='mean'),
    'mean_squared_error': pd.NamedAgg(column='squared_error', aggfunc='mean'),
}

# Executed Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print(f"Invalid arguments. {len(sys.argv)} was given but 2 expected")
        print(('Usage: %s <model_dir>' % sys.argv[0]))
        print("Example: python src/forecast_eval.py forecasts/fmc_forecast_test/")
        sys.exit(-1)

    f_dir = sys.argv[1]
    fconf = Dict(read_yml(osp.join(f_dir, "forecast_config.yaml")))
    out_dir = osp.join(f_dir, "error_analysis")
    os.makedirs(out_dir, exist_ok=True)


    # Read output files for forecast analysis run
    ## Get all files in outputs
    files = os.listdir(osp.join(f_dir, 'forecast_outputs'))
    files = sorted(files, key=lambda x: int(re.search(r'_(\d+)\.h5$', x).group(1))) # Sort by task number, shouldn't be necessary but for clarity
    ## Read and combine into dataframe, add indicator column for replication number from file name
    rnn = read_hdf_list(files, key="rnn")
    ode = read_hdf_list(files, key="ode")
    xgb = read_hdf_list(files, key="xgb")
    clim = read_hdf_list(files, key="clim")
    clim = clim[(~clim.preds.isna()) & (~clim.fm.isna())]

    # Evaluate Accuracy
    ## First calculate errors (residuals)
    rnn = calc_errs(rnn); rnn["Model"] = "RNN"
    ode = calc_errs(ode); ode["Model"] = "ODE"
    xgb = calc_errs(xgb); xgb["Model"] = "XGB"
    clim = calc_errs(clim); clim["Model"] = "CLIMATOLOGY"

    ## Combine all individual predictions so someone can easily reproduce summary tables that aggregate
    ## Add spatial info and calculate hour of dayi
    print(f"Combining and evaluating errors for forecast period:")
    print(f"    {fconf.f_start=}")
    print(f"    {fconf.f_end=}")

    df = pd.concat([rnn, xgb, ode, clim], ignore_index=True)
    ml_dict = read_pkl(osp.join(f_dir, "ml_data.pkl"))
    loc_df = pd.DataFrame.from_dict(
        {k: v["loc"] for k, v in ml_dict.items()},
        orient="index"
    )
    del ml_dict


    # Data very big to write as h5, and only done as external double check on calculations. 
    # Use fperiod files directly for reproducing error calcs
    #print(f"Writing all forecast and errors to {osp.join(out_dir, 'all_errors.h5')}")
    #df.to_hdf(osp.join(out_dir, "all_errors.h5"), key="all_errors", mode="w", complib="blosc")
    loc_df.to_csv(osp.join(out_dir, "stid_locs.csv"), index=False)

    ## Overall Error, averaged over every hour, location, and replication
    ## Bounds from +/- 1std for replications
    table1 = summary_table(df, group_vars = ["Model", "rep"], bound_vars ="rep") 
    table1.to_csv(osp.join(out_dir, "overall.csv"), index=False)

    ## Error by Station, averaged over all times and replications. Bounds from reps
    table_st = summary_table(df, group_vars = ["Model", "stid", "rep"], bound_vars ="rep")
    table_st.to_csv(osp.join(out_dir, "by_stid.csv"), index=False)
    ## Error by hour of day (0-23)
    ## averaged over all stations and days and reps, fixed hour 0 at 00:00 UTC
    df["hod"] = pd.to_datetime(df.date_time).dt.hour
    table_hod = summary_table(df, group_vars = ["Model", "hod", "rep"], bound_vars ="rep")       
    table_hod.to_csv(osp.join(out_dir, "by_hod.csv"), index=False)

    ## Error by hour & day, averaged over all stations and reps
    table_t = summary_table(df, group_vars = ["Model", "date_time", "rep"], bound_vars ="rep")
    table_t.to_csv(osp.join(out_dir, "by_dt.csv"), index=False)



