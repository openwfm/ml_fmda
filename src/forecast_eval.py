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
        print("Example: python src/forecast_eval.py forecasts/fmc_forecast_test/")
        sys.exit(-1)

    f_dir = sys.argv[1]
    fconf = Dict(read_yml(osp.join(f_dir, "forecast_config.yaml")))
    out_dir = osp.join(f_dir, "error_analysis")
    os.makedirs(out_dir, exist_ok=True)

    # Read and format climatology outputs.
    ## NOTE: climatology run separately from other models since so different, no train and test sets
    ## Climatology, extract needed periods and stations, then calculate MSE
    ## Climatology forecasts wont exist for certain stations, eg new stations without long climate history
    clim_file = fconf.climatology_file
    clim = read_pkl(osp.join(PROJECT_ROOT, clim_file))
    raws_file = osp.join(PROJECT_ROOT, "data", "raws_rocky_2024.pkl")
    raws = read_pkl(osp.join(PROJECT_ROOT, raws_file))

    ## Filter by test times
    test_times = pd.to_datetime(time_range(fconf.f_start, fconf.f_end))
    assert all(pd.Timestamp(dt) in clim.columns for dt in test_times), "Climatology missing so    me target forecast periods, can't make comparison"
    clim.columns = pd.to_datetime(clim.columns)
    clim = clim.loc[:, clim.columns.isin(test_times)]
    fm_list = []
    for st in clim.index:
        # Small change of station in climatology not in observed data for 2024, skip over
        if st in raws:
            dat = raws[st]['RAWS']
            dat.date_time = pd.to_datetime(dat.date_time)
            dat = dat[dat.date_time.isin(clim.columns)]
            dat = dat[["stid", "date_time", "fm"]]
            fm_list.append(dat)
    fm = pd.concat(fm_list)
    fm.date_time = fm.date_time.astype(str)
    clim = clim.reset_index().melt(id_vars='stid', var_name='date_time', value_name='preds')
    clim.date_time = clim.date_time.astype(str)
    clim = clim.merge(fm, on=["stid", "date_time"], how="left")
    clim = clim[(~clim.preds.isna()) & (~clim.fm.isna())]

    # Read output files for forecast analysis run
    ## Get all files in outputs
    files = os.listdir(osp.join(f_dir, 'forecast_outputs'))
    files = sorted(files, key=lambda x: int(re.search(r'_(\d+)\.h5$', x).group(1))) # Sort by task number, shouldn't be necessary but for clarity
    ## Read and combine into dataframe, add indicator column for replication number from file name
    rnn = read_hdf_list(files, key="rnn")
    ode = read_hdf_list(files, key="ode")
    xgb = read_hdf_list(files, key="xgb")

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

    df = pd.concat([rnn, xgb, ode], ignore_index=True)
    ml_dict = read_pkl(osp.join(f_dir, "ml_data.pkl"))
    loc_df = pd.DataFrame.from_dict(
        {k: v["loc"] for k, v in ml_dict.items()},
        orient="index"
    )     
    df = df.merge(loc_df, on="stid", how="left") 
    df["hod"] = pd.Series(str2time(df["date_time"].tolist())).dt.hour
    print(f"Writing all forecast and errors to {osp.join(out_dir, 'all_errors.h5')}")
    df.to_hdf(osp.join(out_dir, "all_errors.h5"), key="all_errors")

    ## Overall Error, averaged over every hour and location
    summary = df.groupby(["Model"], sort=False).agg(**agg_named).reset_index()
    print(f"Writing overall error summary to {osp.join(out_dir, 'overall.csv')}")
    summary.to_csv(osp.join(out_dir, "overall.csv"))

    ## Error by Station, averaged over all times
    summary = df.groupby(["Model", "stid"], sort=False).agg(**agg_named).reset_index()
    summary = summary.merge(loc_df, on="stid", how="left")
    print(f"Writing by station error summary to {osp.join(out_dir, 'by_stid.csv')}")
    summary.to_csv(osp.join(out_dir, "by_stid.csv"))

    ## Error by hour of the day, averaged over all stations and days
    summary = df.groupby(["Model", "hod"], sort=False).agg(**agg_named).reset_index()
    print(f"Writing by hour of day error summary to {osp.join(out_dir, 'by_hod.csv')}")
    summary.to_csv(osp.join(out_dir, "by_hod.csv"))

    ## Error by hour & day, averaged over all stations
    summary = df.groupby(["Model", "date_time"], sort=False).agg(**agg_named).reset_index()
    print(f"Writing per date time error summary to {osp.join(out_dir, 'by_dt.csv')}")
    summary.to_csv(osp.join(out_dir, "by_dt.csv"))

    ## Error by replication number, averaged over all stations and times
    ## NOTE: climatology is excluded, generates a prediction for all stations in forecast period. No uncertainty that can be estimated from replication
    summary = df.loc[df["Model"] != "CLIMATOLOGY"].groupby(["Model", "rep"], sort=False).agg(**agg_named).reset_index()
    print(f"Writing by replication error summary to {osp.join(out_dir, 'by_rep.csv')}")
    summary.to_csv(osp.join(out_dir, "by_rep.csv"))
    
