# Functions to run climatology method

import numpy as np
import copy
import pandas as pd
import random
import os
import os.path as osp
import sys
import warnings
from dateutil.relativedelta import relativedelta
# from joblib import Parallel, delayed
import multiprocessing as mp
from functools import partial

# Set up project paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CURRENT_DIR = osp.dirname(osp.normpath(osp.abspath(__file__)))
PROJECT_ROOT = osp.dirname(osp.dirname(osp.normpath(CURRENT_DIR)))
sys.path.append(osp.join(PROJECT_ROOT, "src"))
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import Dict, time_range, read_yml, print_dict_summary, is_consecutive_hours, read_pkl


# Read Metadata
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
params_models = read_yml(osp.join(CONFIG_DIR, "params_models.yaml"))
raws_meta = read_yml(osp.join(CONFIG_DIR, "variable_metadata", "raws_metadata.yaml"))

# Update stash path. We do this here so it works if module called from different locations
raws_meta.update({'raws_stash_path': osp.join(PROJECT_ROOT, raws_meta['raws_stash_path'])})



# Climatology Method
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def time_to_climtimes(t, nyears = 10, ndays=15):
    """
    Given a time, get the corresponding times that will be used for the climatology method.

    Arguments
        t : datetime
            Reference time for method
        nyears: int
            Number of years to look back for data
        ndays: int
            Number of days to bracket the target time, so t +/- ndays is the goal
    """      

    t_years = time_range(
        start = t - relativedelta(years = nyears),
        end = t,
        freq = pd.DateOffset(years=1)
    )

    # For each year, get range of days plus/minus ndays and append to running times object
    ts = []
    for ti in t_years:
        ti_minus_days = ti - relativedelta(days = ndays)
        ti_plus_days = ti + relativedelta(days = ndays)
        ti_grid = time_range(ti_minus_days, ti_plus_days, freq="1d")
        ts.extend(ti_grid)

    # Trim times based on before input time
    ts = np.array(ts)
    ts = ts[ts < t]
    
    return ts

## Helper functions for climatology

def _load_and_filter_pickle(file_path, sts):
    """Load a pickle file using pd.read_pickle and filter by 'stid' column."""
    try:
        df = pd.read_pickle(file_path)
        df.columns = df.columns.str.lower()
        if isinstance(df, pd.DataFrame) and "stid" in df.columns:
            return df[df["stid"].isin(sts)]  # Filter rows
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

# def _parallel_load_pickles(file_list, sts, num_workers=8):
#     """Parallel loading and filtering, using helper function above."""
#     results = Parallel(n_jobs=num_workers, backend="loky")(delayed(_load_and_filter_pickle)(f, sts) for f in file_list)
#     return pd.concat([df for df in results if df is not None], ignore_index=True)


def _parallel_load_pickles(file_list, sts, num_workers=8):
    """Parallel loading and filtering using multiprocessing with a progress bar."""
    from tqdm import tqdm
    
    with mp.Pool(processes=num_workers) as pool:
        # Use functools.partial to fix the second argument 'sts'
        func = partial(_load_and_filter_pickle, sts=sts)

        results = []
        with tqdm(total=len(file_list), desc="Processing Files") as pbar:
            for result in pool.imap_unordered(func, file_list):
                results.append(result)
                pbar.update(1)
    
    return pd.concat([df for df in results if df is not None], ignore_index=True)

def _filter_clim_data(clim_data, clim_times):
    """
    Filters clim_data to include only rows where the 'datetime' column matches 
    any datetime in clim_times based on year, month, day, and hour.
    
    Parameters:
    - clim_data (pd.DataFrame): DataFrame containing 'datetime' column (numpy datetime64).
    - clim_times (np.ndarray): Array of datetime objects to match.

    Returns:
    - pd.DataFrame: Filtered DataFrame.
    """
    # Convert clim_times to a DataFrame for efficient merging
    clim_times_df = pd.DataFrame({
        "year": [t.year for t in clim_times],
        "month": [t.month for t in clim_times],
        "day": [t.day for t in clim_times],
        "hour": [t.hour for t in clim_times]
    }).drop_duplicates()  # Remove duplicates to speed up filtering

    # Extract the relevant time components from clim_data
    clim_data_filtered = clim_data.assign(
        year=clim_data["datetime"].dt.year,
        month=clim_data["datetime"].dt.month,
        day=clim_data["datetime"].dt.day,
        hour=clim_data["datetime"].dt.hour
    ).merge(clim_times_df, on=["year", "month", "day", "hour"], how="inner")

    return clim_data_filtered.drop(columns=["year", "month", "day", "hour"])

def _mean_fmc_by_stid(filtered_df, min_years):
    """
    Computes the average fm10 grouped by 'stid', but returns NaN if the number 
    of unique years in 'datetime' is less than nyears.

    Parameters:
    - filtered_df (pd.DataFrame): DataFrame containing 'stid', 'datetime', and 'fm10'.
    - nyears (int): Minimum number of unique years required per 'stid'.

    Returns:
    - pd.Series: Averaged fm10 per 'stid' (NaN if unique years < nyears).
    """
    # Extract unique years for each STID
    year_counts = filtered_df.groupby("stid")["datetime"].apply(lambda x: x.dt.year.nunique())

    # Compute fm10 average per STID
    fm10_avg = filtered_df.groupby("stid")["fm10"].mean()

    # Set to NaN where unique years < nyears
    fm10_avg[year_counts < min_years] = np.nan

    return fm10_avg

    
def build_climatology(start, end, bbox, clim_params=None, n_workers = 8):
    """
    Given time period and spatial domain, get all RAWS fm10 data from
    stash based on params. start and end define the forecast hours. 
    Params includes 
        - nyears: number of years back from forecast time to look for data
        - min_years: required number of unique years with available data for a given time and RAWS
        - ndays: number of days to bracket target forecast hour, so target time plus/minus ndays are collected
    """

    from ingest.RAWS import get_stations, get_file_paths
    
    if clim_params is None:
        clim_params = Dict(params_models["climatology"])
    nyears = clim_params.nyears
    ndays = clim_params.ndays
    min_years = clim_params.min_years
    

    # Retrieve data
    ## Note, many station IDs will be empty, the list of stids was for the entire bbox region in history
    print(f"Retrieving climatology data from {start} to {end}")
    print("Params for Climatology:")
    print(f"    Number of years to look back: {nyears}")
    print(f"    Number of days to bracked target hour: {ndays}")
    print(f"    Required number of years of data: {min_years}")
    
    # Get target RAWS stations
    sts_df = get_stations(bbox)
    sts = list(sts_df["stid"])

    # Forecast Times, and needed RAWS file hours based on params
    ftimes = time_range(start, end)
    t0 = ftimes.min() - relativedelta(years=clim_params.nyears) - relativedelta(days = clim_params.ndays)
    t1 = ftimes.max()
    all_times = time_range(t0, t1)
    print(f"Total hours to retrieve for climatology: {len(all_times)}")    
    
    raws_files = get_file_paths(all_times)
    raws_files = [f for f in raws_files if os.path.exists(f)]
    print(f"Existing RAWS Files: {len(raws_files)}")    
    
    # Load data and get forecasts
    print(f"Reading RAWS Files with {n_workers} workers")
    clim_data = _parallel_load_pickles(raws_files, sts, num_workers = n_workers)

    return clim_data

def calculate_fm_forecasts(ftimes, clim_data, clim_params=None):
    """
    Runs `time_to_climtimes` on each time in `ftimes`, filters `clim_data`,
    computes the average `fm10` per `stid`, and combines results.

    Parameters:
    - ftimes (np.ndarray): Array of datetime objects to process.
    - clim_data (pd.DataFrame): DataFrame containing 'stid', 'datetime', and 'fm10'.
    - clim_params: Object containing `nyears` and `ndays` parameters.

    Returns:
    - pd.DataFrame: Combined results with average fm10 per stid for each ftime.
    """
    if clim_params is None:
        clim_params = Dict(params_models["climatology"])
    
    results = []

    for ftime in ftimes:
        # Generate climtimes for the given ftime
        clim_times = time_to_climtimes(ftime, nyears=clim_params.nyears, ndays=clim_params.ndays)
        
        # Filter clim_data based on clim_times
        filtered_data = _filter_clim_data(clim_data, clim_times)

        # Compute the average fm10 per stid
        fm_forecast = _mean_fmc_by_stid(filtered_data, min_years=clim_params.min_years)

        # Store results with corresponding ftime
        df_result = fm_forecast.reset_index()
        df_result["forecast_time"] = ftime  # Add ftime column
        results.append(df_result)

    # Combine all results into a single DataFrame
    df = pd.concat(results, ignore_index=True)
    df = df.pivot(index="stid", columns="forecast_time", values="fm10")    

    # Filter out all NA
    dropped_stids = df.index[df.isna().all(axis=1)].tolist()
    df = df.dropna(how="all")
    print(f"No Data Found for STIDS: {dropped_stids}")
    print(f"Returning forecasts for: {df.shape[0]} unique STIDs")
    
    return df
    

if __name__ == '__main__':

    print("Imports successful, no executable code")
