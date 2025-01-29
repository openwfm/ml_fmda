# Set of functions and executable to retrieve and manipulate RAWS data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Realtime RAWS Data is retrieved using SynopticPy python package, which engages the Synoptic Data API
## Stashed RAWS data retrieved from MesoDB, maintained by Angel Farguell, ask him for access
## Credit to Brian Blaylock for SynopticPy package
## NOTE: Polars dataframes are used in SynopticPy instead of pandas. We use polars for the code here but convert to pandas to allow for pickle save which is listed as not implemented in polars as of Dec 17 2024

import sys
import synoptic
import numpy as np
import polars as pl
import pandas as pd
import pickle
import json
import os.path as osp
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
from pathlib import Path

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

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import read_yml, read_pkl, Dict, time_intp, str2time, rename_dict, time_range


# Read RAWS Metadata and Data Params for high/low bounds
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
raws_meta = read_yml(osp.join(CONFIG_DIR, "variable_metadata", "raws_metadata.yaml"))
# Update stash path. We do this here so it works if module called from different locations
raws_meta.update({'raws_stash_path': osp.join(PROJECT_ROOT, raws_meta['raws_stash_path'])})

params_data = Dict(read_yml(osp.join(CONFIG_DIR, "params_data.yaml")))



# API Module Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_stations(bbox):
    """
    Get list of RAWS station ID strings given input spatial domain bbox. Return a polars dataframe of RAWS sensor data and associated units. Shift the start time by 1 hour since most stations return data some minutes after requested time, we do this for time interpolation to have endpoints
    
    Parameters:
    -----------
    bounding_box : list of numeric
        Format [min_lat, min_lon, max_lat, max_lon] to match wrfxpy
        NOTE different format used by Synoptic, this function will convert internally

        
    Returns:
    --------
    st : list
        List of RAWS STIDs

    """ 
    bbox_reordered = [bbox[1], bbox[0], bbox[3], bbox[2]]
    sts = synoptic.Metadata(
        bbox=bbox_reordered,
        vars=["fuel_moisture"], # We only want to include stations with FMC. Other "raws_vars" are bonus later
    ).df()

    return sts



    
def vals_to_na(df, col, verbose=True):
    """
    Set values of a column in a dataframe to NA if outside physically reasonable range of values. 
    """
    low = params_data['min_'+col]
    high = params_data['max_'+col]
    if verbose:
        print(f"Setting {col} observations to NA if outside range: {low} - {high}")

    # Modify In place
    df[col] = df[col].where((df[col] >= low) & (df[col] <= high), np.nan)



def format_raws(df, 
                static_vars=raws_meta["raws_static_vars"], 
                weather_vars=raws_meta["raws_weather_vars"]
               ):
    """
    Return a polars dataframe of RAWS sensor data and associated units.
    
    Parameters:
    -----------
    df : polars DataFrame 

    static_vars : list
        List of Synoptic variables that are static in time, i.e. physical features of the RAWS stations
        See https://demos.synopticdata.com/variables/index.html for more info
    weather_vars : list
        List of Synoptic variables that are dynamic in time, a.k.a sensor variables
        See https://demos.synopticdata.com/variables/index.html for more info
        
    Returns:
    --------
    dat, units : tuple of dataframe, dict
        Formatted and pivotted dataframe of RAWS data, and dictionary of associated units.

    """ 

    assert "fuel_moisture" in df["variable"], "fuel_moisture not detected in input dictionary"
    units = {} # stores units for variables
    
    
    for var in weather_vars:
        if var in df['variable']:
            df_temp = df.filter(df['variable'] == var)
            unit = df_temp['units'].unique()
            if len(unit) != 1:
                raise ValueError(f"Variable {var} has multiple values for units")
            units[var] = unit[0]
    
    dat = df.filter(pl.col("variable").is_in(weather_vars))
    dat = dat.pivot(
        values="value",
        index=["date_time"]+static_vars,
        on="variable"
    )

    print(f"Found {dat.shape[0]} FMC records")
    
    # Fix column units
    if "air_temp" in dat.columns and units['air_temp'] == "Celsius":
        print("Converting RAWS air temp from C to K")
        units['air_temp'] = "Kelvin"
        dat = dat.with_columns(
                (pl.col("air_temp")+273.15).alias("air_temp")
            )
        
    if 'elevation' in static_vars: # convert ft to meters
        print("Converting RAWS elevation from ft to meters")
        # loc['elevation'] = loc['elevation'] * 0.3048
        dat = dat.with_columns(
                (pl.col("elevation") * 0.3048).alias("elevation")
            )
        units['elevation'] = "m"    
        
    return dat, units


def get_static(df, st, static_vars=raws_meta["raws_static_vars"], name_mapping = raws_meta["rename_synoptic"]):
    loc = {col: values[0] for col, values in df.filter(df["stid"] == st).select(static_vars).to_dict(as_series=False).items()}
    loc = rename_dict(loc, name_mapping)
    loc["elev"] = loc["elev"] * 0.3048 # Convert ft to M
    
    return loc


def time_intp_df(df, target_times, 
                 static_cols=raws_meta["raws_static_vars"], 
                 time_cols=raws_meta["raws_weather_vars"]):
    """
    Interp and ...
    """

    # Get raw datetime values as numpy array
    time_raws = np.array(df["date_time"].to_list())    

    # Interpolate time dynamic columns only for columns that exist in the dataframe
    weather_data = {
        var: time_intp(
            time_raws, 
            df[var].to_numpy(), 
            target_times
        ) for var in time_cols if var in df.columns
    }
    # Create a Polars DataFrame from the interpolated results
    weather_df = pl.DataFrame(weather_data)
    weather_df = weather_df.with_columns(pl.Series("date_time", target_times))

    # Expand only for columns that exist in the dataframe
    nrow = weather_df.shape[0]
    static_data = {
        var: np.repeat(df[var].to_numpy()[0], nrow)
        for var in static_cols if var in df.columns
    }
    static_df = pl.DataFrame(static_data)  
    
    # Combine interpolated weather data and expanded static variables
    result_df = pl.concat([weather_df, static_df], how="horizontal")
    result_df = result_df.select(df.columns) # reorder columns to match original
    
    return result_df


def build_raws_dict_api(start, end, bbox, rename=True, verbose = True, save_path = None):
    """
    Wrapper function that applies the module functions. Given config dictionary, it returns a formatted dictionary of RAWS data
    """

    if verbose:
        print(f"Start Date of RAWS retrieval: {start}")
        print(f"End Date of retrieval: {end}")
        print(f"Spatial Domain: {bbox}")    

    # Get station metadata within bbox and time period
    if type(start) is str:
        start = str2time(start)
    if type(end) is str:
        end = str2time(end)
    sts = get_stations(bbox)

    # Collect RAWS data
    ## FMC is required, but collect all other available weather data
    ## Shifting the start time by 1 hour since most stations return data some minutes after requested time, we do this for time interpolation to have endpoints
    raws_weather_vars = raws_meta["raws_weather_vars"]
    raws_dict = {}
    
    for st in sts["stid"]:
        print("~"*50)
        print(f"Attempting retrival of station {st}")
        try:
            df = synoptic.TimeSeries(
                stid=st,
                start=start-relativedelta(hours=1),
                end=end+relativedelta(hours=1),
                vars=raws_weather_vars,
                units = "metric"
            ).df()
        
            dat, units = format_raws(df)
            loc = get_static(sts, st)
            raws_dict[st] = {
                'RAWS': dat,
                'units': units,
                'loc': loc,
                'misc': "Data retrieved using `synoptic.TimeSeries` and formatted with custom functions within `ml_fmda` project."
            }
        except Exception as e:
            print(f"An error occured: {e}")

    # Fix Time, Interpolate, and Rename 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    times = time_range(start, end, freq="1h")

    print(f"Interpolating missing data in time from {times.min()} to {times.max()}")
    if rename:
        print(f"Renaming RAWS columns based on raws_metadata file")
    for st in raws_dict:
        nsteps = raws_dict[st]["RAWS"].shape[0]
        raws_dict[st]["RAWS"] = time_intp_df(raws_dict[st]["RAWS"], times)
        raws_dict[st]["RAWS"] = pd.DataFrame(raws_dict[st]["RAWS"], columns = raws_dict[st]["RAWS"].columns) # convert to pandas for pickle save
        raws_dict[st]["times"] = times
        if raws_dict[st]["RAWS"].shape[0] != nsteps:
            print("~"*75)
            print(st)
            raws_dict[st]["misc"] += " Interpolated data with numpy linear interpolation."
            print(f"    Original Dataframe time steps: {nsteps}")
            print(f"    Interpolated DataFrame time steps: {raws_dict[st]['RAWS'].shape[0]}")
            print(f"        interpolated {raws_dict[st]['RAWS'].shape[0] - nsteps} time steps")

        if rename:
            raws_dict[st]["units"] = rename_dict(raws_dict[st]["units"], raws_meta["rename_synoptic"])
            raws_dict[st]["RAWS"] = raws_dict[st]["RAWS"].rename(columns = raws_meta["rename_synoptic"])
            raws_dict[st]["loc"] = rename_dict(raws_dict[st]["loc"], raws_meta["rename_synoptic"])

    # Save if path provided
    if save_path is not None:
        with open(save_path, 'wb') as handle:
            pickle.dump(raws_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return raws_dict


# Stash Module Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def get_file_paths(times):
    """
    Get file paths for RAWS stash from given start and end dates.

    Arguments
        times: 1d numpy array of datetime
    """  

    assert osp.exists(raws_meta["raws_stash_path"]), f"Stash path given in RAWS metadata file does not exist"

    # Create list of file paths based on needed hours
    paths = [
        osp.join(
            raws_meta["raws_stash_path"],        
            str(time.year),     
            time.strftime('%j'),    #  Julian day of year, 001-366
            f"{str(time.year)}{time.strftime('%j')}{time.strftime('%H')}.pkl" )# Join with hour of day, 00-23
        for time in times
    ]
    
    return paths


def build_raws_dict_stash(start, end, bbox, rename=True, verbose = True, save_path=None):
    """
    Wrapper function that applies the module functions. Given config dictionary, it returns a formatted dictionary of RAWS data
    """

    if verbose:
        print(f"Start Date of RAWS retrieval: {start}")
        print(f"End Date of retrieval: {end}")
        print(f"Spatial Domain: {bbox}")  


    # Get station metadata within bbox and time period

    if type(start) is str:
        start = str2time(start)
    if type(end) is str:
        end = str2time(end)
    sts = get_stations(bbox)

    # Based on time arguments, get list of needed file paths in stash
    # Offset by 1 hr for data retrieval, so interpolation has endpoints
    times = time_range(start-relativedelta(hours=1), end+relativedelta(hours=1))
    paths = get_file_paths(times)

    # Create return dictionary with static info and other metadata
    raws_dict = {
        st: {
            "RAWS": [],
            "units": {"fm": "%", "elev": "m"},
            "loc": get_static(sts, st),
            "misc": "FMC data collected from RAWS stash."
            } 
        for st in sts["stid"]}
    
    
    # Loop through file paths and extract info from need STID
    for path in paths:
        try:
            dat = read_pkl(path)           
            for st in sts["stid"]:
                # Filter the data for the current station and append
                filtered = dat[dat['STID'] == st]
                if not filtered.empty:
                    raws_dict[st]["RAWS"].append(filtered)
        except Exception as e:
            print(f"An error occured: {e}") 
            
    # Combine the lists of DataFrames for each station into a single DataFrame, rename, and interpolate
    for st in raws_dict:
        if raws_dict[st]["RAWS"]:  # Check if the list is not empty
            raws_dict[st]["RAWS"] = pd.concat(raws_dict[st]["RAWS"], ignore_index=True)
            # Add a few static vars
            raws_dict[st]["RAWS"]["lat"] = raws_dict[st]["loc"]["lat"]
            raws_dict[st]["RAWS"]["lon"] = raws_dict[st]["loc"]["lon"]
            raws_dict[st]["RAWS"]["elev"] = raws_dict[st]["loc"]["elev"]      
        else:
            raws_dict[st]["RAWS"] = pd.DataFrame()  # Set an empty DataFrame if no data was found
        if rename:
            raws_dict[st]["RAWS"].rename(columns=raws_meta["rename_stash"], inplace=True)

    # Remove Stations with missing data
    no_data = []
    for st in list(raws_dict.keys()):
        if raws_dict[st]["RAWS"].shape[0] == 0:
            no_data.append(st)
            raws_dict.pop(st)
    print(f"No data found for stations {no_data}, removing")
    print(f"Retrieved data for {len(raws_dict.keys())} stations")
    
    # Interpolate
    # No start time offset here
    # Hard coded static and time columns
    times = time_range(start, end)
    
    for st in raws_dict:
        vals_to_na(raws_dict[st]["RAWS"], "fm", verbose=False) # Filter extreme values based on data params
        nsteps = raws_dict[st]["RAWS"].shape[0]
        d = time_intp_df(raws_dict[st]["RAWS"], times, static_cols = ["stid", "lat", "lon", "elev"], time_cols = ["fm"])
        d = pd.DataFrame(d, columns = d.columns)
        raws_dict[st]["RAWS"] = d
        raws_dict[st]["times"] = times
        if raws_dict[st]["RAWS"].shape[0] != nsteps:
            raws_dict[st]["misc"] += " Interpolated data with numpy linear interpolation."
            if verbose:
                print("~"*75)
                print(st)
                print(f"    Original Dataframe time steps: {nsteps}")
                print(f"    Interpolated DataFrame time steps: {raws_dict[st]['RAWS'].shape[0]}")
                print(f"        interpolated {raws_dict[st]['RAWS'].shape[0] - nsteps} time steps")
    
    # Save if path provided
    if save_path is not None:
        with open(save_path, 'wb') as handle:
            pickle.dump(raws_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return raws_dict

    