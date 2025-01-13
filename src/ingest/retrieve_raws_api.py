# Set of functions and executable to retrieve and manipulate RAWS data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## RAWS Data is retrieved using SynopticPy python package, which engages the Synoptic Data API
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
from utils import read_yml, Dict, time_intp, str2time, rename_dict


# Read RAWS Metadata
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
raws_meta = read_yml(osp.join(CONFIG_DIR, "variable_metadata", "raws_metadata.yaml"))


# Module Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_stations(bounding_box):
    """
    Return a polars dataframe of RAWS sensor data and associated units. Shift the start time by 1 hour since most stations return data some minutes after requested time, we do this for time interpolation to have endpoints
    
    Parameters:
    -----------
    bounding_box : list of numeric
        Format [min_lon, min_lat, max_lon, max_lat], NOTE different format than wrfxpy rtma_cycler

        
    Returns:
    --------
    st : list
        List of RAWS STIDs

    """ 
    sts = synoptic.Metadata(
        bbox=bounding_box,
        vars=["fuel_moisture"], # We only want to include stations with FMC. Other "raws_vars" are bonus later
    ).df()

    return sts

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


def build_raws_dict(config, rename=True, verbose = True):
    """
    Wrapper function that applies the module functions. Given config dictionary, it returns a formatted dictionary of RAWS data
    """

    # Extract config info
    bbox = config.bbox
    start = config.start_time
    end = config.end_time
    print(f"Start Date of RAWS retrieval: {start}")
    print(f"End Date of retrieval: {end}")
    print(f"Spatial Domain: {bbox}")    

    # Get station metadata within bbox and time period
    bbox_reordered = [bbox[1], bbox[0], bbox[3], bbox[2]] # Synoptic uses different bbox order
    start_dt = str2time(start)
    end_dt = str2time(end)
    sts = get_stations(bbox_reordered)

    # Collect RAWS data
    ## FMC is required, but collect all other available weather data
    ## Shifting the start time by 1 hour since most stations return data some minutes after requested time, we do this for time interpolation to have endpoints
    raws_weather_vars = config.get("raws_weather_vars", raws_meta["raws_weather_vars"])
    raws_dict = {}
    
    for st in sts["stid"]:
        print("~"*50)
        print(f"Attempting retrival of station {st}")
        try:
            df = synoptic.TimeSeries(
                stid=st,
                start=start_dt-relativedelta(hours=1),
                end=end_dt,
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
    times = pl.datetime_range(
        start=start_dt,
        end=end_dt,
        interval="1h",
        time_zone = "UTC",
        eager=True
    ).alias("time")
    times = np.array(times.to_list())

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
            
    return raws_dict







# Executed Code 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':

    # Handle arguments
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ## expects a config file with times and bbox, option argument saves output
    ## Config bbox format should match format in wrfxpy rtma_cycler: [latmin, lonmin, latmax, lonmax]
    if len(sys.argv) not in {2, 3}: 
        print(f"Invalid arguments. {len(sys.argv)} was given but 2 or 3 expected")
        print(('Usage: %s <config_file> <optional_output_file>' % sys.argv[0]))
        print("Example: python src/ingest/retrieve_raws_api.py etc/training_data_config.json data/raws.pkl")
        sys.exit(-1)

    
    # Handle config
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    conf_file = sys.argv[1]
    with open(conf_file, "r") as json_file:
        config = json.load(json_file)   
        config = Dict(config)
    print(json.dumps(config, indent=4))

    bbox = config.bbox
    start = config.start_time
    end = config.end_time
    print(f"Start Date of RAWS retrieval: {start}")
    print(f"End Date of retrieval: {end}")
    print(f"Spatial Domain: {bbox}")


    # Build Data Dictionary
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    raws_dict = build_raws_dict(config)
      

    # Write output if path provided
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    

    if len(sys.argv) == 3:
        output_path = sys.argv[2]
        print(f"Writing output to {output_path}")
        with open(output_path, 'wb') as file:
            pickle.dump(raws_dict, file)

    