# Set of functions and executable to retrieve and manipulate RAWS data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## RAWS Data is retrieved using SynopticPy python package, which engages the Synoptic Data API
## Credit to Brian Blaylock for SynopticPy package
## NOTE: Polars dataframes are used in SynopticPy instead of pandas. We use polars for the code here but convert to pandas to allow for pickle save which is listed as not implemented in polars as of Dec 17 2024

import synoptic
import numpy as np
import polars as pl
import pandas as pd
import pickle
import json
import os.path as osp
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
###################
# FOR TESTING, In production, this is set in the shell file
import sys
sys.path.append("src")
###################
from utils import Dict, filter_nan_values, time_intp, str2time

# Hard-coded dictionary used to distinguish unchanging physical attributes of RAWS, or static variables, from time dynamic attributes that may be subject to temporal interpolation, or the weather vars
raws_vars_dict = {
    'raws_weather_vars': ["air_temp", "relative_humidity", "precip_accum", "fuel_moisture", "wind_speed", "solar_radiation", "pressure", "soil_moisture", "soil_temp", "snow_depth", "snow_accum", "wind_direction"],
    'raws_static_vars': ["stid", "latitude", "longitude", "elevation", "name", "state", "id"]
}

def format_raws(df, 
                static_vars=raws_vars_dict["raws_static_vars"], 
                weather_vars=raws_vars_dict["raws_weather_vars"]
               ):
    """
    Return a dataframe of RAWS sensor data and associated units.
    
    Parameters:
    -----------
    df : DataFrame 

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


def get_static(df, static_vars=raws_vars_dict["raws_static_vars"]):
    """
    Given dataframe of timeseries observations from RAWS station, get dictionary of static info, such as identifiers and physical attributes of station.
    
    Parameters:
    -----------
        df: Input dataframe with timeseries observations.
        static_vars: List of column names to extract static information from.
    
    Returns:
    -----------
        A dictionary called "loc" containing the unique value for each column in static_vars.
    
    """
    
    loc = {}
    for col in static_vars:
        if col in df.columns:
            unique_values = df[col].unique()
            if len(unique_values) == 1:
                loc[col] = unique_values[0]
            else:
                raise ValueError(f"Column '{col}' has more than one unique value: {unique_values}")
        else:
            raise KeyError(f"Column '{col}' not found in the dataframe.")
    return loc


def time_intp_df(df, target_times, 
                 static_cols=raws_vars_dict["raws_static_vars"], 
                 time_cols=raws_vars_dict["raws_weather_vars"]):
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


## FOR LATER TESTING

# Dataframe used to standardize naming from different data sources. 'fmda_name' are the variable names used within this project
name_df_raws = pl.DataFrame({
    "raws_name": [
        "air_temp", 
        "fuel_moisture", 
        "relative_humidity", 
        "solar_radiation", 
        "wind_speed", 
        "precip_accum", 
        "soil_moisture",
        "soil_temp"
    ],
    "fmda_name": [
        "temp", 
        "fm", 
        "rh", 
        "solar", 
        "wind", 
        "precip_accum", 
        "soilm",
        "soilt"
    ]
})



def rename_raws_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Renames columns in a Polars DataFrame based on a globally available mapping DataFrame. Hard coded so raws names turned into names defined in fmda project

    Parameters:
        df (pl.DataFrame): Input Polars DataFrame.

    Returns:
        pl.DataFrame: DataFrame with renamed columns.
    """
    # Extract raws_name and fmda_name as a list of tuples
    rename_dict = {
        row[0]: row[1] for row in name_df_raws.rows() if row[0] in df.columns
    }
    
    # Rename the columns
    return df.rename(rename_dict)



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

    
    # Get station metadata within bbox and time period
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    bbox_reordered = [bbox[1], bbox[0], bbox[3], bbox[2]] # Synoptic uses different bbox order
    start_dt = str2time(start)
    end_dt = str2time(end)
    sts = synoptic.Metadata(
        bbox=bbox_reordered,
        vars=["fuel_moisture"], # Note we only want to include stations with FMC. Other "raws_vars" are bonus later
        obrange=(start_dt-relativedelta(hours=1), end_dt),
    ).df()

    
    # Collect RAWS data
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ## FMC is required, but collect all other available weather data
    ## Shifting the start time by 1 hour since most stations return data some minutes after requested time, we do this for time interpolation to have endpoints
    raws_dict = {}
    
    for st in sts['stid']:
        print("~"*50)
        print(f"Attempting retrival of station {st}")
        try:
            df = synoptic.TimeSeries(
                stid=st,
                start=start_dt-relativedelta(hours=1),
                end=end_dt,
                vars=raws_vars_dict["raws_weather_vars"],
                units = "metric"
            ).df()
        
            dat, units = format_raws(df)
            loc = get_static(dat)
            raws_dict[st] = {
                'RAWS': dat,
                'units': units,
                'loc': loc,
                'misc': "Data retrieved using `synoptic.TimeSeries` and formatted with custom functions within `ml_fmda` project."
            }
        except Exception as e:
            print(f"An error occured: {e}")


    # Fix Time and Interpolate
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


    # Write output if path provided
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    

    if len(sys.argv) == 3:
        output_path = sys.argv[2]
        print(f"Writing output to {output_path}")
        with open(output_path, 'wb') as file:
            pickle.dump(raws_dict, file)

    