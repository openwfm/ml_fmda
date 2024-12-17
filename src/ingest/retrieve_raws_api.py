# Set of functions to manipulate RAWS data
# RAWS Data is retrieved using SynopticPy python package, which engages the Synoptic Data API
# Credit to Brian Blaylock for SynopticPy package

import synoptic
import numpy as np
import polars as pl
import pandas as pd
import sys
sys.path.append("..")
from utils import filter_nan_values, time_intp

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





    