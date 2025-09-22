# Set of functions and executable to retrieve and manipulate RAWS data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Realtime RAWS Data is retrieved using SynopticPy python package, which engages the Synoptic Data API
## Stashed RAWS data retrieved from MesoDB, maintained by Angel Farguell, ask him for access
## Credit to Brian Blaylock for SynopticPy package
## NOTE: Polars dataframes are used in SynopticPy instead of pandas. We use polars for the code here but convert to pandas to allow for pickle save which is listed as not implemented in polars as of Dec 17 2024
## Metadata files for RAWS in etc/variable_metadata/raws_metadata.yaml, and filters for identifying bad RAWS data at etc/variable_metadata/data_params.yaml 

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
import ast

# Set up project paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CURRENT_DIR = osp.dirname(osp.normpath(osp.abspath(__file__)))
PROJECT_ROOT = osp.dirname(osp.dirname(osp.normpath(CURRENT_DIR)))
sys.path.append(osp.join(PROJECT_ROOT, "src"))
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import read_yml, read_pkl, Dict, time_intp, str2time, rename_dict, time_range


# Read RAWS Metadata and Data Params for high/low bounds
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
raws_meta = read_yml(osp.join(CONFIG_DIR, "variable_metadata", "raws_metadata.yaml"))
project_paths = read_yml(osp.join(CONFIG_DIR, "paths.yaml"))
raws_stash_path = project_paths['raws_stash_path']

params_data = Dict(read_yml(osp.join(CONFIG_DIR, "variable_metadata", "data_filters.yaml")))



# API Module Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_stations(bbox, source="api"):
    """
    Get list of RAWS station ID strings given input spatial domain bbox. Return a polars dataframe of RAWS sensor data and associated units. Shift the start time by 1 hour since most stations return data some minutes after requested time, we do this for time interpolation to have endpoints
    
    Parameters:
    -----------
    bounding_box : list of numeric
        Format [min_lat, min_lon, max_lat, max_lon] to match wrfxpy
        NOTE different format used by Synoptic, this function will convert internally
    source: str, one of "api" or "stash"
        
    Returns:
    --------
    st : DF
        dataframe of RAWS STIDs

    """ 
    if source == "api":
        sts = _get_stations_api(bbox)
    elif source == "stash":
        sts = _get_stations_stash(bbox)
    else:
        raise ValueError(f"Input source not one of api/stash, {source=}")

    return sts

def _get_stations_api(bbox):
    """
    Get list of RAWS inside bbox from synoptic API
    """
    bbox_reordered = [bbox[1], bbox[0], bbox[3], bbox[2]]
    sts = synoptic.Metadata(
        bbox=bbox_reordered,
        vars=["fuel_moisture"], # We only want to include stations with FMC. Other "raws_vars" are bonus later
    ).df()
    return sts

def _get_stations_stash(bbox):
    """
    Get list of RAWS inside bbox from RAWS stash, path found in global paths etc/paths.yaml
    """
    print(f"Getting list of stations inside {bbox} from: {raws_stash_path}/stations.pkl")
    sts = pd.read_pickle(osp.join(raws_stash_path, "stations.pkl"))
    sts["LATITUDE"] = pd.to_numeric(sts["LATITUDE"], errors="coerce")
    sts["LONGITUDE"] = pd.to_numeric(sts["LONGITUDE"], errors="coerce")
    sts = sts[(sts.LATITUDE>=bbox[0]) & (sts.LONGITUDE>=bbox[1]) & (sts.LATITUDE<=bbox[2]) & (sts.LONGITUDE <= bbox[3])]
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
    # NOTE: checking if elevation is null and setting to NA if not, can't confirm as of Aug 15 2025 whether to fully filter these stations out or now
    if loc["elev"]:
        loc["elev"] = loc["elev"] * 0.3048 # Convert ft to M
    else:
        loc["elev"] = np.nan

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

    assert osp.exists(raws_stash_path), f"Stash path given in RAWS metadata file does not exist"

    # Create list of file paths based on needed hours
    paths = [
        osp.join(
            raws_stash_path,        
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
    
    # Loop through file paths and extract info from needed STID
    for path in paths:
        try:
            #dat = read_pkl(path)           
            print(f"loading file {path}")
            dat = pd.read_pickle(path)
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

    # Interpolate
    # No start time offset here
    # Hard coded static and time columns
    times = time_range(start, end)
    no_data = []
    for st in raws_dict:
        if raws_dict[st]["RAWS"].shape[0] == 0:
            no_data.append(st)
        else:
            vals_to_na(raws_dict[st]["RAWS"], "fm", verbose=False) # Filter extreme values based on data params
            if np.mean(np.isnan(raws_dict[st]["RAWS"]["fm"])) == 1:
                no_data.append(st)
            else:
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

    # Remove Missing Data
    print(f"No data found for stations {len(no_data)} stations within input bbox for given time period. Removing")
    for st in no_data:
        raws_dict.pop(st)

    print(f"Retrieved data for {len(raws_dict.keys())} stations")

    # Save if path provided
    if save_path is not None:
        with open(save_path, 'wb') as handle:
            pickle.dump(raws_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return raws_dict


def parse_bbox(box_str):
    try:
        # Use ast.literal_eval to safely parse the string representation
        # This will only evaluate literals and avoids security risks associated with eval
        box = ast.literal_eval(box_str)
        # Check if the parsed box is a list and has four elements
        if isinstance(box, list) and len(box) == 4:
            return box
        else:
            raise ValueError("Invalid bounding box format")
    except (SyntaxError, ValueError) as e:
        print("Error parsing bounding box:", e)
        sys.exit(-1)
        return None

#if __name__ == '__main__':
#    if len(sys.argv) != 5:
#        print(f"Invalid arguments. {len(sys.argv)} was given but 4 expected")
#        print(('Usage: %s <esmf_from_utc> <esmf_to_utc> <bbox> <output_dir>' % sys.argv[0]))
#        print("Example: python src/ingest/RAWS.py '2024-01-01T00:00:00Z' '2024-03-01T00:00:00Z' '[40,-111,45,-110]' data/raws_test.pkl")
#        print("bbox format should match rtma_cycler: [latmin, lonmin, latmax, lonmax]")
#        print("Times should match format: 2023-06-01T00:00:00Z")
#        sys.exit(-1)
#
#    start = sys.argv[1]
#    end = sys.argv[2]
#    bbox = parse_bbox(sys.argv[3])
#    output_file = sys.argv[4]
#
#    print(f"Retrieving data from RAWS stash")
#    raws_dict = build_raws_dict_stash(start, end, bbox, save_path = output_file)



if __name__ == '__main__':

    print("Imports successful, no executable code")
