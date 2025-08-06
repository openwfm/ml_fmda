# Set of functions and executable to retrieve and manipulate HRRR data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# HRRR Data is retrieved using Brian Blaylock for Herbie package

import pandas as pd
import herbie
from herbie import FastHerbie
from datetime import datetime
import numpy as np
import os
import os.path as osp
import sys
import xarray as xr
from dateutil.relativedelta import relativedelta

# Set up project paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CURRENT_DIR = osp.dirname(osp.normpath(osp.abspath(__file__)))
PROJECT_ROOT = osp.dirname(osp.dirname(osp.normpath(CURRENT_DIR)))
sys.path.append(osp.join(PROJECT_ROOT, "src"))
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import read_yml, Dict, time_intp, str2time, print_dict_summary, rename_dict, time_range


# Read HRRR Metadata
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
hrrr_meta = read_yml(osp.join(CONFIG_DIR, "variable_metadata", "hrrr_metadata.yaml"))
project_paths = read_yml(osp.join(CONFIG_DIR, "paths.yaml"))
hrrr_stash_path = project_paths['hrrr_stash_path']

def features_to_searchstr(flist):
    """
    Given features list, return dictionary of regex search strings to be used in Herbie package
    """
    
    # Initialize the output dictionary
    search_strings = {
        "surface": "",
        "2m": "",
        "10m": ""
    }

    for feat in flist:
        feature_info = Dict(hrrr_meta[feat])
        feature_type = feature_info.feature_type
        
        if feature_type == "hrrr_data":
            layer = feature_info.layer
            ss = feature_info.herbie_str
            search_strings[layer] += f"|{ss}"
            if not ss in search_strings[layer]:
                search_strings[layer] += f"|{ss}"            
        elif feature_type == "engineered_data":
            fnames = feature_info.required_fmda_names
            for fn in fnames:
                feature_info2 = Dict(hrrr_meta[fn])
                layer = feature_info2.layer
                ss = feature_info2.herbie_str
                if not ss in search_strings[layer]:
                    search_strings[layer] += f"|{ss}"
    # Remove the initial pipe if it exists
    for key in search_strings:
        search_strings[key] = search_strings[key][1:] if search_strings[key].startswith("|") else search_strings[key]    
    
    return search_strings

def merge_datasets(ds_dict):
    """Merge list of Datasets together. Modified from Brian Blaylocks Herbie docs

    Since cfgrib doesn't merge data in different "hypercubes", we will
    do the merge ourselves.

    Parameters
    ----------
    ds_dict : dict
        A dictionary of xarray.Datasets organized by layer.

    Returns
    ----------
    ds : xarray DataSet
        Merged dataset with height above ground dimension added
    """


    # Check that all datasets in the dictionary have the same sizes
    sizes_list = [ds.sizes for ds in ds_dict.values()]
    if not all(sizes == sizes_list[0] for sizes in sizes_list):
        raise ValueError("All datasets must have the same .sizes attribute. Found mismatched dimensions")
    
    # Drop surface and heightAboveGround coordinate if it exists
    for key, ds in ds_dict.items():
        if "heightAboveGround" in ds.coords:
            ds_dict[key] = ds.drop_vars("heightAboveGround")
        if "surface" in ds.coords:
            ds_dict[key] = ds.drop_vars("surface")            

    ds = xr.merge(ds_dict.values())
    
    return ds


def calc_eq(ds):
    """
    Calculate wetting and drying equilibrium moisture content from the relative humidity and air temp

    Parameters:
        - ds: xarray dataset
    
    Returns: None
        Operation is in-place
    """
    if ds.t2m.units == "C":
        print("Converting from C to K")
        ds.t2m = ds.t2m + 273.15
    
    temp = ds.t2m
    rh = ds.r2

    print("Calculating equilibrium moisture content from air temp and rh")
    Ed = 0.924 * rh**0.679 + 0.000499 * np.exp(0.1 * rh) + 0.18 * (21.1 + 273.15 - temp) * (1 - np.exp(-0.115 * rh))
    Ew = 0.618 * rh**0.753 + 0.000454 * np.exp(0.1 * rh) + 0.18 * (21.1 + 273.15 - temp) * (1 - np.exp(-0.115 * rh))

    ds["Ed"] = Ed
    ds["Ew"] = Ew

    # Doing in-place modifying, no return

def calc_times(ds):
    """
    Calculate hour of day (HOD) and day of year (DOY) for given xarray object based on the valid_time. The valid time accounts for the forecast hour

    Parameters:
        - ds: xarray dataset
    
    Returns: xarray dataset
        Dataset with extended coordinates
    """

    ds = ds.assign_coords({
        "hod": ds.valid_time.dt.hour,
        "doy": ds.valid_time.dt.dayofyear
    })

    return ds

def rename_ds(ds):
    """
    Renames variables in an xarray Dataset based on the hrrr_meta dictionary.
    
    Parameters:
        ds (xr.Dataset): The input xarray Dataset.
        hrrr_meta (dict): Metadata dictionary containing xarray variable names and their new keys.
    
    Returns:
        xr.Dataset: Dataset with variables renamed.
    """
    rename_dict = {
        v['xarray_name']: key
        for key, v in hrrr_meta.items()
        if 'xarray_name' in v and v['xarray_name'] in ds
    }
    return ds.rename(rename_dict)

def get_units_xr(ds):
    """
    Get units from xarray object. 
    Looks for GRIB_parameterUnits and returns in a dictionary format for each data variable
    Prints warning if no units found, which should be the case for engineered variables like Ew, Ed
    Intended to be run after renaming, so it will be compatible with units from RAWS process
    """
    
    units = {}
    
    # Iterate through all data variables in the dataset
    for var in ds.data_vars:
        # Check if 'GRIB_parameterUnits' exists in the variable's attributes
        if 'GRIB_parameterUnits' in ds[var].attrs:
            # Add it to the dictionary
            units[var] = ds[var].attrs['GRIB_parameterUnits']
        else:
            # Print a warning message
            print(f"Warning: 'GRIB_parameterUnits' not found for variable '{var}'.")
    
    return units

def retrieve_hrrr(start, end, all_features = True, forecast_step = 3, save_to_stash=True, read_to_memory=True):
    """
    Function called by user to retrieve hrrr data. Checks for existence of data in HRRR stash from project paths. If exists, reads it, if not calls the API retrieval function
    
    Args:
        - save_to_stash: whether to save formatted HRRR data to stash directory set by paths.yaml project. NOTE: only implemented to always save to stash as of May 26 2025
        - read_to_memory: if true, return xarray ds to env that called it. if false, just retrieve and save to stash. NOTE: only makes sense to use read_to_memory False if save_to_stash is True
    """
    # Extract Time range as datetime objects 
    if type(start) is str:
        start = str2time(start)
    if type(end) is str:
        end = str2time(end) 

    print(f"Retrieving HRRR data from {start} to {end}")

    # Check for day files in HRRR stash given start and end
    # Based on time range, get list of days that exist in stash and list that don't
    start_day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    end_day = end.replace(hour=0, minute=0, second=0, microsecond=0)
    days = time_range(start_day, end_day, freq="1d")
    stashed_days = []
    needed_days = []
    for d in days:
        path = osp.join(hrrr_stash_path, f"{d.year}", f"{d.strftime('%m')}")
        file_name=f"hrrr_prs03_{d.year}-{d.strftime('%m')}-{d.strftime('%d')}.nc" # formatted file name
        if osp.exists(osp.join(path, file_name)):
            stashed_days.append(d)
        else:
            needed_days.append(d)

    # Run API retieval on days not stashed
    print(f"    Days already stashed: {stashed_days}")
    print(f"    Days need to retrive: {needed_days}")
    for d in needed_days:
        end_d = d.replace(hour=23)
        retrieve_hrrr_api(d, end_d, all_features=True, forecast_step=3, save_to_stash=save_to_stash)
    
    if not read_to_memory:
        print(f"    {read_to_memory=}, exiting function")
        return
    # Read all HRRR days from stash, subset to exact times given
    if len(stashed_days+needed_days) < 1:
        print(f"No days of data remaining, check input times {start=}, {end=}")
        sys.exit(-1)
    datasets = []
    days = sorted(stashed_days+needed_days)
    for d in days:
        file_path = osp.join(hrrr_stash_path, f"{d.year}", f"{d.strftime('%m')}", f"hrrr_prs03_{d.year}-{d.strftime('%m')}-{d.strftime('%d')}.nc")
        ds = xr.open_dataset(file_path)
        datasets.append(ds)
    if len(datasets)==1:
        combined = datasets[0]
    else:
        combined = xr.concat(datasets, dim="time", combine_attrs="drop_conflicts")
    # Filter to exact needed times. Data collection gets whole days, this can return partial hours
    dt = pd.to_datetime(combined.date_time.to_numpy(), utc=True)
    mask = (dt >= start) & (dt <= end)
    combined = combined.isel(time=mask)

    return combined


def retrieve_hrrr_api(start, end, all_features = True, forecast_step = 3, save_to_stash=True):
    """
    Wrapper function to get HRRR data 

    Parameters
    -------------


    all_features : bool

    """

    # Extract Time range as datetime objects 
    if type(start) is str:
        start = str2time(start)
    if type(end) is str:
        end = str2time(end)
    print(f"Retrieving HRRR data from {start} to {end} from API")

    # Handle Features List
    if all_features:
        # All top level keys from hrrr_meta file into a list
        features_list = [*hrrr_meta.keys()]
    else:
        raise NotImplementedError()
   
    # Adjust times for forecast step, save input time for writing file to stash
    start0 = start
    end0 = end
    start = start - relativedelta(hours = forecast_step)
    end = end - relativedelta(hours = forecast_step)
    print(f"Shifting retrieval time to account for forecast step of {forecast_step}.")
    print(f"    Data retrieval start: {start}")
    print(f"    Data retrieval end: {end}")
    
    # Create a range of dates
    dates = pd.date_range(
        start = start.replace(tzinfo=None),
        end = end.replace(tzinfo=None),
        freq="1h"
    )

    # Open Data Connection w Herbie
    FH = FastHerbie(
        dates, 
        model="hrrr", 
        product="prs",
        fxx=range(forecast_step, forecast_step+1)
    )

    # Set up search strings
    print(f"Target Features List: {features_list}")
    search_strings = features_to_searchstr(features_list)
    print("HRRR Search Strings:")
    print_dict_summary(search_strings)    

    # Read data, grouped by layer
    ds_dict = {}
    for layer in search_strings:
        print(f"Reading HRRR data for layer: {layer}")
        print(f"    search strings: {search_strings[layer]}")
        ds_dict[layer] = FH.xarray(search_strings[layer], remove_grib=False) # Keep grib for easier re-use, delete later
    ds = merge_datasets(ds_dict)

    # Store Regular i,j grid coordinates
    ds = ds.assign_coords({
        'grid_x' : ds.x,
        'grid_y' : ds.y
    })

    # Construct Other Predictors
    if any(s in features_list for s in ["hod", "doy"]):
        calc_eq(ds)
    if any(s in features_list for s in ["hod", "doy"]):
        ds = calc_times(ds)
    # Add date_time col based on valid_time with UTC timezone
    times = ds["valid_time"].values
    ds["date_time"] = ("time", times)

    # Save to HRRR stash
    # Structure is subdirectories that specify year and month, file name specifies day
    # NOTE: usage assumes retrieval only run for 1 day at a time, code won't break otherwise just store data confusingly
    if save_to_stash:
        print(f"Saving data to stash path: {hrrr_stash_path}")
        write_hrrr_ds(ds, start0, end0)

    return ds

def write_hrrr_ds(ds, start, end):
    """
    Helper function to write HRRR as netcdf. Uses convention of subdirectories in the hrrr stash path for year and then month, file name for the day of the start of retrieval. NOTE: this is intended to be written for a start and end date that correspond to a full 24 hour day UTC. Naming convention won't make sense if times given that don't work this way
    """
    # Set up paths
    os.makedirs(hrrr_stash_path, exist_ok=True)
    out_dir=osp.join(hrrr_stash_path, f"{start.year}", f"{start.strftime('%m')}")
    os.makedirs(out_dir, exist_ok=True)
    file_name=f"hrrr_prs03_{start.year}-{start.strftime('%m')}-{start.strftime('%d')}" # formatted file name
    print(f"Saving file {file_name} to {out_dir}")
    # Write
    ds.to_netcdf(osp.join(out_dir, f"{file_name}.nc"))

def subset_hrrr_bbox(ds, bbox):
    """
    Subset HRRR spatial data with a spatial bounding box
    """
    # Not implemented yet, might require command line gdal instead of python
    pass

def subset_hrrr2raws(ds, raws):
    """
    Format training set of atmospheric HRRR data. subset HRRR spatial data by selecting data interpolated to a set of input points. In this project, it is the locations of the RAWS stations. Then rename and subset to a desired features list
    Parameters
    ----------
    ds : xarray Dataset
        A dictionary of xarray.Datasets organized by layer.
    raws : dict
        Formatted raws dict, typically the output of build_raws_dict

    Returns
    ----------
    ds_pts : xarray DataSet
        HRRR data with number of locations equal to input pts, all HRRR data interpolated to those locations
    """    

    # Build dataframe with longitude and latitude as cols from input data dict
    longitude = []
    latitude = []
    stid = []
    for key in raws:
        longitude.append(raws[key]["loc"]["lon"])
        latitude.append(raws[key]["loc"]["lat"])
        stid.append(raws[key]["loc"]["stid"])
    pts = pd.DataFrame({
        "longitude" : longitude,
        "latitude" : latitude,
        "stid": stid
    })

    # Perform spatial interp 
    ds_pts = ds.herbie.pick_points(pts, method = "nearest", k=1)   
    
    return ds_pts



def format_hrrr_forecast():
    pass



if __name__ == '__main__':

    print("Imports successful, no executable code")
