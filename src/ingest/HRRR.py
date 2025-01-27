# Set of functions and executable to retrieve and manipulate HRRR data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# HRRR Data is retrieved using Brian Blaylock for Herbie package

import pandas as pd
import herbie
from herbie import FastHerbie
from datetime import datetime
import numpy as np
import os.path as osp
import sys
import xarray as xr
from dateutil.relativedelta import relativedelta

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
from utils import read_yml, Dict, time_intp, str2time, print_dict_summary, rename_dict


# Read HRRR Metadata
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
hrrr_meta = read_yml(osp.join(CONFIG_DIR, "variable_metadata", "hrrr_metadata.yaml"))


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

def retrieve_hrrr_train(start, end, bbox, all_features = True, forecast_step = 3):
    """
    Wrapper function to get HRRR data given config.

    Parameters
    -------------


    all_features : bool
        Logical argument whether to retrieve all possible features or not. If True, get all features in hrrr_metadata.yaml, if False use the config features list

    Notes
    --------------
    The intended use of the all_features argument is to collect everything for training so different feature subsets can be tested. Then for forecasting a trained model, use the features from the config file which are probably a smaller set and quicker to work with
    
    """

    # Extract Config Info
    start = str2time(start)
    end = str2time(cend)
    print(f"Collecting HRRR data within {bbox} from {start} to {end}")

    # Handle Features List
    if all_features:
        # All top level keys from hrrr_meta file into a list
        features_list = [*hrrr_meta.keys()]
    else:
        features_list = config.features_list
    
    # Adjust times for forecast step
    start = start - relativedelta(hours = forecast_step)
    end = end - relativedelta(hours = forecast_step)
    print(f"Shifting retrieval time to account for forecast step of {forecast_step}.")
    print(f"Data retrieval start: {start}")
    print(f"Data retrieval end: {end}")
    
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
    ds["date_time"] = ("time", pd.to_datetime(ds["valid_time"].values).tz_localize("UTC").to_numpy())

    
    return ds


def subset_hrrr_bbox(ds, bbox):
    """
    Subset HRRR spatial data with a spatial bounding box
    """     
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


