# Set of functions and executable to retrieve and manipulate HRRR data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RAWS Data is retrieved using SynopticPy python package, which engages the Synoptic Data API
# Credit to Brian Blaylock for Herbie package

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
from utils import read_yml, Dict, time_intp, str2time, print_dict_summary
from data_funcs import rename_dict


# Read HRRR Metadata
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
hrrr_meta = read_yml(osp.join(CONFIG_DIR, "variable_metadata", "hrrr_metadata.yaml"))


def features_to_searchstr(flist):
    """
    Given features list, return dictionary of search strings to be used in Herbie package
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


def retrieve_hrrr(config):
    """
    Wrapper function to get HRRR data given config.
    """

    # Extract Config Info
    bbox = config.bbox
    start = str2time(config.start_time)
    end = str2time(config.end_time)
    features_list = config.features_list
    forecast_step = config.forecast_step
    print(f"Collecting HRRR data within {bbox} from {start} to {end}")

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


# Old Commented Out Stuff
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# # Dataframe used to track names and metadata for various HRRR bands. The names used within the HRRR grib files differs from the xarray objects returned by Herbie, and we want to standardize those names to those used within this project from other data sources e.g. RAWS
# hrrr_name_df = pd.DataFrame({
#     'band_prs': [616, 620, 624, 629, 661, (561, 563, 565, 567, 569, 571, 573, 575, 577), (560, 562, 564, 566, 568, 570, 572, 574, 576), 612, 643, 610, 615, 613, 607, 639, 640],
#     'hrrr_name': ['TMP', 'RH', "WIND", 'APCP',
#                   'DSWRF', 'SOILW', "TSOIL", 'CNWAT', 'GFLUX', "ASNOW", "SNOD", "WEASD", "PRES", "SFCR", "FRICV"],
#     'hrrr_level': ["2m", "2m", "10m", "surface", "surface", "multiple", "multiple",
#                   "surface", "surface", "surface", "surface", "surface", "surface", "surface", "surface"],
#     'herbie_str': ["TMP:2 m", "RH:2 m", "WIND:10 m", ":APCP:surface:2-3 hour acc", "DSWRF:surface", ":SOILW:", 
#                    ":TSOIL:", "CNWAT:surface", "GFLUX:surface", "ASNOW:surface", ":SNOD:surface:3 hour fcst", ":WEASD:surface:2-3 hour acc", 
#                    ":PRES:surface:3 hour fcst", ":SFCR:surface:3 hour fcst", ":FRICV:surface:3 hour fcst	"],
#     'xarray_name': ["t2m", "r2", "si10", "tp", "dswrf", "soilw", "tsoil", "cnwat", "gflux", "unknown", "sde", "sdwe", "sp", "fsr", "fricv"],
#     'fmda_name': ["temp", "rh", "wind", "precip_accum",
#                  "solar", "soilm", "soilt", "canopyw", "groundflux", "asnow", "snod", "weasd", "pres", "rough", "fricv"],
#     'descr': ['2m Temperature [K]', 
#               '2m Relative Humidity [%]', 
#               '10m Wind Speed [m/s]',
#               'surface Total Precipitation [kg/m^2]',
#               'surface Downward Short-Wave Radiation Flux [W/m^2]',
#               'Volumetric Soil Moisture Content [Fraction]',
#               'Soil Temperature [K]',
#               'Plant Canopy Surface Water [kg/m^2]',
#               'surface Ground Heat Flux [W/m^2]',
#               'Total Snowfall [m]',
#               'Snow Depth [m]',
#               'Water Equivalent of Accumulated Snow Depth [kg/m^2]',
#               'Surface air pressure [Pa]',
#               'Surface Roughness [m]',
#               'Frictional Velocity [m/s]'
#              ],
#     'notes': ["", "", "", "", "", "9 different depths, from 0-3m below ground", "9 different depths, from 0-3m below ground", "", "", "0-3 hr accumulated", "", 
#               "0-3 hr accumulated, listed as `deprecated` in gribs", "", "", ""]
# })
