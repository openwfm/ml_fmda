# Set of functions and executable to retrieve and manipulate HRRR data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RAWS Data is retrieved using SynopticPy python package, which engages the Synoptic Data API
# Credit to Brian Blaylock for Herbie package

import pandas as pd
import herbie
from herbie import FastHerbie
from datetime import datetime
import numpy as np
###################
# FOR TESTING, In production, this is set in the shell file
import sys
sys.path.append("src")
###################
from utils import str2time

# Dataframe used to track names and metadata for various HRRR bands. The names used within the HRRR grib files differs from the xarray objects returned by Herbie, and we want to standardize those names to those used within this project from other data sources e.g. RAWS
hrrr_name_df = pd.DataFrame({
    'band_prs': [616, 620, 624, 629, 661, (561, 563, 565, 567, 569, 571, 573, 575, 577), (560, 562, 564, 566, 568, 570, 572, 574, 576), 612, 643, 610, 615, 613, 607, 639, 640],
    'hrrr_name': ['TMP', 'RH', "WIND", 'APCP',
                  'DSWRF', 'SOILW', "TSOIL", 'CNWAT', 'GFLUX', "ASNOW", "SNOD", "WEASD", "PRES", "SFCR", "FRICV"],
    'hrrr_level': ["2m", "2m", "10m", "surface", "surface", "multiple", "multiple",
                  "surface", "surface", "surface", "surface", "surface", "surface", "surface", "surface"],
    'herbie_str': ["TMP:2 m", "RH:2 m", "WIND:10 m", ":APCP:surface:2-3 hour acc", "DSWRF:surface", ":SOILW:", 
                   ":TSOIL:", "CNWAT:surface", "GFLUX:surface", "ASNOW:surface", ":SNOD:surface:3 hour fcst", ":WEASD:surface:2-3 hour acc", 
                   ":PRES:surface:3 hour fcst", ":SFCR:surface:3 hour fcst", ":FRICV:surface:3 hour fcst	"],
    'xarray_name': ["t2m", "r2", "si10", "tp", "dswrf", "soilw", "tsoil", "cnwat", "gflux", "unknown", "sde", "sdwe", "sp", "fsr", "fricv"],
    'fmda_name': ["temp", "rh", "wind", "precip_accum",
                 "solar", "soilm", "soilt", "canopyw", "groundflux", "asnow", "snod", "weasd", "pres", "rough", "fricv"],
    'descr': ['2m Temperature [K]', 
              '2m Relative Humidity [%]', 
              '10m Wind Speed [m/s]',
              'surface Total Precipitation [kg/m^2]',
              'surface Downward Short-Wave Radiation Flux [W/m^2]',
              'Volumetric Soil Moisture Content [Fraction]',
              'Soil Temperature [K]',
              'Plant Canopy Surface Water [kg/m^2]',
              'surface Ground Heat Flux [W/m^2]',
              'Total Snowfall [m]',
              'Snow Depth [m]',
              'Water Equivalent of Accumulated Snow Depth [kg/m^2]',
              'Surface air pressure [Pa]',
              'Surface Roughness [m]',
              'Frictional Velocity [m/s]'
             ],
    'notes': ["", "", "", "", "", "9 different depths, from 0-3m below ground", "9 different depths, from 0-3m below ground", "", "", "0-3 hr accumulated", "", 
              "0-3 hr accumulated, listed as `deprecated` in gribs", "", "", ""]
})

# Dataframe used to control which HRRR bands need to be retrieved given an input list of features
derived_feature_df = pd.DataFrame({
    'feature_name': ["Ed", "Ew", "rain", "hod", "doy"],
    'required_fmda_name': [("rh", "temp"), ("rh", "temp"), "precip_accum", "time", "time"]
})


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

    # Doing in-place modifying
    # return ds




def int2fstep(forecast_step):
    """
    Converts an integer forecast step into a formatted string with a prefix 'f' 
    and zero-padded to two digits. Format of HRRR data forecast hours

    Parameters:
    - forecast_step (int): The forecast step to convert. Must be an integer 
      between 0 and max_hour (inclusive).

    Returns:
    - str: A formatted string representing the forecast step, prefixed with 'f' 
      and zero-padded to two digits (e.g., 'f01', 'f02').

    Raises:
    - TypeError: If forecast_step is not an integer.
    - ValueError: If forecast_step is not between 0 and max_hour (inclusive).

    Example:
    >>> int2fstep(3)
    'f03'
    """
    if not isinstance(forecast_step, int):
        raise TypeError(f"forecast_step must be an integer.")
    if not (0 <= forecast_step):
        raise ValueError(f"forecast_step must be positive.")
        
    fstep='f'+str(forecast_step).zfill(2)
    return fstep