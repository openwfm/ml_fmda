# General purpose utilities used within project

import numpy as np
from datetime import datetime
import logging
import sys
import ast
import yaml
import pickle
import pandas as pd
import os
import os.path as osp

def save_yaml(yfile, output_dir, filename):
    """Save yaml once, safely across concurrent processes."""
    os.makedirs(output_dir, exist_ok=True)
    path = osp.join(output_dir, filename)

    try:
        with open(path, "x") as f:
            yaml.dump(yfile, f, default_flow_style=False, sort_keys=False)
    except FileExistsError:
        pass


class Dict(dict):
    """
    A dictionary that allows member access to its keys.
    A convenience class.
    """

    def __init__(self, d):
        """
        Updates itself with d.
        """
        self.update(d)

    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, item, value):
        self[item] = value

    def __getitem__(self, item):
        if item in self:
            return super().__getitem__(item)
        else:
            for key in self:
                if isinstance(key,(range,tuple)) and item in key:
                    return super().__getitem__(key)
            raise KeyError(item)

    def keys(self):
        if any([isinstance(key,(range,tuple)) for key in self]):
            keys = []
            for key in self:
                if isinstance(key,(range,tuple)):
                    for k in key:
                        keys.append(k)
                else:
                    keys.append(key)
            return keys
        else:
            return super().keys()

# Generic helper function to read yaml files
def read_yml(yaml_path, subkey=None):
    """
    Reads a YAML file and optionally retrieves a specific subkey.

    Parameters:
    -----------
    yaml_path : str
        The path to the YAML file to be read.
    subkey : str, optional
        A specific key within the YAML file to retrieve. If provided, only the value associated 
        with this key will be returned. If not provided, the entire YAML file is returned as a 
        dictionary. Default is None.

    Returns:
    --------
    dict or any
        The contents of the YAML file as a dictionary, or the value associated with the specified 
        subkey if provided.

    """    
    with open(yaml_path, 'r') as file:
        d = yaml.safe_load(file)
        if subkey is not None:
            d = d[subkey]
    return d
    
# Generic helper function to read pickle files
def read_pkl(file_path, verbose=True):
    """
    Reads a pickle file and returns its contents.

    Parameters:
    -----------
    file_path : str
        The path to the pickle file to be read.

    Returns:
    --------
    any
        The object stored in the pickle file.

    Prints:
    -------
    A message indicating the file path being loaded.

    Notes:
    ------
    This function uses Python's `pickle` module to deserialize the contents of the file. Ensure 
    that the pickle file was created in a safe and trusted environment to avoid security risks 
    associated with loading arbitrary code.

    """    
    with open(file_path, 'rb') as file:
        if verbose:
            print(f"loading file {file_path}")
        d = pickle.load(file)
    return d

        
# Time Manipulation Funcitons
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def str2time(input):
    """
    Convert string or list of strings to datetime, supporting multiple formats.
    """
    formats = [
        '%Y-%m-%dT%H:%M:%S%z',      # ISO 8601 with 'T'
        '%Y-%m-%d %H:%M:%S%z',      # ISO 8601 with space instead of 'T'
        '%Y-%m-%dT%H:%M:%S',        # No timezone
        '%Y-%m-%d %H:%M:%S',        # No timezone, space separator
    ]

    def parse(s):
        s = s.replace('Z', '+00:00')
        for fmt in formats:
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        raise ValueError(f"Unsupported datetime format: {s}")

    if isinstance(input, str):
        return parse(input)
    elif isinstance(input, list):
        return [parse(s) for s in input]
    elif isinstance(input, datetime):
        return input    
    else:
        raise ValueError("Input must be a string or a list of strings or datetime")

# Calculate hour of day (HOD) and day of year (DOY) from times
def calc_times(times):
    """
    Calculate hour of day (HOD) and day of year (DOY) from a given input timnes. Should work with a pandas series or a numpy array, also for datetime or timestamp

    Parameters:
        - times: numpy array or pandas series of datetime/timestamps

    Returns: (hod, doy) tuple of arrays
        hod and doy arrays
    """
    times = pd.to_datetime(times)
    hod=times.hour
    doy=times.dayofyear

    return hod, doy

def time_range(start, end, freq="1h"):
    """
    Wrapper function for pandas date range. Checks to allow for input of datetimes or strings
    """
    if (type(start) is str) and (type(end) is str):
        start = str2time(start)
        end = str2time(end)
    else:
        assert isinstance(start, datetime) and isinstance(end, datetime), "Args start and end must be both strings or both datetimes"

    times = pd.date_range(start, end, freq=freq)
    times = times.to_pydatetime()
    return times

def time_intp(t1, v1, t2):
    # Check if t1 v1 t2 are 1D arrays
    if t1.ndim != 1:
        # logging.error("Error: t1 is not a 1D array. Dimension: %s", t1.ndim)
        # return None
        raise ValueError("")
    if v1.ndim != 1:
        # logging.error("Error: v1 is not a 1D array. Dimension %s:", v1.ndim)
        # return None
        raise ValueError("")
    if t2.ndim != 1:
        # logging.errorr("Error: t2 is not a 1D array. Dimension: %s", t2.ndim)
        # return None
        raise ValueError("")
    # Check if t1 and v1 have the same length
    if len(t1) != len(v1):
        # logging.error("Error: t1 and v1 have different lengths: %s %s",len(t1),len(v1))
        # return None
        raise ValueError("")
    t1_no_nan, v1_no_nan = filter_nan_values(t1, v1)
    # print('t1_no_nan.dtype=',t1_no_nan.dtype)
    # Convert datetime objects to timestamps
    t1_stamps = np.array([t.timestamp() for t in t1_no_nan])
    t2_stamps = np.array([t.timestamp() for t in t2])
    
    # Interpolate using the filtered data
    v2_interpolated = np.interp(t2_stamps, t1_stamps, v1_no_nan)
    if np.isnan(v2_interpolated).any():
        # logging.error('time_intp: interpolated output contains NaN')
        raise ValueError("")

    return v2_interpolated


def is_consecutive_hours(times):
    """
    Check whether input array are consecutive 1 hour increments
    """
    # Convert to numpy timedelta64[h] for hour differences
    time_diffs = np.diff(times).astype('timedelta64[h]')
    return np.all(time_diffs == np.timedelta64(1, 'h'))


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
        sys.exit(1)
        return None


if __name__ == '__main__':

    print("Imports successful, no executable code")
