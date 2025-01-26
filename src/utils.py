# General purpose utilities used within project

import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import logging
import sys
import inspect
import yaml
import hashlib
import pickle
import pandas as pd
import os
import os.path as osp
from urllib.parse import urlparse
import subprocess
# import tensorflow as tf
import shutil
from itertools import islice
from datetime import datetime
    

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


def rename_dict(input_dict, rename_mapping):
    """
    Renames the top-level keys of a dictionary based on a provided rename mapping. Mapping dictionary can be found in variable metadata yaml files 

    Parameters:
        input_dict (dict): The dictionary whose keys need to be renamed.
        rename_mapping (dict): A dictionary mapping old key names to new key names.

    Returns:
        dict: A new dictionary with renamed keys.
    
    """
    return {rename_mapping.get(key, key): value for key, value in input_dict.items()}

def remove_key_list(d, ls, verbose=False):
    for key in ls:
        if key in d:
            if verbose:
                print(f"Removing key {key} due to data flags")
            del d[key]

# def print_first(item_list,num=3,indent=0,id=None):
#     """
#     Print the first num items of the list followed by '...' 

#     :param item_list: List of items to be printed
#     :param num: number of items to list
#     """
#     indent_str = ' ' * indent
#     if id is not None:
#         print(indent_str, id)
#     if len(item_list) > 0:
#         print(indent_str,type(item_list[0]))
#     for i in range(min(num,len(item_list))):
#         print(indent_str,item_list[i])
#     if len(item_list) > num:
#         print(indent_str,'...')

def print_dict_summary(d,indent=0,first=[],first_num=3):
    """
    Prints a summary for each array in the dictionary, showing the key and the size of the array.

    Arguments:
     d (dict): The dictionary to summarize.
     first_items (list): Print the first items for any arrays with these names
    
    """
    indent_str = ' ' * indent
    for key, value in d.items():
        # Check if the value is list-like using a simple method check
        if isinstance(value, dict):
            print(f"{indent_str}{key}")
            print_dict_summary(value,first=first,indent=indent+5,first_num=first_num)
        elif isinstance(value,np.ndarray):
            if np.issubdtype(value.dtype, np.number):
                print(f"{indent_str}{key}: NumPy array of shape {value.shape}, min: {value.min()}, max: {value.max()}")
            else:
                # Handle non-numeric arrays differently 
                print(f"{indent_str}{key}: NumPy array of shape {value.shape}, type {value.dtype}")
        elif hasattr(value, "__iter__") and not isinstance(value, str):  # Check for iterable that is not a string
            print(f"{indent_str}{key}: Array of {len(value)} items")
        else:
            print(indent_str,key,":",value)
        if key in first:
            print_first(value,num=first_num,indent=indent+5)


# Utility to retrieve files from URL
def retrieve_url(url, dest_path, force_download=False):
    """
    Downloads a file from a specified URL to a destination path, using `curl` or `wget`.

    Parameters:
    -----------
    url : str
        The URL from which to download the file.
    dest_path : str
        The destination path where the file should be saved.
    force_download : bool, optional
        If True, forces the download even if the file already exists at the destination path.
        Default is False.

    Warnings:
    ---------
    Prints a warning if the file extension of the URL does not match the destination file extension.

    Raises:
    -------
    AssertionError:
        If the download fails and the file does not exist at the destination path.

    Notes:
    ------
    This function uses the `wget` command-line tool to download the file. Ensure that `wget` is 
    installed and accessible from the system's PATH.

    Prints:
    -------
    A message indicating whether the file was downloaded or if it already exists at the 
    destination path.
    """    
    # Determine which command is available (curl preferred)
    download_cmd = None
    if shutil.which("curl"):
        download_cmd = "curl"
    elif shutil.which("wget"):
        download_cmd = "wget"
    else:
        raise EnvironmentError("Neither curl nor wget is installed on the system.")
    
    if not osp.exists(dest_path) or force_download:
        print(f"Attempting to downloaded {url} to {dest_path} using {download_cmd}")
        target_extension = osp.splitext(dest_path)[1]
        url_extension = osp.splitext(urlparse(url).path)[1]
        if target_extension != url_extension:
            print("Warning: file extension from url does not match destination file extension")
        # Construct download command
        if download_cmd == "curl":
            command = f"curl -L -o {dest_path} {url}"
        elif download_cmd == "wget":
            command = f"wget -O {dest_path} {url}"
        
        
        subprocess.run(command, shell=True, check=True)
        assert osp.exists(dest_path)
        print(f"Successfully downloaded {url} to {dest_path}")
    else:
        print(f"Target data already exists at {dest_path}")


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
def read_pkl(file_path):
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
        print(f"loading file {file_path}")
        d = pickle.load(file)
    return d

        
## Function for Hashing numpy arrays 
def hash_ndarray(arr: np.ndarray) -> str:
    """
    Generates a unique hash string for a NumPy ndarray.

    Parameters:
    -----------
    arr : np.ndarray
        The NumPy array to be hashed.

    Returns:
    --------
    str
        A hexadecimal string representing the MD5 hash of the array.

    Notes:
    ------
    This function first converts the NumPy array to a bytes string using the `tobytes()` method, 
    and then computes the MD5 hash of this bytes string. Performance might be bad for very large arrays.
    
    Example:
    --------
    >>> arr = np.array([1, 2, 3])
    >>> hash_value = hash_ndarray(arr)
    >>> print(hash_value)
    '2a1dd1e1e59d0a384c26951e316cd7e6'
    """    
    # If input is list, attempt to concatenate and then hash
    if type(arr) == list:
        arr = np.vstack(arr)
        arr_bytes = arr.tobytes()
    else:
        # Convert the array to a bytes string
        arr_bytes = arr.tobytes()
    # Use hashlib to generate a unique hash
    hash_obj = hashlib.md5(arr_bytes)
    return hash_obj.hexdigest()
    
## Function for Hashing tensorflow models
def hash_weights(model):
    """
    Generates a unique hash string for a the weights of a given Keras model.

    Parameters:
    -----------
    model : A keras model
        The Keras model to be hashed.

    Returns:
    --------
    str
        A hexadecimal string representing the MD5 hash of the model weights.

    """
    # Extract all weights and biases
    weights = model.get_weights()
    
    # Convert each weight array to a string
    weight_str = ''.join([np.array2string(w, separator=',') for w in weights])
    
    # Generate a SHA-256 hash of the combined string
    weight_hash = hashlib.md5(weight_str.encode('utf-8')).hexdigest()
    
    return weight_hash



# Time Manipulation Funcitons
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def str2time(input):
    """
    Convert a single string timestamp or a list of string timestamps to corresponding datetime object(s).
    """
    if isinstance(input, str):
        return datetime.strptime(input.replace('Z', '+00:00'), '%Y-%m-%dT%H:%M:%S%z')
    elif isinstance(input, list):
        return [str2time(s) for s in input]
    else:
        raise ValueError("Input must be a string or a list of strings")

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

    

def filter_nan_values(t1, v1):
    # Filter out NaN values from v1 and corresponding times in t1
    valid_indices = ~np.isnan(v1)  # Indices where v1 is not NaN
    t1_filtered = np.array(t1)[valid_indices]
    v1_filtered = np.array(v1)[valid_indices]
    return t1_filtered, v1_filtered

    
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



