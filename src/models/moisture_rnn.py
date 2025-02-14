import numpy as np
import math
import matplotlib.pyplot as plt
import copy
from abc import ABC, abstractmethod
import xgboost as xg
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random
import reproducibility
import os.path as osp
import sys
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
from utils import Dict, is_consecutive_hours
import data_funcs
from data_funcs import MLData

# RNN Data Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# RNN Data Batching Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def staircase(df, sequence_length=12, features_list=None, y_col="fm"):
    """
    Get sliding-window style sequences from input data frame. 
    Checks date_time column for consecutive hours and only
    returns sequences with consecutive hours.

    NOTE: this replaces the staircase function from earlier versions of this project.

    Args:
        - df: (pandas dataframe) input data frame
        - sequence_length: (int) number of hours to set samples, equivalent to timesteps param in RNNs
        - features_list: (list) list of strings used to subset data
        - y_col: (str) target column name
        - verbose: (bool) whether to print debug info

    Returns:
        - X: (numpy array) array of shape (n_samples, sequence_length, n_features)
        - y: (numpy array) array of shape (n_samples, sequence_length, 1)
        - y_times: (numpy array) array of shape (n_samples, sequence_length, 1) containing datetime objects
    """
    
    times = df["date_time"].values

    if features_list is not None:
        data = df[features_list].values  # Extract feature columns
    
    target = df[y_col].values        # Extract target column
    X = []
    y = []
    t = []

    for i in range(len(df) - sequence_length + 1):
        time_window = times[i : i + sequence_length]
        if is_consecutive_hours(time_window):
            X.append(data[i : i + sequence_length])
            y.append(target[i : i + sequence_length])
            t.append(time_window)

    X = np.array(X)
    y = np.array(y)[..., np.newaxis]  # Ensure y has extra singleton dimension
    t = np.array(t)[..., np.newaxis]  # Ensure y_times has extra singleton dimension

    return X, y, t


def staircase_dict(dict0, sequence_length = 12, features_list=["Ed", "Ew", "rain"], y_col="fm", verbose=True):
    """
    Wraps extract_sequences to apply to a dictionary and run for each case.
    Intended to be run on train dict only
    """
    if verbose:
        print(f"Extracting all consecutive sequences of length {sequence_length}")
        print(f"Subsetting to features: {features_list}, target: {y_col}")    
    
    X_list, y_list, t_list = [], [], []
    
    for st, station_data in dict0.items():
        dfi = station_data["data"]  # Extract DataFrame
        Xi, yi, ti = staircase(dfi, sequence_length=sequence_length, features_list=features_list, y_col=y_col)

        # if verbose:
        #     print(f"Station: {st}")
        #     print(f"All Sequences Shape: {Xi.shape}")
        
        X_list.append(Xi)
        y_list.append(yi)
        t_list.append(ti)
        
    return X_list, y_list, t_list

def _batch_random(X_list, y_list, random_state = None):
    """
    Randomly shuffle samples
    """
    if random_state is not None:
        reproducibility.set_seed(random_state)  

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    indices = np.concatenate([np.full(len(x), i) for i, x in enumerate(X_list)])
    
    # Random shuffle location indicices
    locs = np.random.permutation(len(X_all))
    X_rand = X_all[locs]
    y_rand = y_all[locs]
    indices = indices[locs] 

    return X_rand, y_rand, indices



def build_training_batches(X_list, y_list, 
                           batch_size = 32, timesteps=12,
                           return_sequences=False, method="random", 
                           verbose=True, random_state=None
                          ):
    """

    Args:
        - method: (str) One of "random" or "stateful". NOTE: as of Feb 14 2025 stateful not implemented
    """


    if method == "random":
        X, y, loc_indices = _batch_random(X_list, y_list, random_state=random_state)
    elif method == "stateful":
        raise ValueError("Stateful not implemented yet for spatial data")
    else:
        raise ValueError(f"Unrecognized batching method: {method}")
    
    if verbose:
        print(f"{batch_size=}")
        print(f"{timesteps=}")
        print(f"X train shape: {X.shape}")
        print(f"y train shape: {y.shape}")
        print(f"Unique locations: {len(np.unique(loc_indices))}")
        print(f"Total Batches: {X.shape[0] // batch_size}")
    
    return X, y, loc_indices

class RNNData(MLData):
    """
    Custom class to handle RNN data. Performs data scaling and stateful batch structuring.
    In this context, a single "sample" from RNNData is a timeseries with dimensionality (timesteps, n_features)
    """
    def __init__(self, train, val=None, test=None, scaler="standard", features_list=None,
                 method="random", random_state=None):    
        super().__init__(train, val, test, scaler, features_list, random_state)
        
        
    def _setup_data(self, train, val, test, y_col="fm", method="random",
                    random_state = None, verbose=True):
        """
        Combines DataFrames under 'data' keys for train, val, and test. 
        Batch structure using staircase functions.

        Creates numpy ndarrays X_train, y_train, X_val, y_val, X_test, y_test
        """
        
        self.train_locs = [*train.keys()]
        
        if verbose:
            print(f"Subsetting input data to {self.features_list}")   
            
        train = data_funcs.sort_train_dict(train)
        # Get training samples with staircase, and construct batches
        # Subset features happens at this step
        X_list, y_list, t_list = staircase_dict(train, features_list = self.features_list)
        X_train, y_train, loc_train_indices = build_training_batches(
            X_list, y_list,
            method=method, random_state = random_state
        )
        self.X_train = X_train
        self.y_train = y_train


        if val:
            X_val = self._combine_data(val)
            self.y_val = X_val[y_col].to_numpy()
            self.X_val = X_val[self.features_list].to_numpy()            
        self.X_test, self.y_test = (None, None)
        if test:
            X_test = self._combine_data(test)
            self.y_test = X_test[y_col].to_numpy()
            self.X_test = X_test[self.features_list].to_numpy()

        if verbose:
            print(f"X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")
            if self.X_val is not None:
                print(f"X_val shape: {self.X_val.shape}, y_val shape: {self.y_val.shape}")
            if self.X_test is not None:
                print(f"X_test shape: {self.X_test.shape}, y_test shape: {self.y_test.shape}")        














## This is my attempt at stateful batching when samples from
## different time periods
## Does not work, too complicated going with random for now

# def stateful_batches(X_list, y_list, batch_size = 32, timesteps=12, 
#                            return_sequences=False, start_times="zeros", verbose=True):
#     """
#     Construct data for RNN training (and validation data) with format (batch_size, timesteps, features) 
#     Intended to be run on train set and validation set (if using)

#     Given list of staircase structured data, i.e. output of staircase_dict, create batches by getting samples from
#     each list element, so samples within a batch are from different physical locations.

#     If start_times is zeros, in the first batch, and any new batch with all new locations, select the 0th (aka first in python)
#     sample to build for the batch.

#     Args:
#         - X_list: (list) list of numpy ndarrays of predictors
#         - y_list: (list) list of numpy ndarrays of response data
#         - batch_size: (int) number of samples of length timesteps to include in a single iteration of weight updates
#         - timesteps: (int) number of discrete time steps that defines a single sample
#         - return_sequences: (bool) Whether to include all response y values for timesteps, or just last step
#         - start_times: if "zeros" all samples start at time 0. (Only one for now)
#     Returns:
#         XX, yy: tuple of structured predictors and outcomes variables. 
#             XX shape will be (num_samples, timesteps, features), where num_samples determined by batch size and input X length
#             yy shape will be (num_samples, 1) OR (num_samples, timesteps) if return sequences
#     """

#     # Run some checks
#     if len(X_list) != len(y_list):
#         raise ValueError(f"Mismatch data. {len(X_list)=}, {len(y_list)=}. Check they were created together")
#     if len(X_list) < batch_size:
#         raise ValueError(f"Batch size greter than number of locations. Method not implemented for this, try a smaller batch size. {len(X_list)=}, {batch_size=}.")

#     # Set up return objects    
#     X_batches = []
#     y_batches = []
#     loc_batches = []
#     t_batches = []
    
#     # Set up indices for first batch
#     loc_index = np.arange(batch_size)
#     loc_counter = loc_index.max() # used to iterate to new locations
#     loc_resets = []
#     X_set = set(np.arange(len(X_list)))
#     # t_index0 = np.arange(batch_size) # used to reset times on new location
#     # t_index = np.arange(batch_size)
#     t_index0 = np.zeros(batch_size)
#     t_index = np.zeros(batch_size)
    
#     b = 0 # batch index     
#     run = True
#     while run:
#         print("~"*75)
#         print(f"Batch {b}:")

#         print(f"Location Indices: {loc_index}")
#         print(f"Time Indices: {t_index}")
        
#         # Get data
#         X_batch = np.array([X_list[loc][int(t)] for loc, t in zip(loc_index, t_index)])
#         y_batch = np.array([y_list[loc][int(t)] for loc, t in zip(loc_index, t_index)])
#         if not return_sequences:
#             y_batch = y_batch[:, -1, :] # Get last time step of sequence

#         # Save batch info by appending
#         X_batches.append(X_batch.copy())
#         y_batches.append(y_batch.copy())
#         t_batches.append(t_index.copy())
#         loc_batches.append(loc_index.copy())
        
#         # Update indices for next iteration
#         t_index += timesteps # iterate time index by timesteps param

#         # Check times and locations, adjust if needed
#         for i in range(0, len(loc_index)):
#             loci = loc_index[i]
#             ti = t_index[i]
#             Xi = X_list[loci]
#             if Xi.shape[0] <= ti:
#                 # Condition triggered that requested time index is 
#                 # greater than samples available for given location
#                 # So iterate location index and reset time to t_index0
#                 t_index[i] = t_index0[i]
#                 loc_counter += 1
#                 new_loc_i = loc_counter % len(X_list)
#                 loc_resets.append(loc_index[i].copy()) # Keep track of which locations get reset

#                 if not set(loc_resets) - X_set:                
#                     # Condition triggered when maximum loc index has been reset to 0
#                     # Indicates we have cycles through all locations, STOP
#                     print(f"Stopping at batch {b}")
#                     run = False
#                     break
#                 loc_index[i] = new_loc_i
#                 print(f"Changing location {i} index to: {new_loc_i}")
#                 print(f"    With Time index to: {t_index0[i]}")
        
#         b += 1 # iterate batch


#     # return np.array(X_batches), np.array(y_batches), t_batches, loc_batches
#     return np.concatenate(X_batches, axis=0), np.concatenate(y_batches, axis=0), t_batches, loc_batches













