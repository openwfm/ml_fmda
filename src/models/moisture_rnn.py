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
from utils import Dict
from data_funcs import MLData

# RNN Data Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def staircase(x,y,timesteps,datapoints,return_sequences=False, verbose = False):
    # x [datapoints,features]    all inputs
    # y [datapoints,outputs]
    # timesteps: split x and y into samples length timesteps, shifted by 1
    # datapoints: number of timesteps to use for training, no more than y.shape[0]
    if verbose:
        print('staircase: shape x = ',x.shape)
        print('staircase: shape y = ',y.shape)
        print('staircase: timesteps=',timesteps)
        print('staircase: datapoints=',datapoints)
        print('staircase: return_sequences=',return_sequences)
    outputs = y.shape[1]
    features = x.shape[1]
    samples = datapoints-timesteps+1
    if verbose:
        print('staircase: samples=',samples,'timesteps=',timesteps,'features=',features)
    x_train = np.empty([samples, timesteps, features])
    if return_sequences:
        if verbose:
            print('returning all timesteps in a sample')
        y_train = np.empty([samples, timesteps, outputs])  # all
        for i in range(samples):
            for k in range(timesteps):
                x_train[i,k,:] = x[i+k,:]
                y_train[i,k,:] = y[i+k,:]
    else:
        if verbose:
            print('returning only the last timestep in a sample')
        y_train = np.empty([samples, outputs])
        for i in range(samples):
            for k in range(timesteps):
                x_train[i,k,:] = x[i+k,:]
            y_train[i,:] = y[i+timesteps-1,:]

    return x_train, y_train

def staircase_2(x,y,timesteps,batch_size=None,trainsteps=np.inf,return_sequences=False, verbose = False):
    # create RNN training data in multiple batches
    # input:
    #     x (,features)  
    #     y (,outputs)
    #     timesteps: split x and y into sequences length timesteps
    #                a.k.a. lookback or sequence_length    
    
    # print params if verbose
   
    if batch_size is None:
        raise ValueError('staircase_2 requires batch_size')
    if verbose:
        print('staircase_2: shape x = ',x.shape)
        print('staircase_2: shape y = ',y.shape)
        print('staircase_2: timesteps=',timesteps)
        print('staircase_2: batch_size=',batch_size)
        print('staircase_2: return_sequences=',return_sequences)
    
    nx,features= x.shape
    ny,outputs = y.shape
    datapoints = min(nx,ny,trainsteps)   
    if verbose:
        print('staircase_2: datapoints=',datapoints)
    
    # sequence j in a given batch is assumed to be the continuation of sequence j in the previous batch
    # https://www.tensorflow.org/guide/keras/working_with_rnns Cross-batch statefulness
    
    # example with timesteps=3 batch_size=3 datapoints=15
    #     batch 0: [0 1 2]      [1 2 3]      [2 3 4]  
    #     batch 1: [3 4 5]      [4 5 6]      [5 6 7] 
    #     batch 2: [6 7 8]      [7 8 9]      [8 9 10] 
    #     batch 3: [9 10 11]    [10 11 12]   [11 12 13] 
    #     batch 4: [12 13 14]   [13 14 15]    when runs out this is the last batch, can be shorter
    #
    # TODO: implement for multiple locations, same starting time for each batch
    #              Loc 1         Loc 2       Loc 3
    #     batch 0: [0 1 2]      [0 1 2]      [0 1 2]  
    #     batch 1: [3 4 5]      [3 4 5]      [3 4 5] 
    #     batch 2: [6 7 8]      [6 7 8]      [6 7 8] 
    # TODO: second epoch shift starting time at batch 0 in time
    
    # TODO: implement for multiple locations, different starting times for each batch
    #              Loc 1       Loc 2       Loc 3
    #     batch 0: [0 1 2]   [1 2 3]      [2 3 4]  
    #     batch 1: [3 4 5]   [4 5 6]      [5 6 57 
    #     batch 2: [6 7 8]   [7 8 9]      [8 9 10] 
    
    #
    #     the first sample in batch j starts from timesteps*j and ends with timesteps*(j+1)-1
    #     e.g. the final hidden state of the rnn after the sequence of steps [0 1 2] in batch 0
    #     becomes the starting hidden state of the rnn in the sequence of steps [3 4 5] in batch 1, etc.
    #     
    #     sample [0 1 2] means the rnn is used twice to map state 0 -> 1 -> 2
    #     the state at time 0 is fixed but the state is considered a variable at times 1 and 2 
    #     the loss is computed from the output at time 2 and the gradient of the loss function by chain rule which ends at time 0 because the state there is a constant -> derivative is zero
    #     sample [3 4 5] means the rnn is used twice to map state 3 -> 4 -> 5    #     the state at time 3 is fixed to the output of the first sequence [0 1 2]
    #     the loss is computed from the output at time 5 and the gradient of the loss function by chain rule which ends at time 3 because the state there is considered constant -> derivative is zero
    #     how is the gradient computed? I suppose keras adds gradient wrt the weights at 2 5 8 ... 3 6 9... 4 7 ... and uses that to update the weights
    #     there is only one set of weights   h(2) = f(h(1),w)  h(1) = f(h(0),w)   but w is always the same 
    #     each column is a one successive evaluation of h(n+1) = f(h(n),w)  for n = n_startn n_start+1,... 
    #     the cannot be evaluated efficiently on gpu because gpu is a parallel processor
    #     this of it as each column served by one thread, and the threads are independent because they execute in parallel, there needs to be large number of threads (32 is a good number)\
    #     each batch consists of independent calculations
    #     but it can depend on the result of the previous batch (that's the recurrent parr)
    
    
    
    max_batches = datapoints // timesteps
    max_sequences = max_batches * batch_size

    if verbose:
        print('staircase_2: max_batches=',max_batches)
        print('staircase_2: max_sequences=',max_sequences)
                                      
    x_train = np.zeros((max_sequences, timesteps, features)) 
    if return_sequences:
        y_train = np.empty((max_sequences, timesteps, outputs))
    else:
        y_train = np.empty((max_sequences, outputs ))
        
    # build the sequences    
    k=0
    for i in range(max_batches):
        for j in range(batch_size):
            begin = i*timesteps + j
            next  = begin + timesteps
            if next > datapoints:
                break
            if verbose:
                print('sequence',k,'batch',i,'sample',j,'data',begin,'to',next-1)
            x_train[k,:,:] = x[begin:next,:]
            if return_sequences:
                 y_train[k,:,:] = y[begin:next,:]
            else:
                 y_train[k,:] = y[next-1,:]
            k += 1   
    if verbose:
        print('staircase_2: shape x_train = ',x_train.shape)
        print('staircase_2: shape y_train = ',y_train.shape)
        print('staircase_2: sequences generated',k)
        print('staircase_2: batch_size=',batch_size)
    k = (k // batch_size) * batch_size
    if verbose:
        print('staircase_2: removing partial and empty batches at the end, keeping',k)
    x_train = x_train[:k,:,:]
    if return_sequences:
         y_train = y_train[:k,:,:]
    else:
         y_train = y_train[:k,:]

    if verbose:
        print('staircase_2: shape x_train = ',x_train.shape)
        print('staircase_2: shape y_train = ',y_train.shape)

    return x_train, y_train


# Dictionary of scalers, used to avoid multiple object creation and to avoid multiple if statements
scalers = {
    'minmax': MinMaxScaler(),
    'standard': StandardScaler() 
}


def batch_setup(ids, batch_size):
    """
    Sets up stateful batched training data scheme for RNN training.

    This function takes a list or array of identifiers (`ids`) and divides them into batches of a specified size (`batch_size`). If the last batch does not have enough elements to meet the `batch_size`, the function will loop back to the start of the identifiers and continue filling the batch until it reaches the required size.

    Parameters:
    -----------
    ids : list or numpy array
        A list or numpy array containing the ids to be batched.

    batch_size : int
        The desired size of each batch. 

    Returns:
    --------
    batches : list of lists
        A list where each element is a batch (itself a list) of identifiers. Each batch will contain exactly `batch_size` elements.

    Example:
    --------
    >>> ids = [1, 2, 3, 4, 5]
    >>> batch_size = 3
    >>> batch_setup(ids, batch_size)
    [[1, 2, 3], [4, 5, 1]]

    Notes:
    ------
    - If `ids` is shorter than `batch_size`, the returned list will contain a single batch where identifiers are repeated from the start of `ids` until the batch is filled.
    """   
    # Ensure ids is a numpy array
    x = np.array(ids)
    
    # Initialize the list to hold the batches
    batches = []
    
    # Use a loop to slice the list/array into batches
    for i in range(0, len(x), batch_size):
        batch = list(x[i:i + batch_size])
        
        # If the batch is not full, continue from the start
        while len(batch) < batch_size:
            # Calculate the remaining number of items needed
            remaining = batch_size - len(batch)
            # Append the needed number of items from the start of the array
            batch.extend(x[:remaining])
        
        batches.append(batch)
    
    return batches

def staircase_spatial(X, y, batch_size, timesteps, hours=None, start_times = None, verbose = True):
    """
    Prepares spatially formatted time series data for RNN training by creating batches of sequences across different locations, stacked to be compatible with stateful models.

    This function processes multi-location time series data by slicing it into batches and formatting it to fit into a recurrent neural network (RNN) model. It utilizes a staircase-like approach to prepare sequences for each location and then interlaces them to align with stateful RNN structures.

    Parameters:
    -----------
    X : list of numpy arrays
        A list where each element is a numpy array containing features for a specific location. The shape of each array is `(total_time_steps, features)`.

    y : list of numpy arrays
        A list where each element is a numpy array containing the target values for a specific location. The shape of each array is `(total_time_steps,)`.

    batch_size : int
        The number of sequences to include in each batch.

    timesteps : int
        The number of time steps to include in each sequence for the RNN.

    hours : int, optional
        The length of each time series to consider for each location. If `None`, it defaults to the minimum length of `y` across all locations.

    start_times : numpy array, optional
        The initial time step for each location. If `None`, it defaults to an array starting from 0 and incrementing by 1 for each location.

    verbose : bool, optional
        If `True`, prints additional information during processing. Default is `True`.

    Returns:
    --------
    XX : numpy array
        A 3D numpy array with shape `(total_sequences, timesteps, features)` containing the prepared feature sequences for all locations.

    yy : numpy array
        A 2D numpy array with shape `(total_sequences, 1)` containing the corresponding target values for all locations.

    n_seqs : int
        Number of sequences per location. Used to reset states when location changes. Hidden state of RNN will be reset after n_seqs number of batches

    Notes:
    ------
    - The function handles spatially distributed time series data by batching and formatting it for stateful RNNs.
    - `hours` determines how much of the time series is used for each location. If not provided, it defaults to the shortest series in `y`.
    - If `start_times` is not provided, it assumes each location starts its series at progressively later time steps.
    - The `batch_setup` function is used internally to manage the creation of location and time step batches.
    - The returned feature sequences `XX` and target sequences `yy` are interlaced to align with the expected input format of stateful RNNs.
    """
    
    # Generate ids based on number of distinct timeseries provided
    n_loc = len(y) # assuming each list entry for y is a separate location
    loc_ids = np.arange(n_loc)
    
    # Generate hours and start_times if None
    if hours is None:
        print("Setting total hours to minimum length of y in provided dictionary")
        hours = min(len(yi) for yi in y)
    if start_times is None:
        print(f"Setting Start times to offset by 1 hour by location, from 0 through {batch_size} (batch_size)")
        start_times = np.tile(np.arange(batch_size), n_loc // batch_size + 1)[:n_loc]
    elif start_times == "zeros":
        start_times = np.zeros(n_loc)

    
    # Set up batches
    loc_batch, t_batch =  batch_setup(loc_ids, batch_size), batch_setup(start_times, batch_size)
    if verbose:
        print(f"Location ID Batches: {loc_batch}")
        print(f"Start Times for Batches: {t_batch}")

    # Loop over batches and construct with staircase_2
    Xs = []
    ys = []
    for i in range(0, len(loc_batch)):
        locs_i = loc_batch[i]
        ts = t_batch[i]
        for j in range(0, len(locs_i)):
            t0 = int(ts[j])
            tend = t0 + hours
            # Create RNNData Dict
            # Subset data to given location and time from t0 to t0+hours
            k = locs_i[j] # Used to account for fewer locations than batch size
            X_temp = X[k][t0:tend,:]
            y_temp = y[k][t0:tend].reshape(-1,1)

            # Format sequences
            Xi, yi = staircase_2(
                X_temp, 
                y_temp, 
                timesteps = timesteps, 
                batch_size = 1,  # note: using 1 here to format sequences for a single location, not same as target batch size for training data
                verbose=False)
        
            Xs.append(Xi)
            ys.append(yi)    

    # Drop incomplete batches
    lens = [yi.shape[0] for yi in ys]
    n_seqs = min(lens)
    if verbose:
        print(f"Minimum number of sequences by location: {n_seqs}")
        print(f"Applying minimum length to other arrays.")
    Xs = [Xi[:n_seqs] for Xi in Xs]
    ys = [yi[:n_seqs] for yi in ys]

    # Interlace arrays to match stateful structure
    n_features = Xi.shape[2]
    XXs = []
    yys = []
    for i in range(0, len(loc_batch)):
        locs_i = loc_batch[i]
        XXi = np.empty((Xs[0].shape[0]*batch_size, timesteps, n_features))
        yyi = np.empty((Xs[0].shape[0]*batch_size, 1))
        for j in range(0, len(locs_i)):
            XXi[j::(batch_size)] =  Xs[locs_i[j]]
            yyi[j::(batch_size)] =  ys[locs_i[j]]
        XXs.append(XXi)
        yys.append(yyi)
    yy = np.concatenate(yys, axis=0)
    XX = np.concatenate(XXs, axis=0)

    if verbose:
        print(f"Spatially Formatted X Shape: {XX.shape}")
        print(f"Spatially Formatted y Shape: {yy.shape}")
    
    
    return XX, yy, n_seqs





