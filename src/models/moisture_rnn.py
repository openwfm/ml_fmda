import numpy as np
import math
import copy
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import reproducibility
import os.path as osp
import sys
from dateutil.relativedelta import relativedelta
from tensorflow.keras.callbacks import Callback, EarlyStopping, TerminateOnNaN
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import LSTM, SimpleRNN, Input, Dropout, Dense
from tensorflow.keras.optimizers import Adam
import warnings


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
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import Dict, is_consecutive_hours, read_yml, hash_weights, hash_ndarray
import data_funcs
from data_funcs import MLData
import reproducibility

# Read Metadata
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
params_models = read_yml(osp.join(CONFIG_DIR, "params_models.yaml"))


# RNN Data Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
                           return_sequences=True, method="random", 
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

    if not return_sequences:
        y = y[:, -1, :]
    
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
        
        
    def _setup_data(self, train, val, test, y_col="fm", method="random", random_state = None, verbose=True):
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
            self.X_val = self._combine_data(val, self.features_list)
            self.y_val = self._combine_data(val, [y_col])
         
        self.X_test, self.y_test = (None, None)
        if test:
            self.X_test = self._combine_data(test, self.features_list)
            self.y_test = self._combine_data(test, [y_col])

        if verbose:
            print(f"X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")
            if self.X_val is not None:
                print(f"X_val shape: {self.X_val.shape}, y_val shape: {self.y_val.shape}")
            if self.X_test is not None:
                print(f"X_test shape: {self.X_test.shape}, y_test shape: {self.y_test.shape}")        
                
    def _combine_data(self, data_dict, features_list):
        """Combines all DataFrames under 'data' keys into a single DataFrame, with dimesionality (n_locs, n_times, features)."""
        return np.array([v["data"][features_list] for v in data_dict.values()])
    
    def scale_data(self, verbose=True):
        """
        Scales the training data using the set scaler. This requires
        reshaping the 3d train data to 2 before fitting the scaler
        NOTE: this converts pandas dataframes into numpy ndarrays.
        Tensorflow requires numpy ndarrays so this is intended behavior

        Parameters:
        -----------
        verbose : bool, optional
            If True, prints status messages. Default is True.

        Returns:
        ---------
        Nothing, modifies in place
        """        

        if not hasattr(self, "X_train"):
            raise AttributeError("No X_train within object. Run train_test_split first. This is to avoid fitting the scaler with prediction data.")
        if verbose:
            print(f"Scaling training data with scaler {self.scaler}, fitting on X_train")

        # Fit scaler on training data, need to reshape
        n_samples, timesteps, features = self.X_train.shape
        X_train2 = self.X_train.reshape(-1, features) 
        self.scaler.fit(X_train2)
        # Transform data using fitted scaler
        X_train2 = self.scaler.transform(X_train2)
        self.X_train = X_train2.reshape(n_samples, timesteps, features)
        
        if hasattr(self, 'X_val'):
            if self.X_val is not None:
                n_locs, timesteps, features = self.X_val.shape
                X_val = self.X_val.reshape(-1, features)
                X_val = self.scaler.transform(X_val)
                self.X_val = X_val.reshape(n_locs, timesteps, features)
        if self.X_test is not None:
            n_locs, timesteps, features = self.X_test.shape
            X_test = self.X_test.reshape(-1, features)
            X_test = self.scaler.transform(X_test)
            self.X_test = X_test.reshape(n_locs, timesteps, features)

    def inverse_scale(self, save_changes=False, verbose=True):
        """
        Inversely scales the data to its original form. Either save changes internally, or return tuple X_train, X_val, X_test. Need to
        reshape 3d train array for this

        Parameters:
        -----------
        return_X : str, optional
            Specifies what data to return after inverse scaling. Default is 'all_hours'.
        save_changes : bool, optional
            If True, updates the internal data with the inversely scaled values. Default is False.
        verbose : bool, optional
            If True, prints status messages. Default is True.
        """        
        if verbose:
            print("Inverse scaling data...")
        n_samples, timesteps, features = self.X_train.shape
        X_train2 = self.X_train.reshape(-1, features)
        X_train2 = self.scaler.inverse_transform(X_train2)
        X_train = X_train2.reshape(n_samples, timesteps, features)


        n_loc, timesteps, features = self.X_val.shape
        X_val = self.X_val.reshape(-1, features)
        X_val = self.scaler.inverse_transform(X_val)
        X_val = X_val.reshape(n_loc, timesteps, features)
        
        n_loc, timesteps, features = self.X_test.shape
        X_test = self.X_test.reshape(-1, features)
        X_test = self.scaler.inverse_transform(X_test)
        X_test = X_test.reshape(n_loc, timesteps, features)

        if save_changes:
            print("Inverse transformed data saved")
            self.X_train = X_train
            self.X_val = X_val
            self.X_test = X_test
        else:
            if verbose:
                print("Inverse scaled, but internal data not changed.")
            return X_train, X_val, X_test  












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


# RNN Model Class
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class RNN_Flexible(Model):
    """
    Custom Class for RNN with flexible batch size and timesteps. Training and prediction can be on arbitrary batches of arbitrary length sequences. 

    Based on params, forces batch_size and timesteps to be None, and forces return sequences. Will raise warning if otherwise in params
    """
    
    def __init__(self, n_features, params: dict = None, random_state=None):
        if params is None:
            params = Dict(params_models["rnn"])
        self.params = Dict(params)
        self.params.update({'n_features': n_features})
        
        if random_state is not None:
            reproducibility.set_seed(random_state)
            self.params.update({"random_state": random_state})
        
        # Define model type.
        if 'lstm' in self.params["hidden_layers"]:
            self.params['mod_type'] = "LSTM"
        elif 'rnn' in self.params["hidden_layers"]:
            self.params["mod_type"] = "SimpleRNN"
        else:
            self.params["mod"] = "NN"

        # Build model architectures based on input params
        self._check_params()
        self._build_model()        
        # Compile Models
        optimizer=Adam(learning_rate=self.params['learning_rate'])
        self.compile(loss='mean_squared_error', optimizer=optimizer)

    def _check_params(self):
        """
        Ensures return_sequences is True and batch_size and timesteps are None. 
        Raises a warning if they were set differently in params.
        """
        for param in ["timesteps"]:
            if self.params.get(param) is not None:
                warnings.warn(f"{param} should be None for flexible RNNs. Overriding to None.")
                self.params[param] = None
        
        if self.params.get("return_sequences") is not True:
            warnings.warn("return_sequences should be True for flexible RNNs. Overriding to True.")
            self.params["return_sequences"] = True          

    def _build_hidden_layers(self, x, stateful=False):
        """
        Helper function used to define neural network layers using TF functional interface.
        Has checks for the "return_sequences" setting. If a recurrent layer feeds in to 
        another recurrent layer or an attention layer, forces return_sequences to be True

        Uses params where hidden layers are listed in a single list, and corresponding hidden units and activation functions in a single list. If layer is attention or dropout, corresponding units and activation function should be None
        """
        params = self.params
     
        
        # Loop over each layer specified in 'hidden_layers'
        for i, layer_type in enumerate(params['hidden_layers']):
            units = params['hidden_units'][i]
            activation = params['hidden_activation'][i]
    
            if layer_type == 'dense':
                x = layers.Dense(units=units, activation=activation)(x)
    
            elif layer_type == 'dropout':
                x = layers.Dropout(params['dropout'])(x)
            
            elif layer_type == 'rnn':
                x = layers.SimpleRNN(units=units, activation=activation, dropout=params['dropout'], recurrent_dropout=params['recurrent_dropout'], stateful=stateful,
                                     return_sequences=True)(x)
            
            elif layer_type == 'lstm':
                x = layers.LSTM(units=units, activation=activation, dropout=params['dropout'], recurrent_dropout=params['recurrent_dropout'], stateful=stateful,
                                return_sequences=True)(x)    
            
            elif layer_type == 'attention':
                x = layers.Attention()([x, x])
            elif layer_type == 'conv1d':
                kernel_size = params.get('kernel_size', 3)
                x = layers.Conv1D(filters=units, kernel_size=kernel_size, activation=activation, padding='same')(x)
            else:
                raise ValueError(f"Unrecognized layer type: {layer_type}, skipping")
        
        return x     

    def _build_model(self):
        """
        Build the model architecture using functional API without creating an internal model object.
        """
        params = self.params
        
        inputs = Input(batch_shape=(None, None, params['n_features']))
        x = self._build_hidden_layers(inputs, stateful=params['stateful'])    
        
        if params['output_layer'] == 'dense':
            outputs = layers.Dense(units=params['output_dimension'], activation=params['output_activation'])(x)
        else:
            raise ValueError("Unsupported output layer type: {}".format(params['output_layer']))
        
        super().__init__(inputs=inputs, outputs=outputs)

    
    def _setup_callbacks(self, val=False):
        """
        Create list of callbacks used in fitting stage based on model params.
        Always use TerminateOnNaN to stop training if loss is ever NA.
        Other supported callbacks are ResetStates, which controls when hidden states
        of recurrent layers are reset, and EarlyStopping, which stops training when
        validation error stops improving for a certain number of times. Early stopping only
        used when validation data is used
        """
        callbacks = [TerminateOnNaN()]
        
        if self.params["reset_states"]:
            print("Using ResetStatesCallback.")
            callbacks=callbacks+[ResetStatesCallback(verbose=False)]

        if val:
            print("Using EarlyStoppingCallback")
            early_stop = EarlyStoppingCallback(patience = self.params['early_stopping_patience'])
            callbacks=callbacks+[early_stop]
        else:
            early_stop = None
        
        return callbacks, early_stop

    def is_stateful(self):
        """
        Checks whether any of the layers in the internal model (self.model_train) are stateful.

        Returns:
        bool: True if at least one layer in the model is stateful, False otherwise.
        
        This method iterates over all the layers in the model and checks if any of them
        have the 'stateful' attribute set to True. This is useful for determining if 
        the model is designed to maintain state across batches during training.

        Example:
        --------
        model.is_stateful()
        """          
        for layer in self.model_train.layers:
            if hasattr(layer, 'stateful') and layer.stateful:
                return True
        return False

    def plot_history(self, history, plot_title, create_figure=True):
        """
        Plots the training history. Uses log scale on y axis for readability.

        Parameters:
        -----------
        history : History object
            The training history object from model fitting. Output of keras' .fit command
        plot_title : str
            The title for the plot.
        """
        import matplotlib.pyplot as plt
        
        if create_figure:
            plt.figure(figsize=(10, 6))
        plt.semilogy(history.history['loss'], label='Training loss')
        if 'val_loss' in history.history:
            plt.semilogy(history.history['val_loss'], label='Validation loss')
        plt.title(f'{plot_title} Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.show()

    def fit(self, X_train, y_train, batch_size = 32, epochs=100,
            verbose_fit = False, verbose_weights=False, 
            plot_history=True, plot_title = '', 
            weights=None, callbacks=[], validation_data=None, return_epochs=False, *args, **kwargs):
            """
            Trains the model on the provided training data. Formats a list of callbacks to use within the fit method based on params input
    
            Parameters:
            -----------
            X_train : np.ndarray
                The input matrix data for training.
            y_train : np.ndarray
                The target vector data for training.
            plot_history : bool, optional
                If True, plots the training history. Default is True.
            plot_title : str, optional
                The title for the training plot. Default is an empty string.
            weights : optional
                Initial weights for the model. Default is None.
            callbacks : list, optional
                A list of callback functions to use during training. Default is an empty list.
            validation_data : tuple, optional
                Validation data to use during training, expected format (X_val, y_val). Default is None.
            return_epochs : bool
                If True, return the number of epochs that training took. Used to test and optimize early stopping
            """    
        
            # Check if GPU is available
            if tf.config.list_physical_devices('GPU'):
                print("Training is using GPU acceleration.")
            else:
                print("Training is using CPU.")
        
            if verbose_weights:
                print(f"Training simple RNN with params: {self.params}")
                
            # Setup callbacks, Check if validation data exists to modify callbacks
            val = validation_data is not None
            callbacks, early_stop = self._setup_callbacks(val)

            fit_args = {
                "epochs": epochs,
                "batch_size": batch_size,
                "callbacks": callbacks,
                "verbose": verbose_fit,
                **kwargs
            }
            
            if validation_data is not None:
                fit_args["validation_data"] = validation_data
            
            history = super().fit(X_train, y_train, **fit_args)      
            
            if plot_history:
                self.plot_history(history,plot_title)
                
            if verbose_weights:
                print(f"Fitted Weights Hash: {hash_weights(self.model_train)}")

            if return_epochs:
                # Epoch counting starts at 0, adding 1 for the count
                return early_stop.best_epoch + 1        

    def test_eval(self, X_test, y_test):
        """
        Runs predict and calculates accuracy metrics for given test set.
        Can also be used on validation data in hyperparameter tuning runs
        """
        preds = self.predict(X_test)
        # Overall RMSE
        rmse = np.sqrt(mean_squared_error(y_test.flatten(), preds.flatten()))
        print(f"Overall Test RMSE: {rmse}")
        
        # Per loc RMSE
        batch_rmse = np.array([
            np.sqrt(mean_squared_error(y_test[i].reshape(-1), preds[i].reshape(-1)))
            for i in range(y_test.shape[0])
        ])
        print(f"Per-Location Mean Test RMSE: {batch_rmse.mean()}")
        errs = {
            'rmse': rmse,
            'loc_rmse': batch_rmse
        }
        return errs
        


    
# class RNN():
#     import matplotlib.pyplot as plt
    
#     def __init__(self, n_features, params: dict = None, random_state=None):
#         if params is None:
#             params = Dict(params_models["rnn"])
#         self.params = Dict(params)
#         self.params.update({'n_features': n_features})
        
#         # super().__init__(params)
#         if random_state is not None:
#             reproducibility.set_seed(random_state)
#             self.params.update({"random_state": random_state})

#         # Define model type.
#         if 'lstm' in self.params["hidden_layers"]:
#             self.params['mod_type'] = "LSTM"
#         elif 'rnn' in self.params["hidden_layers"]:
#             self.params["mod_type"] = "SimpleRNN"
#         else:
#             self.params["mod"] = "NN"

#         # Build model architectures based on input params
#         self.model_train = self._build_model_train()
#         self.model_predict = self._build_model_predict()
#         # Compile Models
#         optimizer=Adam(learning_rate=self.params['learning_rate'])
#         self.model_train.compile(loss='mean_squared_error', optimizer=optimizer)
#         self.model_predict.compile(loss='mean_squared_error', optimizer=optimizer)

#     def _build_hidden_layers(self, x, stateful=False, return_sequences=True):
#         """
#         Helper function used to define neural network layers using TF functional interface.
#         Has checks for the "return_sequences" setting. If a recurrent layer feeds in to 
#         another recurrent layer or an attention layer, forces return_sequences to be True

#         Uses params where hidden layers are listed in a single list, and corresponding hidden units and activation functions in a single list. If layer is attention or dropout, corresponding units and activation function should be None
#         """
#         params = self.params
#         last_recurrent = None
        
#         # Identify the last RNN/LSTM layer, unless an Attention layer follows it
#         for i, layer_type in enumerate(params['hidden_layers']):
#             if layer_type in ['rnn', 'lstm']:
#                 # Check if there's an Attention layer following the current RNN/LSTM layer
#                 if i < len(params['hidden_layers']) - 1 and params['hidden_layers'][i + 1] == 'attention':
#                     continue
#                 last_recurrent = i        
        
#         # Loop over each layer specified in 'hidden_layers'
#         for i, layer_type in enumerate(params['hidden_layers']):
#             units = params['hidden_units'][i]
#             activation = params['hidden_activation'][i]
    
#             if layer_type == 'dense':
#                 x = layers.Dense(units=units, activation=activation)(x)
    
#             elif layer_type == 'dropout':
#                 x = layers.Dropout(params['dropout'])(x)
            
#             elif layer_type == 'rnn':

#                 print()
                
#                 is_last_recurrent = (i == last_recurrent)
#                 return_seqs_logic = not is_last_recurrent or return_sequences
#                 x = layers.SimpleRNN(units=units, activation=activation, dropout=params['dropout'], recurrent_dropout=params['recurrent_dropout'], stateful=stateful,
#                                      return_sequences=return_seqs_logic)(x)
            
#             elif layer_type == 'lstm':
#                 is_last_recurrent = (i == last_recurrent)
#                 return_seqs_logic = not is_last_recurrent or return_sequences
#                 x = layers.LSTM(units=units, activation=activation, dropout=params['dropout'], recurrent_dropout=params['recurrent_dropout'], stateful=stateful,
#                                 return_sequences=return_seqs_logic)(x)    
            
#             elif layer_type == 'attention':
#                 # Self-attention mechanism
#                 x = layers.Attention()([x, x])
#             elif layer_type == 'conv1d':
#                 kernel_size = params.get('kernel_size', 3)  # Check for kernel size, use 3 if missing
#                 x = layers.Conv1D(filters=units, kernel_size=kernel_size, activation=activation, padding='same')(x)
#             else:
#                 raise ValueError(f"Unrecognized layer type: {layer_type}, skipping")
        
#         return x
            
#     def _build_model_train(self):
#         """
#         Build training model, where input chape is (batch_size, timesteps, features). Adds the input and output layers and allows for return_sequences True or False, which changes the output shape of the final recurrent layer. When model.fit called for this model class, the training model is used. 
#         """
#         params = self.params
        
#         # Define the input layer with the specified batch size, timesteps, and features
#         inputs = Input(batch_shape=(params['batch_size'], params['timesteps'], params['n_features']))
#         x = inputs
#         # Build hidden layers
#         x = self._build_hidden_layers(x, stateful = params['stateful'], return_sequences = params['return_sequences'])    

#         # Add the output layer
#         if params['output_layer'] == 'dense':
#             outputs = layers.Dense(units=params['output_dimension'], activation=params['output_activation'])(x)
#         else:
#             raise ValueError("Unsupported output layer type: {}".format(params['output_layer']))
        
#         # Create the model
#         model = Model(inputs=inputs, outputs=outputs)

#         if self.params["verbose_weights"]:
#             print(f"Initial Weights Hash: {hash_weights(model)}")
        
#         return model

#     def _build_model_predict(self, return_sequences=True):
#         """
#         Build prediction model, where formal input shape is (None, None, features) allowing for prediction when batching new data based on (n_locations, n_times, features) for arbitrary n_locations and n_times. Forces return sequences True to allow for this flexible shape. When model.fit is called, weights from training model are copied over. When model.predeict is called, the prediction model is deployed
#         """
#         params = self.params
        
#         # Define the input layer with flexible batch size and sequence length
#         inputs = Input(shape=(None, params['n_features']))
#         x = inputs
#         # Build hidden layers
#         x = self._build_hidden_layers(x, stateful=False, return_sequences = True)    
    
#         # Add the output layer
#         if params['output_layer'] == 'dense':
#             outputs = layers.Dense(units=params['output_dimension'], activation=params['output_activation'])(x)
#         else:
#             raise ValueError("Unsupported output layer type: {}".format(params['output_layer']))
        
#         # Create the prediction model
#         model = Model(inputs=inputs, outputs=outputs)
#         return model

#     def _setup_callbacks(self, val=False):
#         """
#         Create list of callbacks used in fitting stage based on model params.
#         Always use TerminateOnNaN to stop training if loss is ever NA.
#         Other supported callbacks are ResetStates, which controls when hidden states
#         of recurrent layers are reset, and EarlyStopping, which stops training when
#         validation error stops improving for a certain number of times. Early stopping only
#         used when validation data is used
#         """
#         callbacks = [TerminateOnNaN()]
        
#         if self.params["reset_states"]:
#             print("Using ResetStatesCallback.")
#             callbacks=callbacks+[ResetStatesCallback(verbose=False)]

#         if val:
#             print("Using EarlyStoppingCallback")
#             early_stop = EarlyStoppingCallback(patience = self.params['early_stopping_patience'])
#             callbacks=callbacks+[early_stop]
#         else:
#             early_stop = None
        
#         return callbacks, early_stop

#     def is_stateful(self):
#         """
#         Checks whether any of the layers in the internal model (self.model_train) are stateful.

#         Returns:
#         bool: True if at least one layer in the model is stateful, False otherwise.
        
#         This method iterates over all the layers in the model and checks if any of them
#         have the 'stateful' attribute set to True. This is useful for determining if 
#         the model is designed to maintain state across batches during training.

#         Example:
#         --------
#         model.is_stateful()
#         """          
#         for layer in self.model_train.layers:
#             if hasattr(layer, 'stateful') and layer.stateful:
#                 return True
#         return False

#     def plot_history(self, history, plot_title, create_figure=True):
#         """
#         Plots the training history. Uses log scale on y axis for readability.

#         Parameters:
#         -----------
#         history : History object
#             The training history object from model fitting. Output of keras' .fit command
#         plot_title : str
#             The title for the plot.
#         """
        
#         if create_figure:
#             plt.figure(figsize=(10, 6))
#         plt.semilogy(history.history['loss'], label='Training loss')
#         if 'val_loss' in history.history:
#             plt.semilogy(history.history['val_loss'], label='Validation loss')
#         plt.title(f'{plot_title} Model loss')
#         plt.ylabel('Loss')
#         plt.xlabel('Epoch')
#         plt.legend(loc='upper left')
#         plt.show()

#     def fit(self, X_train, y_train, verbose_fit = False, verbose_weights=False, 
#                 plot_history=True, plot_title = '', 
#                 weights=None, callbacks=[], validation_data=None, return_epochs=False, *args, **kwargs):
#             """
#             Trains the model on the provided training data. Uses the fit method of the training model and then copies the weights over to the prediction model, which has a less restrictive input shape. Formats a list of callbacks to use within the fit method based on params input
    
#             Parameters:
#             -----------
#             X_train : np.ndarray
#                 The input matrix data for training.
#             y_train : np.ndarray
#                 The target vector data for training.
#             plot_history : bool, optional
#                 If True, plots the training history. Default is True.
#             plot_title : str, optional
#                 The title for the training plot. Default is an empty string.
#             weights : optional
#                 Initial weights for the model. Default is None.
#             callbacks : list, optional
#                 A list of callback functions to use during training. Default is an empty list.
#             validation_data : tuple, optional
#                 Validation data to use during training, expected format (X_val, y_val). Default is None.
#             return_epochs : bool
#                 If True, return the number of epochs that training took. Used to test and optimize early stopping
#             """        
#             if verbose_weights:
#                 print(f"Training simple RNN with params: {self.params}")
                
#             # Setup callbacks, Check if validation data exists to modify callbacks
#             val = validation_data is not None
#             callbacks, early_stop = self._setup_callbacks(val)

#             fit_args = {
#                 "epochs": self.params["epochs"],
#                 "batch_size": self.params["batch_size"],
#                 "callbacks": callbacks,
#                 "verbose": verbose_fit,
#                 **kwargs
#             }
            
#             if validation_data is not None:
#                 fit_args["validation_data"] = validation_data
            
#             history = self.model_train.fit(X_train, y_train, **fit_args)            
            
#             if plot_history:
#                 self.plot_history(history,plot_title)
                
#             if verbose_weights:
#                 print(f"Fitted Weights Hash: {hash_weights(self.model_train)}")
    
#             # Update Weights for Prediction Model
#             w_fitted = self.model_train.get_weights()
#             self.model_predict.set_weights(w_fitted)
    
#             if return_epochs:
#                 # Epoch counting starts at 0, adding 1 for the count
#                 return early_stop.best_epoch + 1

#     def predict(self, X_test, verbose=True):
#         if verbose:
#             print("Predicting test data")
#         preds = self.model_predict.predict(X_test)
        
#         return preds



            
# Callbacks
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ResetStatesCallback(Callback):
    """
    Class used to control reset of hidden states for recurrent models.
    """
    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose

    def _reset_rnn_states(self):
        """Reset states for all RNN layers in the model. Helper function that can be called at various times"""
        for layer in self.model.layers:
            if hasattr(layer, "reset_states"):
                layer.reset_states()
        if self.verbose:
            print("Reset hidden states.")
    
    def on_train_batch_end(self, batch, logs=None):
        """
        Reset after each batch of training. This treats batches as independent and intended for a non-stateful model. Would need to be adjusted for a stateful model
        """
        self._reset_rnn_states()
        if self.verbose:
            print(f"Reset hidden states at end of train batch {batch}")
            
    def on_epoch_end(self, epoch, logs=None):
        """
        Redundant with on_train_batch_end unless stateful model or a batch is skipped internally for some reason
        """
        self._reset_rnn_states()
        if self.verbose:
            print(f"Reset hidden states at end of epoch {epoch}")


class UpdatePredictionCallback(Callback):
    """
    Class used to copy weights over from the training model to the prediction model at the end of each epoch. This is done so that the flexibility of the input shape of the prediction model can be used in the validation step at the end of an epoch, rather than forcing the validation data into the training model input shape. 
    """
    def __init__(self, model_predict, verbose=False):
        super().__init__()
        self.verbose=verbose
        self.model_predict = model_predict  # Store reference to the prediction model

    def on_epoch_end(self, epoch, logs=None):
        if self.verbose:
            print("Updating Prediction Model")
        self.model_predict.set_weights(self.model.get_weights())  # Copy weights        



def EarlyStoppingCallback(patience=5):
    """
    Creates an EarlyStopping callback with the specified patience.

    Args:
        patience (int): Number of epochs with no improvement after which training will be stopped.

    Returns:
        EarlyStopping: Configured EarlyStopping callback.
    """
    return EarlyStopping(
        monitor='val_loss',
        patience=patience,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )
