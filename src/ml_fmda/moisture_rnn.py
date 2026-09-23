import numpy as np
import math
import copy
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os.path as osp
import sys
from dateutil.relativedelta import relativedelta
from tensorflow.keras.callbacks import Callback, EarlyStopping, TerminateOnNaN
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import LSTM, SimpleRNN, Input, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from keras.saving import register_keras_serializable
import warnings
from itertools import product

# Set up project paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CURRENT_DIR = osp.dirname(osp.normpath(osp.abspath(__file__)))
PROJECT_ROOT = osp.dirname(osp.dirname(osp.normpath(CURRENT_DIR)))
sys.path.append(osp.join(PROJECT_ROOT, "src"))
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from . import reproducibility
from .utils import Dict, read_yml, is_consecutive_hours

# Read Metadata
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
params_models = read_yml(osp.join(CONFIG_DIR, "params_models.yaml"))

# RNN-Specific Utilities
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def predict_auto_batch(model,
                       X,
                       batch_sizes=(16384, 8192, 4096, 2048, 1024, 512, 256, 128, 32),
                       verbose=1, reset_state=True):
    """
    Predict using the largest batch size that fits in memory.

    NOTE: at this step for non-stateful model, batch size in predict is just a performance issue. The bigger the faster
    """
    last_exception = None
    for bs in batch_sizes:
        try:
            if verbose:
                print(f"Trying predict batch_size={bs}")
            preds = model.predict_cycle(X, batch_size=bs, verbose=verbose, reset_state=reset_state)
            if verbose:
                print(f"Success with batch_size={bs}")
            return preds
        except (MemoryError, tf.errors.ResourceExhaustedError) as e:
            last_exception = e
            if verbose:
                print(f"Failed with batch_size={bs}")

    raise RuntimeError(
        "All batch sizes failed during prediction."
    ) from last_exception    


def warp_weights(weights0, bi_warp, bf_warp):
    """
    Given LSTM layer weights and time-warp parameters, return a new list
    of time-warped LSTM weights without modifying the input weights.
    """
    # Copy all arrays to avoid mutating the originals
    w_warped = [w.copy() for w in weights0]
    # Bias vector (Keras LSTM layout: [i, f, c, o])
    b = w_warped[2]
    # Infer number of LSTM units from bias length
    if b.ndim != 1 or b.shape[0] % 4 != 0:
        raise ValueError("Unexpected LSTM bias shape.")
    lstm_units = b.shape[0] // 4
    # Input gate biases (i)
    b[0:lstm_units] += bi_warp
    # Forget gate biases (f)
    b[lstm_units:2 * lstm_units] += bf_warp

    return w_warped

    
# RNN Data Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def staircase(df, sequence_length=12, stride=1, features_list=None, y_col="fm"):
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
    
    for i in range(0, len(df) - sequence_length + 1, stride):
        time_window = times[i : i + sequence_length]
        if is_consecutive_hours(time_window):
            X.append(data[i : i + sequence_length])
            y.append(target[i : i + sequence_length])
            t.append(time_window)

    X = np.array(X)
    y = np.array(y)[..., np.newaxis]  # Ensure y has extra singleton dimension
    t = np.array(t)[..., np.newaxis]  # Ensure y_times has extra singleton dimension

    return X, y, t


# RNN Model Class
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@register_keras_serializable()
class RNN_Flexible(Model):
    """
    Custom Class for RNN with flexible batch size and timesteps. Training and prediction can be on arbitrary batches of arbitrary length sequences. 

    Based on params, forces batch_size and timesteps to be None, and forces return sequences. Will raise warning if otherwise in params
    """
    
    def __init__(self, params: dict, random_state=None, **kwargs):
        super().__init__(**kwargs)

        #if params is None:
        #    params = Dict(params_models["rnn"])
        self.params = Dict(params)
        self.params.update({'n_features': len(params["features_list"])})
        
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
            weights=None, callbacks=[], validation_data=(None, None), return_epochs=False, *args, **kwargs):
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
            val = validation_data[0] is not None
            callbacks, early_stop = self._setup_callbacks(val)

            fit_args = {
                "epochs": epochs,
                "batch_size": batch_size,
                "callbacks": callbacks,
                "verbose": verbose_fit,
                **kwargs
            }
            
            if val:
                fit_args["validation_data"] = validation_data
            else:
                warnings.warn("Running fit with no validation data, consider setting epochs to smaller number to avoid overfitting")

            history = super().fit(X_train, y_train, **fit_args)      
            
            if plot_history:
                self.plot_history(history,plot_title)

            if return_epochs:
                # Epoch counting starts at 0, adding 1 for the count
                return early_stop.best_epoch + 1        

    def test_eval(self, X_test, y_test, verbose=False):
        """
        Runs predict and calculates accuracy metrics for given test set.
        Can also be used on validation data in hyperparameter tuning runs
        """
        preds = self.predict(X_test)
        # Overall MSE
        mse = mean_squared_error(y_test.flatten(), preds.flatten())
        
        # Per loc MSE
        batch_mse = np.array([
            mean_squared_error(y_test[i].reshape(-1), preds[i].reshape(-1))
            for i in range(y_test.shape[0])
        ])
        if verbose:
            print(f"Overall Test MSE: {mse}")
            print(f"Per-Location Mean Test MSE: {batch_mse.mean()}")
        errs = {
            'mse': mse,
            'loc_mse': batch_mse
        }
        return preds, errs
        


    
            
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

# Operational RNN class
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Class used during active prediction
## No fitting. Supports cyclical prediction, 
## recurrent state is saved and set as initial

class OperationalRNNPredictor(Model):
    """ 
    Lightweight RNN model for operational prediction.

    Builds the same flexible architecture as RNN_Flexible from a params dict,
    but omits training callbacks, history plotting, and evaluation helpers.
    """

    def __init__(self, params: dict, compile_model: bool = False, **kwargs):
        params = self._check_params(params)
        inputs, outputs, state_specs = self._build_model(params)
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        self.params = params
        self.state_specs = state_specs
        self._cycle_states = None

        if compile_model:
            optimizer = Adam(learning_rate=self.params["learning_rate"])
            self.compile(loss="mean_squared_error", optimizer=optimizer)

    @staticmethod
    def _check_params(params):
        """ 
        Force flexible, stateless sequence prediction settings.
        """
        if params is None:
            raise ValueError("params must be provided for OperationalRNNPredictor.")            

        params = copy.deepcopy(dict(params))
        params["n_features"] = len(params["features_list"])

        if params.get("timesteps") is not None:
            warnings.warn("timesteps should be None for flexible prediction. Overriding to None.")
            params["timesteps"] = None

        if params.get("return_sequences") is not True:
            warnings.warn("return_sequences should be True for operational prediction. Overriding to True.")
            params["return_sequences"] = True

        if params.get("stateful") is not False:
            warnings.warn("stateful should be False for operational prediction. Overriding to False.")
            params["stateful"] = False

        layer_count = len(params["hidden_layers"])
        if len(params["hidden_units"]) != layer_count or len(params["hidden_activation"]) != layer_count:
            raise ValueError("hidden_layers, hidden_units, and hidden_activation must have the same length.")
        return params

    @staticmethod
    def _build_hidden_layers(x, params):
        """ 
        Build hidden layers from the parallel params lists.
        """
        initial_state_inputs = []
        final_state_outputs = [] 
        state_specs = []
        for i, layer_type in enumerate(params["hidden_layers"]):
            units = params["hidden_units"][i]
            activation = params["hidden_activation"][i]

            if layer_type == "dense":
                x = layers.Dense(units=units, activation=activation)(x)
            elif layer_type == "dropout":
                x = layers.Dropout(params["dropout"])(x)
            elif layer_type == "rnn":
                h0 = Input(shape=(units,), name=f"rnn_{i}_h0")
                x, h = layers.SimpleRNN(
                    units=units,
                    activation=activation,
                    dropout=params["dropout"],
                    recurrent_dropout=params["recurrent_dropout"],
                    stateful=False,
                    return_sequences=True,
                    return_state=True
                )(x, initial_state=[h0])
                initial_state_inputs.append(h0)
                final_state_outputs.append(h)
                state_specs.append({
                    "layer_index": i,
                    "layer_type": "rnn",
                    "state_names": ["h"],
                    "units": units,
                })                
            elif layer_type == "lstm":
                h0 = Input(shape=(units,), name=f"lstm_{i}_h0")
                c0 = Input(shape=(units,), name=f"lstm_{i}_c0")
                x, h, c = layers.LSTM(
                    units=units,
                    activation=activation,
                    dropout=params["dropout"],
                    recurrent_dropout=params["recurrent_dropout"],
                    stateful=False,
                    return_sequences=True,
                    return_state=True
                )(x, initial_state=[h0, c0])
                initial_state_inputs.extend([h0, c0])
                final_state_outputs.extend([h, c])
                state_specs.append({
                    "layer_index": i,
                    "layer_type": "lstm",
                    "state_names": ["h", "c"],
                    "units": units,
                })
            elif layer_type == "attention":
                x = layers.Attention()([x, x])
            elif layer_type == "conv1d":
                kernel_size = params.get("kernel_size", 3)
                x = layers.Conv1D(
                    filters=units,
                    kernel_size=kernel_size,
                    activation=activation,
                    padding="same",
                )(x)
            else:
                raise ValueError(f"Unrecognized layer type: {layer_type}")

        return x, initial_state_inputs, final_state_outputs, state_specs

    @classmethod
    def _build_model(cls, params):
        """
        Build a flexible sequence-to-sequence prediction graph.
        """
        inputs = Input(batch_shape=(None, None, params["n_features"]))
        x, initial_state_inputs, final_state_outputs, state_specs = cls._build_hidden_layers(inputs, params)

        if params["output_layer"] == "dense":
            predictions = layers.Dense(
                units=params["output_dimension"],
                activation=params["output_activation"],
            )(x)
        else:
            raise ValueError(f"Unsupported output layer type: {params['output_layer']}")

        return [inputs] + initial_state_inputs, [predictions] + final_state_outputs, state_specs

    @classmethod
    def from_weights(cls, params, weights_path):
        model = cls(params=params)
        model.load_weights(weights_path)
        return model

    def _zero_cycle_states(self, batch_size, dtype=np.float32):
        """
        Create zero recurrent states matching the model's recurrent layers.
        """
        states = []
        for spec in self.state_specs:
            for _ in spec["state_names"]:
                states.append(np.zeros((batch_size, spec["units"]), dtype=dtype))
        return states

    def _validate_cycle_states(self, states):
        """
        Validate and flatten recurrent states supplied to predict_cycle.
        """
        if states is None:
            return None

        states = list(states)
        n_expected = sum(len(spec["state_names"]) for spec in self.state_specs)
        if len(states) != n_expected:
            raise ValueError(f"Expected {n_expected} recurrent state arrays, got {len(states)}.")
        return states

    def reset_cycle_states(self):
        """
        Clear stored recurrent states.
        """
        self._cycle_states = None

    def predict_cycle(self, X, reset_state=False, initial_states=None, return_states=False, **kwargs):
        """
        Stores recurrent states after prediction and continues from stored states if they exist. Used for operational prediction where input data might come in cycles

        Args
        =========
        X: ndarray, input data (nbatch, ntime, nfeatures)
        reset_state: bool, whether to reset recurrent states (to zeros by default). Use if predicting at a new location or time
        initial_states: list, optional flat list of recurrent states. If None,
            use stored states when available.
        return_states: bool, whether to return final recurrent states as a flat
            list ordered by recurrent layer, with [h] for SimpleRNN and [h, c]
            for LSTM.
        """
        if initial_states is not None:
            cycle_states = self._validate_cycle_states(initial_states)
        elif reset_state or self._cycle_states is None:
            x_array = np.asarray(X)
            cycle_states = self._zero_cycle_states(batch_size=x_array.shape[0], dtype=x_array.dtype)
        else:
            cycle_states = self._validate_cycle_states(self._cycle_states)

        outputs = super().predict([X] + cycle_states, **kwargs)
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        predictions = outputs[0]
        self._cycle_states = outputs[1:]

        if return_states:
            return predictions, self._cycle_states
        return predictions


class TimeWarpedFuelClassPredictors:
    """Build FM1, FM10, FM100, and FM1000 operational predictors from FM10 weights."""

    FUEL_CLASSES = ("fm1", "fm10", "fm100", "fm1000")
    WARPED_FUEL_CLASSES = ("fm1", "fm100", "fm1000")

    def __init__(self, params, weights_path, warps):
        """
        Args:
            params: Operational RNN architecture parameters for the FM10 model.
            weights_path: Path to the pretrained FM10 weights file.
            warps: Mapping from fm1, fm100, and fm1000 to (bi_warp, bf_warp).
        """
        expected_warp_classes = set(self.WARPED_FUEL_CLASSES)
        supplied_warp_classes = set(warps)
        if supplied_warp_classes != expected_warp_classes:
            raise ValueError(
                "warps must contain exactly fm1, fm100, and fm1000; "
                f"got {sorted(supplied_warp_classes)}."
            )

        self.params = copy.deepcopy(dict(params))
        self.warps = dict(warps)
        self.predictors = {
            "fm10": OperationalRNNPredictor.from_weights(
                params=self.params,
                weights_path=weights_path,
            )
        }

        base_weights = self.predictors["fm10"].get_weights()
        for fuel_class in self.WARPED_FUEL_CLASSES:
            predictor = OperationalRNNPredictor(params=self.params)
            predictor.set_weights(base_weights)

            lstm_layers = [
                layer for layer in predictor.layers
                if isinstance(layer, layers.LSTM)
            ]
            if len(lstm_layers) != 1:
                raise ValueError(
                    "Time-warped predictors currently require exactly one LSTM layer."
                )

            bi_warp, bf_warp = self.warps[fuel_class]
            lstm_layer = lstm_layers[0]
            lstm_layer.set_weights(
                warp_weights(lstm_layer.get_weights(), bi_warp, bf_warp)
            )
            self.predictors[fuel_class] = predictor
