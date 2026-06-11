# Module to test RNN
# Runs train with validation data and predicts test data
# Uses random data, so accuracy not expected 


import os.path as osp
import sys
import warnings
import numpy as np

# Set up project paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CURRENT_DIR = osp.dirname(osp.normpath(osp.abspath(__file__)))
PROJECT_ROOT = osp.dirname(osp.normpath(CURRENT_DIR))
sys.path.append(osp.join(PROJECT_ROOT, "src"))
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import Dict, read_pkl, read_yml, hash_ndarray, hash_weights
import models.moisture_rnn as mrnn
import reproducibility

params = Dict(read_yml(osp.join(CONFIG_DIR, "params_models.yaml"), subkey="rnn"))


# Simulation Params
n_train = 100
n_test = 10
n_features = 3
sequence_length=48

if __name__ == '__main__':

    reproducibility.set_seed(123)
    
    # Simulate Random RNN Data
    X_train = np.random.rand(n_train, sequence_length, n_features)
    y_train = np.random.rand(n_train, sequence_length, 1)
    X_val = np.random.rand(n_test, sequence_length, n_features)
    y_val = np.random.rand(n_test, sequence_length, 1)
    X_test = np.random.rand(n_test, sequence_length, n_features)
    y_test = np.random.rand(n_test, sequence_length, 1)

    # Setup params
    params.update({
        'stateful': False,
        'return_sequences': True,
        'hidden_layers': ['lstm', 'dense', 'dropout'],
        'hidden_units': [3, 3, None], 
        'hidden_activation': ['tanh', 'relu', None],
        'batch_size': 4,
        'features_list': ['Ed', 'Ew', 'rain'],
        'timesteps': None,
        'epochs':2,
        'random_state': 42
    })

    # Run Model
    rnn = mrnn.RNN_Flexible(params = params)
    rnn.fit(X_train, y_train, 
            validation_data=(X_val, y_val),
            batch_size = params["batch_size"],
            epochs = params["epochs"],
            verbose_fit = True, plot_history=False
           )
    preds, errs = rnn.test_eval(X_test, y_test)

    # Checks
    assert preds.shape == (n_test, sequence_length, 1), f"Unexpected shape of RNN predictions, {preds.shape=}"
    assert np.isfinite(preds).all()
    assert all(np.all(np.isfinite(errs[metric])) for metric in errs)
    assert rnn.output_shape == (None, None, 1)

    nparams = 100 # LSTM(3 units, 3 inputs) = 84, Dense(3->3) = 12, Dense(3->1) = 4.
    assert rnn.count_params() == nparams, f"Expected {nparams} parameters, got {rnn.count_params()=}"

