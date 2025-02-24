# Module to test RNN
# Runs train with validation data and predicts test data
# Uses very simple "toy" model and few number of epochs,
# So not expected to be accurate, just expected to run without error


import os.path as osp
import sys
import warnings
import numpy as np

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
from utils import Dict, read_pkl, read_yml, hash_ndarray, hash_weights
import models.moisture_rnn as mrnn
import reproducibility

params = Dict(read_yml(osp.join(CONFIG_DIR, "params_models.yaml"), subkey="rnn"))

if __name__ == '__main__':

    reproducibility.set_seed(123)
    
    # Setup RNN Data
    rnn_dat = read_pkl("data/test_data/test_rnn_dat.pkl")
    rnn_dat.X_train = rnn_dat.X_train[0:100, :, :] # subset data
    rnn_dat.y_train = rnn_dat.y_train[0:100, :, :]
    rnn_dat.scale_data()
    
    # Setup params
    params.update({
        'stateful': False,
        'return_sequences': True,
        'hidden_units': [3, 3, None], 
        'batch_size': 4,
        'timesteps': None,
        'epochs':2,
        'random_state': 42
    })
    
    # Run Model
    rnn = mrnn.RNN_Flexible(n_features = rnn_dat.n_features, params = params)
    rnn.fit(rnn_dat.X_train, rnn_dat.y_train, 
            validation_data=(rnn_dat.X_val, rnn_dat.y_val),
            batch_size = params["batch_size"],
            epochs = params["epochs"],
            verbose_fit = True, plot_history=False
           )
    rnn.test_eval(rnn_dat.X_test, rnn_dat.y_test)
    
    print(f"Trained Model Weights Hash: {hash_weights(rnn)}")

    # print("TEST PASSED")
    