# Script used to train an RNN and save to a directory
# Intended for operational use, not for forecast analysis which
# has it's own set of scripts
# NOTE: the fastest GPU training relies on non-deterministic code, to make the code deterministic and reproducibile you can import the module reproducibility.py, but that will slow down training

import sys
import pickle
import os.path as osp
import os
from dateutil.relativedelta import relativedelta
import json
import pandas as pd
import numpy as np
import yaml
from sklearn.metrics import mean_squared_error
import time
from joblib import dump, load

# Set up project paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CURRENT_DIR = osp.dirname(osp.normpath(osp.abspath(__file__)))
PROJECT_ROOT = osp.dirname(osp.normpath(CURRENT_DIR))
sys.path.append(osp.join(PROJECT_ROOT, "src"))
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import read_yml, read_pkl, Dict, str2time, time_range
import data_funcs
#import reproducibility
from models.moisture_rnn import RNN_Flexible, RNNData, scale_3d

# Config and Params
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


if __name__ == '__main__':


    # Optional seed argument. When calling this script with slurm arrays, the array numbers get passed in as random seed
    # If seed passed, import reproduciblity for deterministic ops and set seed
    # If no second argument, run without deterministic ops
    # train_setup.py is run before that creates the directory and copies configs
    if len(sys.argv) not in [2, 3]:
        print(f"Invalid arguments. {len(sys.argv)} was given but 2 or 3 expected")
        print(('Usage: %s <model_dir> [seed]' % sys.argv[0]))
        print("<model_dir> is path to directory with ml_data and configs")
        print("Optional [seed] sets deterministic mode and random seed")
        print("Example: python src/train.py models/train_test/ 42")
        sys.exit(1)

    # Get input args
    t_dir = sys.argv[1]
    conf = Dict(read_yml(osp.join(t_dir, "train_config.yaml")))
    params = Dict(read_yml(osp.join(t_dir, "params.yaml")))

    # Setup output directory.
    tstart = str2time(conf.train_start)
    tend = str2time(conf.train_end)    
    
    seed = None
    out_dir = t_dir
    if len(sys.argv) == 3:
        try:
            seed = int(sys.argv[2])
        except ValueError:
            print("Seed must be an integer.")
            sys.exit(-1)

        import reproducibility
        reproducibility.set_seed(seed)
        out_dir = osp.join(out_dir, f"seed_{seed}")
        os.makedirs(out_dir, exist_ok=True)
    

    # Get needed data
    # Split train/val/test, use task_id for random seed
    ml_data_files = [osp.join(t_dir, "ml_data", f) for f in os.listdir(osp.join(t_dir, "ml_data"))]
    ml_data = {}
    print(f"Combining ML monthly data files")
    for f in ml_data_files:
        print(f"    reading and combining {f}")
        with open(f, "rb") as fp:
            ml_data_new = pickle.load(fp)
            for key, subdict in ml_data_new.items():
                if key not in ml_data:
                    ml_data[key] = subdict
                    continue
                ml_data[key]["data"] = pd.concat(
                    [ml_data[key]["data"], subdict["data"]],
                    ignore_index=True,
                )
                ml_data[key]["times"] = np.concatenate(
                    [ml_data[key]["times"], subdict["times"]]
                )
            del ml_data_new    


    # Extract a validation period for controlling early stopping, no test period
    # NOTE: if random_state set to anything besides None, determinstic TF triggered
    train, val, test = data_funcs.cv_data_wrap(ml_data, fstart=None, fend=None, tstart=tstart, tend=tend, val_hours=conf.val_hours, test_frac = conf.space_test_frac, random_state=None)    
    del ml_data

    # Train RNN 
    # Check if running deterministic, that should only be for testing as it is slower
    print('~'*75)
    print('Training RNN')
    deterministic = os.environ.get("TF_DETERMINISTIC_OPS", "0") == '1'
    if deterministic: print("    Tensorflow running in deterministic mode for reproduciblity"); print("    Warning: this is slower and should only be for testing")
    else: print("    Tensorflow running in non-deterministic mode for better performance, but won't be exactly reproducible")

    params["stride"] = conf.get("stride", 1)
    dat = RNNData(train, val, test=None, method="random", timesteps=params.timesteps, random_state=None, features_list = params.features_list, stride=params["stride"])
    del train, val, test

    dat.scale_data()
    rnn = RNN_Flexible(params=params)
    code_start = time.time() # time fitting to print out
    rnn.fit(dat.X_train, dat.y_train,
            validation_data=(dat.X_val, dat.y_val),
            batch_size = params["batch_size"],
            epochs = params["epochs"],
            verbose_fit = True,
            plot_history=False
           )    

    # Fitted and Val Metrics
    fitted = rnn.predict(dat.X_train)
    code_end = time.time()
    mse_fit = mean_squared_error(fitted.flatten(), dat.y_train.flatten())
    valpreds = rnn.predict(dat.X_val)
    mse_val = mean_squared_error(valpreds.flatten(), dat.y_val.flatten())
    df = pd.DataFrame({'set': ["train", "val"], 'n_samples': [fitted.flatten().shape[0], valpreds.flatten().shape[0]], 'mse': [mse_fit, mse_val]})
    df.to_csv(osp.join(out_dir, "fitting_mse.csv"), index=False)

    # Save Model
    print(f"Saving models weights to: {osp.join(out_dir, 'rnn.weights.h5')}")
    rnn.save_weights(osp.join(out_dir, "rnn.weights.h5"), overwrite=True)
    print(f"Saving model object to: {osp.join(out_dir, 'rnn.keras')}")
    rnn.save(osp.join(out_dir, "rnn.keras"))
    print(f"Saving data scaler to: {osp.join(out_dir, 'scaler')}")
    dump(dat.scaler, osp.join(out_dir, "scaler.joblib"))
    elapsed = code_end - code_start
    print(f"Code Runtime (seconds): {elapsed:.2f}")

