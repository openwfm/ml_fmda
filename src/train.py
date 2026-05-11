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
params_models = read_yml(osp.join(CONFIG_DIR, "params_models.yaml"))


def build_training_dict(days, data_dir):
    print(f"    Days of Data Needed: {days.shape[0]}")
    print(f"    Earliest Day of Data: {days.min()}")
    print(f"    Latest Day of Data: {days.max()}")
    file_paths = [f"{data_dir}/{dt.strftime('%Y%m')}/fmda_{dt.strftime('%Y%m%d')}.pkl" for dt in days]
    all_exist = all(osp.exists(path) for path in file_paths)
    # For now, hard exit if not all data exists. Maybe relax in the future
    if not all_exist:
        print(f"Not all needed file paths exist for target analysis.\n Run src/ingest/get_fmda_data.py with desired dates and data directory to save. Exiting...")
        missing_paths = [path for path in file_paths if not osp.exists(path)]
        print("Missing files:")
        for path in missing_paths:
            print(path)
        sys.exit(-1)
    else:
        print(f"All Needed Data exists in {data_dir}, proceeding...")
    # Read and Format Data
    data = data_funcs.combine_fmda_files(file_paths)
    ml_dict = data_funcs.build_ml_data(data, verbose=False)
    df_valid = pd.read_csv(osp.join(PROJECT_ROOT, conf.valid_path))
    ml_dict = data_funcs.remove_invalid_data(ml_dict, df_valid)
    return ml_dict



if __name__ == '__main__':


    # Optional seed argument. When calling this script with slurm arrays, the array numbers get passed in as random seed
    # If seed passed, import reproduciblity for deterministic ops and set seed
    # If no second argument, run without deterministic ops
    if len(sys.argv) not in [2, 3]:
        print(f"Invalid arguments. {len(sys.argv)} was given but 2 or 3 expected")
        print(('Usage: %s <config_path> [seed]' % sys.argv[0]))
        print("<config_path> is path to yaml file setting up time frame and other analysis parameters")
        print("Optional [seed] sets deterministic mode and random seed")
        print("Example: python src/train.py etc/train_config_TEST.yaml 42")
        sys.exit(-1)

    # Get input args
    conf_path = sys.argv[1]
    conf = read_yml(conf_path)
    t_dir = conf["target_model_dir"]
    
    seed = None
    if len(sys.argv) == 3:
        try:
            seed = int(sys.argv[2])
        except ValueError:
            print("Seed must be an integer.")
            sys.exit(-1)

        import reproducibility
        reproducibility.set_seed(seed)
        t_dir = osp.join(t_dir, f"seed_{seed}")
    
    # Setup output directory.
    os.makedirs(osp.join(t_dir), exist_ok=True)

    params = params_models['rnn']
    with open(osp.join(t_dir, "train_config.yaml"), 'w') as f:
        yaml.dump(conf, f, default_flow_style=False, sort_keys=False)
    with open(osp.join(t_dir, "params.yaml"), 'w') as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)
    
    conf = Dict(conf)
    params = Dict(params)

    tstart = str2time(conf.train_start)
    tend = str2time(conf.train_end)
    print("~"*75)
    print(f"Training RNN from {tstart} to {tend}")
    print(f"Saving trained model to {t_dir}")
    print()
    
    # Build / Read training data dictionary
    # NOTE: stashed data organized in days, so read the full days that bracket input train times
    print(f"    Building Training Data")
    tdays = time_range(tstart, tend, freq="1d")
    data_file =  osp.join(PROJECT_ROOT, t_dir, "ml_data.pkl")
    if osp.exists(data_file):
        print(f"    ml_data already exists in train directory: {t_dir}")
        ml_data = read_pkl(osp.join(t_dir, 'ml_data.pkl'))
    else:
        data_dir = conf.data_dir
        ml_data = build_training_dict(tdays, data_dir)  
        print(f"    Writing training dictionary to {data_file}")
        with open(data_file, 'wb') as f:
            pickle.dump(ml_data, f)

    # Extract a validation period for controlling early stopping, no test period
    # NOTE: if random_state set to anything besides None, determinstic TF triggered
    train, val, test = data_funcs.cv_data_wrap(ml_data, fstart=None, fend=None, tstart=tstart, tend=tend, val_hours=conf.val_hours, test_frac = conf.space_test_frac, random_state=None)    


    # Train RNN 
    # Check if running deterministic, that should only be for testing as it is slower
    print('~'*75)
    print('Training RNN')
    deterministic = os.environ.get("TF_DETERMINISTIC_OPS", "0") == '1'
    if deterministic: print("    Tensorflow running in deterministic mode for reproduciblity"); print("    Warning: this is slower and should only be for testing")
    else: print("    Tensorflow running in non-deterministic mode for better performance, but won't be exactly reproducible")

    dat = RNNData(train, val, test=None, method="random", timesteps=params.timesteps, random_state=None, features_list = params.features_list)
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
    df.to_csv(osp.join(t_dir, "fitting_mse.csv"), index=False)

    # Save Model
    print(f"Saving models weights to: {osp.join(t_dir, 'rnn.weights.h5')}")
    rnn.save_weights(osp.join(t_dir, "rnn.weights.h5"), overwrite=True)
    print(f"Saving model object to: {osp.join(t_dir, 'rnn.keras')}")
    rnn.save(osp.join(t_dir, "rnn.keras"))
    print(f"Saving data scaler to: {osp.join(t_dir, 'scaler')}")
    dump(dat.scaler, osp.join(t_dir, "scaler.joblib"))
    elapsed = code_end - code_start
    print(f"Code Runtime (seconds): {elapsed:.2f}")

