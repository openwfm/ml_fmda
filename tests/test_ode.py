# Module to test ODE+KF


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
from utils import read_pkl, hash_ndarray
import models.moisture_models as mm
import data_funcs

expected_model_hash = "0d09cbb16f29ceec4b39f11d5032cd54"

if __name__ == '__main__':

    # Read test data
    ml_data = read_pkl("data/test_data/test_ml_dat.pkl")
    
    # Get Test Cross-Val Period
    train_times, val_times, test_times = data_funcs.cv_time_setup("2023-01-05T00:00:00Z", train_hours=48*2, forecast_hours=48)

    # Get Test Station List
    stids = [*ml_data.keys()]
    tr_sts, val_sts, te_sts = data_funcs.cv_space_setup(stids, random_state=42)

    # Build data
    ode_data = data_funcs.get_ode_data(ml_data, te_sts, test_times)

    # Run Model
    ode = mm.ODE_FMC()
    m, errs = ode.run_model(ode_data, hours=72, h2=24)
    model_hash = hash_ndarray(m)

    if model_hash != expected_model_hash:
        warning.warn("Hash of ODE model output doesn't match expected")
    else:
        print("TEST PASSED")
    