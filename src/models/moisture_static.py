# Functions to fit static moisture models

import numpy as np
import math
import copy
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import os
import os.path as osp
import sys
import warnings
from xgboost import XGBRegressor


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
from utils import Dict, read_yml, print_dict_summary, read_pkl
import reproducibility

# Read Metadata
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
params_models = read_yml(osp.join(CONFIG_DIR, "params_models.yaml"))


# Static Models Code
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class MLModel(ABC):
    def __init__(self, params: dict):
        self.params = Dict(params)
        if type(self) is MLModel:
            raise TypeError("MLModel is an abstract class and cannot be instantiated directly")
        super().__init__()

    def _filter_params(self, model_cls):
        """Filters out parameters that are not part of the model constructor."""
        model_params = self.params.copy()
        valid_keys = model_cls.__init__.__code__.co_varnames
        filtered_params = {k: v for k, v in model_params.items() if k in valid_keys}
        return filtered_params
        
    
    def fit(self, X_train, y_train, weights=None):
        print(f"Fitting {self.params.mod_type} with params {self.params}")
        self.model.fit(X_train, y_train, sample_weight=weights)  

    def predict(self, X):
        print(f"Predicting with {self.params.mod_type}")
        preds = self.model.predict(X)
        return preds

        
    def test_eval(self, X_test, y_test, verbose=True):
        preds = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        # rmse_ros = np.sqrt(mean_squared_error(ros_3wind(y_test), ros_3wind(preds)))
        if verbose:
            print(f"Overall Test RMSE: {rmse}")
        errs = {
            'rmse': rmse
        }
        return errs        


class XGB(MLModel):
    
    def __init__(self, params: dict = None, random_state=None):
        if params is None:
            params = Dict(params_models["xgb"])
        
        super().__init__(params)
        model_params = self._filter_params(XGBRegressor) 
        if random_state is not None:
            reproducibility.set_seed(random_state)
            model_params.update({"random_state": random_state})

        self.model = XGBRegressor(**model_params)
        self.params['mod_type'] = "XGBoost"


class LM(MLModel):
    def __init__(self, params: dict = None):
        if params is None:
            params = Dict(params_models["lm"])
        
        super().__init__(params)
        model_params = self._filter_params(LinearRegression)
        self.model = LinearRegression(**model_params)
        self.params['mod_type'] = "LinearRegression"


