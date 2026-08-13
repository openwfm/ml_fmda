#!/usr/bin/env python
# Script used to set up training data. Use before actual training called since when using reps we don't want to recreate the main data pool
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
from itertools import groupby
from pathlib import Path

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

# Config and Params
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
params = read_yml(osp.join(CONFIG_DIR, "params_models.yaml"), subkey="rnn")
project_paths = Dict(read_yml(osp.join(CONFIG_DIR, "paths.yaml")))


if __name__ == '__main__':


    # Optional seed argument. When calling this script with slurm arrays, the array numbers get passed in as random seed
    # If seed passed, import reproduciblity for deterministic ops and set seed
    # If no second argument, run without deterministic ops
    if len(sys.argv) != 2:
        print(f"Invalid arguments. {len(sys.argv)} was given but 2 expected")
        print("Example: python src/train_postprocessing.py models/TEST")
        sys.exit(1)

    # Get input conf, construct target directory, and write configs
    train_path = Path(sys.argv[1])
    seed_dirs = sorted(p for p in train_path.glob("seed_*") if p.is_dir())

    # Extract fitting accuracy, summarize over reps
    df_mse = pd.concat(
        [
            pd.read_csv(osp.join(seed_dir, "fitting_mse.csv"))
            .assign(seed=osp.basename(seed_dir))
            for seed_dir in seed_dirs
        ],
        ignore_index=True,
    )
    df_summary = df_mse.groupby("set").agg(
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        n_samples_mean=("n_samples", "mean"),
    )
    print(f"Writing fitting accuracy summary to: {osp.join(train_path, 'summary_fitting_accuracy.csv')}")
    df_summary.to_csv(osp.join(train_path, 'summary_fitting_accuracy.csv'), index=False)

    # Extract Median Accuracy Case
    train = df_mse[df_mse["set"] == "train"]
    median_mse = train["mse"].median()

    median_seed = train.loc[(train["mse"] - median_mse).abs().idxmin()]
    seed = median_seed["seed"]

    val_mse = df_mse.loc[
        (df_mse["seed"] == seed) & (df_mse["set"] == "val"),
        "mse"
    ].iloc[0]

    print(f"Writing median seed report to: {osp.join(train_path, 'median_seed.csv')}")
    pd.DataFrame([{
        "seed": int(seed.split("_")[1]),
        "train_mse": median_seed["mse"],
        "val_mse": val_mse,
        "median_train_mse": median_mse,
    }]).to_csv(osp.join(train_path, "median_seed.csv"), index=False)     
