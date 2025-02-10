# Module to test read cross validation setup

import os.path as osp
import sys
import warnings
import numpy as np
import pandas as pd

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
from utils import str2time, hash_ndarray, time_range
import data_funcs

ts = time_range("2023-01-01T00:00:00Z", "2023-01-31T10:00:00Z")
ts2 = time_range("2024-01-01T00:00:00Z", "2024-01-31T10:00:00Z")
seq_length = 12
n_features = 3

if __name__ == '__main__':
    print("Testing setting up cross validaiton")

    print()
    print("Testing Extracting Sequences")

    # Example should return 0 sequences
    data1 = np.random.randn(len(ts[:seq_length - 1]), n_features)
    test_1 = pd.DataFrame(data1, columns=[f"feature_{i}" for i in range(n_features)])
    test_1["date_time"] = ts[:seq_length - 1]
    test_1 = test_1[["date_time"] + [f"feature_{i}" for i in range(n_features)]]   
    X1 = data_funcs.extract_sequences(test_1, sequence_length = seq_length)
    assert X1.shape[0] == 0, f"TEST FAILED, expected 0 samples but got {X1.shape[0]}"

    
    # Example should return (2*seq_length-seq_length+1) samples
    data2 = np.random.randn(len(ts[:(seq_length*2)]), n_features)
    test_2 = pd.DataFrame(data2, columns=[f"feature_{i}" for i in range(n_features)])
    test_2["date_time"] = ts[:(seq_length*2)]
    test_2 = test_2[["date_time"] + [f"feature_{i}" for i in range(n_features)]]
    X2 = data_funcs.extract_sequences(test_2, sequence_length = seq_length)
    assert X2.shape == (seq_length*2 - seq_length + 1, seq_length, n_features), f"TEST FAILED, expected {seq_length*2 - seq_length + 1} samples but got {X2.shape[0]}"

    # Example should return 1 sequence
    data3 = np.random.randn(seq_length + seq_length // 2, n_features)
    test_3 = pd.DataFrame(data3, columns=[f"feature_{i}" for i in range(n_features)])
    # First part: a valid continuous stretch of seq_length
    test_3["date_time"] = np.concatenate([ts[:seq_length], ts2[: seq_length // 2]])
    test_3 = test_3[["date_time"] + [f"feature_{i}" for i in range(n_features)]]
    X3 = data_funcs.extract_sequences(test_3, sequence_length=seq_length)
    assert X3.shape[0] == 1, f"TEST FAILED, expected 1 sample but got {X3.shape[0]}"
    
    print("TESTS PASSED")