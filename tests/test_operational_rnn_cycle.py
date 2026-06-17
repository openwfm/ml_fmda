# Module to test operational RNN cycle prediction
# Checks that one full prediction matches two chopped predictions with stored state

import os.path as osp
import sys
import numpy as np
import tensorflow as tf

# Set up project paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CURRENT_DIR = osp.dirname(osp.normpath(osp.abspath(__file__)))
PROJECT_ROOT = osp.dirname(osp.normpath(CURRENT_DIR))
sys.path.append(osp.join(PROJECT_ROOT, "src"))

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from models.moisture_rnn_operational import OperationalRNNPredictor


def make_test_params(hidden_layers, hidden_units, hidden_activation):
    return {
        "features_list": ["x1", "x2", "x3"],
        "timesteps": None,
        "hidden_layers": hidden_layers,
        "hidden_units": hidden_units,
        "hidden_activation": hidden_activation,
        "dropout": 0.0,
        "recurrent_dropout": 0.0,
        "output_layer": "dense",
        "output_activation": "linear",
        "output_dimension": 1,
        "learning_rate": 0.001,
        "stateful": False,
        "return_sequences": True,
    }


def run_cycle_equivalence_check(params, label, expected_n_states):
    n_batch = 1
    n_times = 12
    n_features = 3
    split = 6

    X = np.random.random((n_batch, n_times, n_features)).astype(np.float32)
    X_first = X[:, :split, :]
    X_second = X[:, split:, :]

    model = OperationalRNNPredictor(params=params)

    print(f"Running full sequence prediction: {label}.")
    preds_full, states_full = model.predict_cycle(X, reset_state=True, return_states=True, verbose=0)

    print(f"Running chopped sequence prediction with stored recurrent state: {label}.")
    model.reset_cycle_states()
    preds_first, states_first = model.predict_cycle(X_first, reset_state=True, return_states=True, verbose=0)
    preds_second, states_second = model.predict_cycle(X_second, return_states=True, verbose=0)
    preds_chopped = np.concatenate([preds_first, preds_second], axis=1)

    max_abs_diff = np.max(np.abs(preds_full - preds_chopped))
    tolerance = np.finfo(preds_full.dtype).eps

    print(f"Case: {label}")
    print(f"Full prediction shape: {preds_full.shape}")
    print(f"Chopped prediction shape: {preds_chopped.shape}")
    print(f"Number of returned recurrent states: {len(states_full)}")
    print(f"Max absolute prediction difference: {max_abs_diff}")
    print(f"Tolerance: {tolerance}")

    assert preds_full.shape == (n_batch, n_times, 1), f"Unexpected full prediction shape: {preds_full.shape}"
    assert preds_chopped.shape == preds_full.shape, f"Unexpected chopped prediction shape: {preds_chopped.shape}"
    assert len(states_full) == expected_n_states, f"Expected {expected_n_states} states, got {len(states_full)}."
    assert len(states_first) == expected_n_states, f"Expected {expected_n_states} states, got {len(states_first)}."
    assert len(states_second) == expected_n_states, f"Expected {expected_n_states} states, got {len(states_second)}."
    assert np.isfinite(preds_full).all()
    assert np.isfinite(preds_chopped).all()
    assert max_abs_diff <= tolerance, (
        f"Full sequence prediction does not match chopped cycle prediction for {label}: "
        f"{max_abs_diff=} exceeds {tolerance=}"
    )

    print(f"Operational RNN cycle prediction test passed: {label}.")


if __name__ == "__main__":
    np.random.seed(123)
    tf.random.set_seed(123)

    run_cycle_equivalence_check(
        params=make_test_params(
            hidden_layers=["lstm", "dense"],
            hidden_units=[4, 3],
            hidden_activation=["tanh", "relu"],
        ),
        label="one LSTM layer",
        expected_n_states=2,
    )

    run_cycle_equivalence_check(
        params=make_test_params(
            hidden_layers=["lstm", "rnn", "dense"],
            hidden_units=[4, 5, 3],
            hidden_activation=["tanh", "tanh", "relu"],
        ),
        label="one LSTM layer plus one SimpleRNN layer",
        expected_n_states=3,
    )

    print("All operational RNN cycle prediction tests passed.")
