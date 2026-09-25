# Intended conda envs that should work with this: ml_fmda_model, ml_gpu

import numpy as np
import sys
from pathlib import Path

from ml_fmda.moisture_rnn import OperationalRNNPredictor, predict_auto_batch
from ml_fmda.utils import read_yml, Dict

CONFIG_DIR = Path(__file__).parent 
params = Dict(read_yml(CONFIG_DIR / "params_test.yaml"))

# Some params to check

lstm_units = params.hidden_units[params.hidden_layers.index("lstm")]

# Time: hourly for one diurnal cycle
t = np.arange(0, 24, 1)

# Fixed relative humidity (%)
rh = 30

# Temperature: sinusoidal diurnal cycle
temp_mean = 20        # °C
temp_amplitude = 10   # °C
temp_c = temp_mean + temp_amplitude * np.sin(
    2 * np.pi * (t - 8) / 24
)

# Convert to Kelvin for Ed/Ew equations
temp_k = temp_c + 273.15

# Fuel moisture equilibrium values
Ed = (
    0.924 * rh**0.679
    + 0.000499 * np.exp(0.1 * rh)
    + 0.18 * (21.1 + 273.15 - temp_k)
      * (1 - np.exp(-0.115 * rh))
)

Ew = (
    0.618 * rh**0.753
    + 0.000454 * np.exp(0.1 * rh)
    + 0.18 * (21.1 + 273.15 - temp_k)
      * (1 - np.exp(-0.115 * rh))
)
rain = np.zeros(len(t))

# Turn into 3d array (nbatch, ntime, nfeats)
nbatch = 7
ntime = len(t)
expected_output_shape = (nbatch, ntime, 1)

Ed = np.tile(Ed, (nbatch, 1))
Ew = np.tile(Ew, (nbatch, 1))
rain = np.tile(rain, (nbatch, 1))
X = np.stack([Ed, Ew, rain], axis=-1)


# Build RNN
rnn = OperationalRNNPredictor(params=params)

# Tests
def test_predict():
    """
    Vanilla predict, should work as normal RNN object
    """
    preds = rnn.predict(X)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == expected_output_shape
    assert np.all(np.isfinite(preds))

def test_predict_is_stateless():
    """
    Check that multiple predict calls recreate similar output and recurrent state not being stored unexpectedly
    """
    preds1 = rnn.predict(X)
    preds2 = rnn.predict(X)
    assert np.allclose(preds1, preds2)


# Test cycle
def test_cycle_default():
    """Test basic cyclical prediction with default state handling.
    
    Verifies predictions have the expected shape and type.
    """
    preds = rnn.predict_cycle(X)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == expected_output_shape


def test_cycle_reset_state():
    """Test that reset_state=True starts prediction from zero recurrent state."""
    preds1 = rnn.predict_cycle(X, reset_state=True)
    preds2 = rnn.predict_cycle(X, reset_state=True)

    assert np.allclose(preds1, preds2)


def test_cycle_return_states():
    """Test that cyclical prediction can return predictions and recurrent states."""
    preds, states = rnn.predict_cycle(X, reset_state=True, return_states=True)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == expected_output_shape
    assert len(states) == 2

def test_cycle_return_states():
    """Test that cyclical prediction returns predictions and final LSTM states."""
    preds, states = rnn.predict_cycle(
        X, reset_state=True, return_states=True
    )

    assert isinstance(preds, np.ndarray)
    assert preds.shape == expected_output_shape
    assert len(states) == 2
    assert states[0].shape == (nbatch, lstm_units)
    assert states[1].shape == (nbatch, lstm_units)


def test_cycle_continues_state():
    """Test that sequential cycles reproduce a single continuous prediction."""
    preds_full = rnn.predict_cycle(X, reset_state=True)

    preds1 = rnn.predict_cycle(X[:, :2, :], reset_state=True)
    preds2 = rnn.predict_cycle(X[:, 2:, :])

    preds_cycle = np.concatenate([preds1, preds2], axis=1)

    assert np.allclose(preds_cycle, preds_full)

def test_cycle_initial_states():
    """Test that explicitly passing recurrent states reproduces continuous prediction."""
    preds_full = rnn.predict_cycle(X, reset_state=True)

    preds1, states = rnn.predict_cycle(
        X[:, :2, :],
        reset_state=True,
        return_states=True,
    )
    preds2 = rnn.predict_cycle(
        X[:, 2:, :],
        initial_states=states,
    )

    preds_cycle = np.concatenate([preds1, preds2], axis=1)

    assert np.allclose(preds_cycle, preds_full)


def test_cycle_batch_size():
    """Test that Keras processing batch size does not change predictions."""
    preds1 = rnn.predict_cycle(X, reset_state=True, batch_size=nbatch)
    preds2 = rnn.predict_cycle(X, reset_state=True, batch_size=3)

    assert np.allclose(preds1, preds2)


def test_cycle_batch_size():
    """Test that Keras processing batch size does not change predictions."""
    preds1 = rnn.predict_cycle(X, reset_state=True, batch_size=nbatch)
    preds2 = rnn.predict_cycle(X, reset_state=True, batch_size=3)
    preds3 = predict_auto_batch(rnn, X, verbose=0)

    assert np.allclose(preds1, preds2)
    assert np.allclose(preds1, preds3)
