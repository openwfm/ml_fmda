import numpy as np
import sys
from pathlib import Path

from ml_fmda.moisture_rnn import OperationalRNNPredictor
from ml_fmda.utils import read_yml, Dict

CONFIG_DIR = Path(__file__).parent / "configs"
params = Dict(read_yml(CONFIG_DIR / "params_test.yaml"))


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



breakpoint()



