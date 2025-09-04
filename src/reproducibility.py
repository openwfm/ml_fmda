# A collection of random seeds used to make code reproducible
# On import, it sets tensorflow to run in deterministic mode

import os
import random
import numpy as np

# Attempt to import TensorFlow and related functions.
try:
    import tensorflow as tf
    _tf_available = True
except ImportError:
    _tf_available = False

# Set common environment variables.
environ = {
    'PYTHONHASHSEED': '0',
    'TF_CPP_MIN_LOG_LEVEL': '2'
}
os.environ.update(environ)

def set_seed(seed=123):
    """
    Set reproducibility seeds in several places.
    
    If TensorFlow is installed, it sets TensorFlow-related seeds and 
    enables deterministic operations. Otherwise, it only sets seeds for
    Python's random module and NumPy.
    
    Example:
    --------
    >>> set_seed(42)
    """

    print('Resetting random seeds to %i' % seed)
    # Seed Python's random module.
    random.seed(seed)
    # Seed NumPy.
    np.random.seed(seed)
    # Update environment variables.
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = '1'

    if _tf_available:
        # Set TensorFlow seeds.
        tf.random.set_seed(seed)
        # Some TF versions have this additional utility.
        if hasattr(tf.keras.utils, 'set_random_seed'):
            tf.keras.utils.set_random_seed(seed)
        # Enable deterministic operations if available.
        if hasattr(tf.config.experimental, 'enable_op_determinism'):
            tf.config.experimental.enable_op_determinism()






