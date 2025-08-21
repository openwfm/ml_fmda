import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

results = []

# --- GPU check with matrix multiply ---
with tf.device("/GPU:0"):
    a = tf.random.normal([100, 100])
    b = tf.random.normal([100, 100])
    c = tf.matmul(a, b)
    print("Result is on device:", c.device)

if "GPU" in c.device:
    results.append("Matrix multiply on GPU: PASSED")
else:
    results.append("Matrix multiply on GPU: FAILED")

# --- Minimal training step ---
x = np.random.randn(128, 3).astype("float32")
y = np.random.randn(128, 1).astype("float32")

model = keras.Sequential([
    layers.Input(shape=(3,)),
    layers.Dense(8, activation="relu"),
    layers.Dense(1)
])
model.compile(optimizer="adam", loss="mse")
hist = model.fit(x, y, batch_size=32, epochs=1, verbose=0)

# check device placement of first weight tensor
device = model.layers[1].kernel.handle.device
loss_val = float(hist.history["loss"][-1])

if "GPU" in device:
    results.append(f"Model training step: PASSED (loss={loss_val:.4f})")
else:
    results.append(f"Model training step: FAILED (ran on CPU, loss={loss_val:.4f})")

# --- Final summary ---
print("\nTest Status:")
for r in results:
    print(r)

