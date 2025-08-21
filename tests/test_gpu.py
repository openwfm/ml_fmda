import tensorflow as tf
import warnings

with tf.device("/GPU:0"):
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    c = tf.matmul(a, b)
    print("Result is on device:", c.device)

if "GPU" in c.device:
    print("TEST PASSED")
else:
    warnings.warn("TEST FAILED: no GPU detected")

