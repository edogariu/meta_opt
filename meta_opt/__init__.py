import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Disables tensorRT, cuda warnings.
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
# it unavailable to JAX.
tf.config.set_visible_devices([], 'GPU')