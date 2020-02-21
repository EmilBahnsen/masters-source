import tensorflow as tf
import os

# if 'TF_SET' not in os.environ:
#     print(os.environ)
#     tf.config.threading.set_intra_op_parallelism_threads(12)
#     tf.config.threading.set_inter_op_parallelism_threads(2)
#     tf.config.set_soft_device_placement(True)
#     os.environ['TF_SET'] = 'true'

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

float_type = tf.float64
complex_type = tf.complex128

# TESTS
# TODO: make unitary testes on all new matrices/layers
