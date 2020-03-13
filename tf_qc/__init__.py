import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf
import os
from typing import *

# if 'TF_SET' not in os.environ:
#     print(os.environ)
#     tf.config.threading.set_intra_op_parallelism_threads(12)
#     tf.config.threading.set_inter_op_parallelism_threads(2)
#     tf.config.set_soft_device_placement(True)
#     os.environ['TF_SET'] = 'true'

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

float_type = tf.float32
complex_type = tf.complex64

# TODO: Use custom type of this kind?
QubitState = tf.Tensor
Matrix = tf.Tensor
QubitDensityMatrix = Matrix
QubitStateOrDM = Union[QubitState, QubitDensityMatrix]

# TESTS
# TODO: make unitary testes on all new matrices/layers
