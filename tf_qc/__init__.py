import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.set_soft_device_placement(True)

float_type = tf.float64
complex_type = tf.complex128

# TESTS
# TODO: make unitary testes on all new matrices/layers

