from typing import *
import tensorflow as tf
import math

def random_unifrom_complex(shape: Any,
                           minval: int = 0,
                           maxval: Any = 1,
                           dtype: tf.dtypes.DType = tf.dtypes.float64,
                           seed: Any = None,
                           name: Any = None):
    r = tf.random.uniform(shape, minval, maxval, dtype, seed)
    theta = tf.random.uniform(shape, 0, 2*math.pi, dtype, seed)
    return tf.complex(r*tf.cos(theta),
                      r*tf.sin(theta),
                      name=name)

def normalize_state_vectors(state_vectors: tf.Tensor):
    '''
    Normalize state vectors
    :param state_vectors with shape (n_vectors, length, 1)
    :return:
    '''
    root_sum_norm_squares = tf.sqrt(tf.reduce_sum(tf.square(tf.abs(state_vectors)), axis=1, keepdims=True))
    return tf.math.xdivy(state_vectors, tf.cast(root_sum_norm_squares, state_vectors.dtype))


def random_state_vectors(n_vectors, n_qubits):
    return normalize_state_vectors(random_unifrom_complex((n_vectors, 2**n_qubits, 1)))
