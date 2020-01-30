from typing import *
import tensorflow as tf
import math
import numpy as np

# Random number of the unit disk
def random_unifrom_complex(shape: Any,
                           radius: int = 1,
                           dtype: tf.dtypes.DType = tf.dtypes.float64,
                           seed: Any = None,
                           name: Any = None):
    def sample_point(*args):
        x = np.random.uniform(-radius, radius)
        y = np.random.uniform(-radius, radius)
        return (x, y) if x**2 + y**2 <= radius**2 else sample_point()
    sample_point = np.vectorize(sample_point)
    x, y = np.fromfunction(sample_point, shape)
    return tf.complex(x, y, name=name)

def normalize_state_vectors(state_vectors: tf.Tensor):
    '''
    Normalize state vectors
    :param state_vectors with shape (n_vectors, length, 1)
    :return:
    '''
    root_sum_norm_squares = tf.sqrt(tf.reduce_sum(tf.square(tf.abs(state_vectors)), axis=1, keepdims=True))
    return tf.math.xdivy(state_vectors, tf.cast(root_sum_norm_squares, state_vectors.dtype))


def random_state_vectors(n_vectors, n_qubits, seed = None):
    return normalize_state_vectors(random_unifrom_complex((n_vectors, 2**n_qubits, 1), seed=seed))


def intlog2(x: int):
    return x.bit_length() - 1


