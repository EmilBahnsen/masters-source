from typing import *
import tensorflow as tf
import math
import numpy as np
from tf_qc import complex_type, float_type
from tf_qc.qc import append_zeros, partial_trace_v2

π = np.pi


# Random number of the unit disk
def random_unifrom_complex(shape: Any,
                           radius: int = 1,
                           dtype: tf.dtypes.DType = float_type,
                           seed: Any = None,
                           name: Any = None):
    def sample_point(*args):
        x = np.random.uniform(-radius, radius)
        y = np.random.uniform(-radius, radius)
        return (x, y) if x**2 + y**2 <= radius**2 else sample_point()
    sample_point = np.vectorize(sample_point)
    x, y = np.fromfunction(sample_point, shape)
    # x = tf.random.uniform(shape, -radius, radius)
    # y = tf.map_fn(lambda _x: tf.random.uniform((1,), -tf.sqrt(radius**2 - _x**2), tf.sqrt(radius**2 - _x**2)), x)
    return tf.complex(x, y, name=name)


def normalize_state_vectors(state_vectors: tf.Tensor):
    '''
    Normalize state vectors
    :param state_vectors with shape (n_vectors, length, 1)
    :return:
    '''
    root_sum_norm_squares = tf.sqrt(tf.reduce_sum(tf.square(tf.abs(state_vectors)), axis=-2, keepdims=True))
    return tf.math.xdivy(state_vectors, tf.cast(root_sum_norm_squares, state_vectors.dtype))


def random_state_vectors(n_vectors, n_qubits, seed = None):
    return normalize_state_vectors(random_unifrom_complex((n_vectors, 2**n_qubits, 1), seed=seed))


# rnd. point on simplex: https://arxiv.org/pdf/1102.4598.pdf
def random_simplex(shape, seed=None):
    x = tf.concat([tf.fill((*shape[:-1], 1), 0.),
                   tf.sort(tf.random.uniform((*shape[:-1], shape[-1]-1), 0, 1, seed=seed)),
                   tf.fill((*shape[:-1], 1), 1.)], -1)
    return x[..., 1:] - x[..., :-1]


def random_pure_states(shape, post_zeros: int = 0, seed=None):
    _shape = shape[:-1] if shape[-1] == 1 else shape
    if post_zeros != 0:
        _shape = _shape[:-1] + (_shape[-1]//2**post_zeros,)  # Don't make the states for zero
    s = random_simplex(_shape, seed)  # Here we must be carefol to sample the simplex at random
    # BUt the angles are just uniformly sampled
    angles = tf.cast(tf.concat([tf.fill((*_shape[:-1], 1), 0.),
                                tf.random.uniform((*_shape[:-1], _shape[-1]-1), 0, 2*π, seed=seed)], -1),
                     complex_type)
    result = tf.cast(tf.sqrt(s), complex_type) * tf.exp(1j * angles)
    if post_zeros != 0:
        # result = tf.reshape(result, [result.shape[0], -1, 1])
        result = tf.expand_dims(result, -1)
        result = append_zeros(result, post_zeros)
    result = tf.reshape(result, shape) if shape[-1] == 1 else result
    return result


def convert2sparse(tensor: tf.Tensor):
    idx = tf.where(tf.abs(tensor) < 1e-5)
    return tf.SparseTensor(idx, tf.gather_nd(tensor, idx), tf.shape(tensor))


def partial_trace__(states: tf.Tensor, subsystem: Union[int, List[int]]):
    # if isinstance(subsystem, int):
    #     subsystem = [subsystem]
    # # Convert to density matrices
    # if states.shape[-1] == 1:
    #     states = density_matrix(states)
    # n_qubits = intlog2(states.shape[-1])
    # batch_shape = states.shape[:-2]
    # # Flatten the tensor to one batch dimension, and then a series of 2d indexes
    # # that represent the states on the from
    # # C_n1..._m1... = <n1...|C|m1...>
    # states = tf.reshape(states, [tf.reduce_sum(batch_shape)] + [2] * 2 * n_qubits)
    # result = states
    # for sub_index in subsystem:
    #     sub_index += 1  # Skip the batch dim.
    #     for i in range()
    # # Now we have fewer qubits!
    # n_qubits_new = n_qubits - len(subsystem)
    # return tf.reshape(result, batch_shape + [2**n_qubits_new, 2**n_qubits_new])
    pass


def partial_trace(states: tf.Tensor, subsystem: Union[int, List[int]], n_qubits: int):
    return partial_trace_v2(states, subsystem, n_qubits)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data0 = normalize_state_vectors(random_unifrom_complex((10000, 3, 1)))
    data = tf.reshape(data0, [-1])
    x = tf.math.real(data)
    y = tf.math.imag(data)
    plt.figure()
    plt.plot(x, y, '.', alpha=0.3)
    plt.show()

    plt.figure()
    x = np.absolute(data0)
    # plt.plot(x[:, 0], x[:, 1], '.b', alpha=0.3)
    plt.plot(x[:, 0]**2, x[:, 1]**2, '.r', alpha=0.1)
    plt.show()

    # https://chaos.if.uj.edu.pl/~karol/pdf2/Trieste11.pdf (page 5)
    plt.figure()
    plt.title('karol')
    shape = (10000, 3, 1)
    data = tf.complex(tf.random.normal((10000, 3, 1)),  tf.random.normal((10000, 3, 1)))
    data = tf.random.uniform(shape, 0, 1) * tf.random.uniform(shape, 0, 2*np.pi)
    norm = tf.cast(tf.math.sqrt(tf.reduce_sum(tf.square(tf.abs(data)), axis=-2, keepdims=True)), dtype=tf.float32)
    print(data.shape, norm.shape)
    data = data / norm
    plt.plot(data[:, 0] ** 2, data[:, 1] ** 2, '.r', alpha=0.1)
    plt.show()

    # plt.figure()
    # plt.title('ingemar')
    # n = 10000
    # K = 3
    # shape = (n, K, 1)
    # ξ = tf.random.uniform(shape, 0, 1)
    # ν = tf.random.uniform(shape, 0, 2*π)
    # power = tf.broadcast_to([ [1/(2*(i+1))] for i in range(K) ], shape)
    # θ = tf.asin(tf.pow(ξ, power))
    # n =
    # plt.plot(data[:, 0] ** 2, data[:, 1] ** 2, '.r', alpha=0.1)
    # plt.show()

    # https://en.wikipedia.org/wiki/N-sphere
    x = tf.random.normal((5000, 3, 1))
    r = tf.sqrt(tf.reduce_sum(tf.square(x), axis=-2, keepdims=True))
    x_sph = x/r
    plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    plt.title('wiki')
    plt.plot(x_sph[:, 0] - 1/math.sqrt(2), x_sph[:, 1] - 1/math.sqrt(2), '.', markersize=2)
    plt.plot(x_sph[:, 0] ** 2, x_sph[:, 1] ** 2, '.r', alpha=0.1, markersize=2)
    plt.show()

    # rnd. point on simplex: https://arxiv.org/pdf/1102.4598.pdf
    def random_simplex(shape):
        x = tf.concat([tf.fill((*shape[:-1], 1), 0.),
                       tf.sort(tf.random.uniform((*shape[:-1], shape[-1]-1), 0, 1)),
                       tf.fill((*shape[:-1], 1), 1.)], -1)
        return x[..., 1:] - x[..., :-1]

    def random_states(shape):
        _shape = shape[:-1] if shape[-1] == 1 else shape
        s = random_simplex(_shape)
        angles = tf.cast(tf.concat([tf.fill((*_shape[:-1], 1), 0.),
                                    tf.random.uniform((*_shape[:-1], _shape[-1]-1), 0, 2*π)], -1),
                         complex_type)
        result = tf.cast(tf.sqrt(s), complex_type) * tf.exp(1j * angles)
        return tf.reshape(result, shape) if shape[-1] == 1 else result


    x = random_states((10000, 3, 1))
    x = tf.abs(x)
    ax.set_aspect('equal', 'box')
    plt.title('Mathematica')
    plt.plot(x[:, 0] - 1 / math.sqrt(2), x[:, 1] - 1 / math.sqrt(2), '.', markersize=2)
    plt.plot(x[:, 0] ** 2, x[:, 1] ** 2, '.r', alpha=0.1, markersize=2)
    plt.show()
