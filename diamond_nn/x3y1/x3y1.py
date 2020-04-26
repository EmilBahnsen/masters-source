from tf_qc import complex_type
from tf_qc.layers import ULayer
from tf_qc.utils import random_pure_states, normalize_state_vectors
from tf_qc.qc import s0, s1, tensor, density_matrix, trace
from txtutils import ndtotext_print
import tensorflow as tf
import math

π = math.pi

s000 = tensor([s0, s0, s0])
s001 = tensor([s0, s0, s1])
s010 = tensor([s0, s1, s0])
s011 = tensor([s0, s1, s1])
s100 = tensor([s1, s0, s0])
s101 = tensor([s1, s0, s1])
s110 = tensor([s1, s1, s0])
s111 = tensor([s1, s1, s1])

targets = [0, 1, 2, 3]
u_layer = ULayer(targets)

data = random_pure_states((10, 2**4, 1), post_zeros=0)
data_shape = data.shape

u_layer.build(data_shape)


def tree_qubit_state(amps):
    amps = tf.reshape(amps, (-1, 2**3, 1))
    return tf.cast(normalize_state_vectors(amps), complex_type)


with tf.device('cpu'):
    lin_size = 9
    space = tf.linspace(0., 1., lin_size)
    space = tf.cast(space, complex_type)
    input_states = \
        tf.reshape(tf.reshape(space, (lin_size, 1, 1)) * s000, (lin_size, 1, 1, 1, 1, 1, 1, 1, 8, 1)) + \
        tf.reshape(tf.reshape(space, (lin_size, 1, 1)) * s001, (1, lin_size, 1, 1, 1, 1, 1, 1, 8, 1)) + \
        tf.reshape(tf.reshape(space, (lin_size, 1, 1)) * s010, (1, 1, lin_size, 1, 1, 1, 1, 1, 8, 1)) + \
        tf.reshape(tf.reshape(space, (lin_size, 1, 1)) * s011, (1, 1, 1, lin_size, 1, 1, 1, 1, 8, 1)) + \
        tf.reshape(tf.reshape(space, (lin_size, 1, 1)) * s100, (1, 1, 1, 1, lin_size, 1, 1, 1, 8, 1)) + \
        tf.reshape(tf.reshape(space, (lin_size, 1, 1)) * s101, (1, 1, 1, 1, 1, lin_size, 1, 1, 8, 1)) + \
        tf.reshape(tf.reshape(space, (lin_size, 1, 1)) * s110, (1, 1, 1, 1, 1, 1, lin_size, 1, 8, 1)) + \
        tf.reshape(tf.reshape(space, (lin_size, 1, 1)) * s111, (1, 1, 1, 1, 1, 1, 1, lin_size, 8, 1))
    input_states = tensor([s0, input_states])  # Extend with the output state
    input_states = normalize_state_vectors(input_states)

    def apply_u(theta):
        u_layer.thetas.assign([theta])
        new_states = u_layer.matrix() @ input_states
        new_states_dm = density_matrix(new_states, [0])
        out_shape = ((lin_size,)*8)
        P0 = new_states_dm[:, 0, 0]
        return tf.reshape(P0, out_shape)

    for t in tf.linspace(0., π, 10):
        P0 = apply_u(t)
        mean = tf.reduce_mean(P0)
        var = tf.math.reduce_std(P0)
        print(f't = {t}: mean(var): {mean}({var})')
