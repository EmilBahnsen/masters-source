from tf_qc import complex_type, float_type
from tf_qc.layers import ULayer
from tf_qc.qc import tensor, trace, density_matrix, QubitState, s0, s1
from tf_qc.utils import normalize_state_vectors
import tensorflow as tf
from txtutils import ndtotext_print
import numpy as np
import matplotlib.pyplot as plt

π = np.pi

shape_in = tf.TensorShape((1, 2**12, 1))
U_layer1 = ULayer()
U_layer1.build(shape_in)


@tf.function
def layer1(w1, w2, w3, x1, x2, x3, t1, t2, t3):
    state_1_in = tensor([w1, x1])
    state_2_in = tensor([w2, x2])
    state_3_in = tensor([w3, x3])
    state_in = tensor([state_1_in, state_2_in, state_3_in])
    U_layer1.thetas.assign([t1, t2, t3])
    state_out = U_layer1.matrix() @ state_in
    out_layer1 = density_matrix(state_out, [0, 4, 8])
    return out_layer1


shape_l2_in = tf.TensorShape((1, 2**4, 1))
U_layer2 = ULayer()
U_layer2.build(shape_l2_in)


@tf.function
def layer2(state_w, dm_in, t4):
    dm_w = density_matrix(state_w)
    dm_in = tensor([dm_w, dm_in])
    # ndtotext_print(dm_in)
    # Remember this is a density matrix, so we must apply U on both sides
    U_layer2.thetas.assign([t4])
    U_matrix = U_layer2.matrix()
    dm_out = U_matrix @ dm_in @ tf.math.conj(U_matrix)
    return trace(dm_out, [1, 2, 3])


@tf.function
def x3y1_4(w1, w2, w3, w4, x1, x2, x3, t1, t2, t3, t4):
    w1 = normalize_state_vectors(w1)
    w2 = normalize_state_vectors(w2)
    w3 = normalize_state_vectors(w3)
    w4 = normalize_state_vectors(w4)
    x1 = normalize_state_vectors(x1)
    x2 = normalize_state_vectors(x2)
    x3 = normalize_state_vectors(x3)
    dm_out_l1 = layer1(w1, w2, w3, x1, x2, x3, t1, t2, t3)
    dm_out = layer2(w4, dm_out_l1, t4)
    # Make sure we kept unitarity along the way
    diff = tf.math.real(tf.linalg.trace(dm_out) - tf.cast(1., complex_type))
    assertion = tf.assert_less(diff, 1e-3, 'Not pure system')
    with tf.control_dependencies([assertion]):
        P0 = dm_out[..., 0, 0]
    return P0


def make_plot():
    P0s = []
    i_range = tf.cast(tf.linspace(0., 1., 20), complex_type)
    with tf.device('cpu'):
        for i in i_range:
            range_state = i*s0 + (i-1)*s1
            phi_p = (s0 + s1)/np.sqrt(2)
            x1 = tf.reshape(tf.convert_to_tensor([1, i, 1, 1, 1, 1, 1, 1], complex_type), (1, 8, 1))
            x2 = tf.reshape(tf.convert_to_tensor([1, 0, 0, 0, 0, 0, 0, 0], complex_type), (1, 8, 1))
            x3 = tf.reshape(tf.convert_to_tensor([1, 0, 0, 0, 0, 0, 0, 0], complex_type), (1, 8, 1))
            w1 = phi_p
            w2 = phi_p
            w3 = phi_p
            w4 = phi_p
            t1 = π/2
            t2 = π/2
            t3 = π/2
            t4 = π/2
            P0 = tf.math.real(x3y1_4(w1, w2, w3, w4, x1, x2 ,x3, t1, t2, t3, t4))
            P0s.append(P0[0].numpy())

    plt.figure()
    plt.plot(i_range, P0s)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()


class x3y1_4_model(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        single_qubit_init = tf.random_uniform_initializer(0, 1)
        diamond_time_init = tf.random_uniform_initializer(0, 2*π)
        self._w1 = tf.Variable(single_qubit_init((2, 1), float_type))
        self._w2 = tf.Variable(single_qubit_init((2, 1), float_type))
        self._w3 = tf.Variable(single_qubit_init((2, 1), float_type))
        self._w4 = tf.Variable(single_qubit_init((2, 1), float_type))

        shape_in = tf.TensorShape((1, 2 ** 12, 1))
        self.U_layer1 = ULayer()
        self.U_layer1.build(shape_in)

        shape_l2_in = tf.TensorShape((1, 2 ** 4, 1))
        self.U_layer2 = ULayer()
        self.U_layer2.build(shape_l2_in)

    @property
    def w1(self): return normalize_state_vectors(self._w1)

    @property
    def w2(self): return normalize_state_vectors(self._w2)

    @property
    def w3(self): return normalize_state_vectors(self._w3)

    @property
    def w4(self): return normalize_state_vectors(self._w4)

    def call(self, inputs, training=None, mask=None):
        x1, x2, x3 = inputs[..., 0, :, :], inputs[..., 1, :, :], inputs[..., 2, :, :]
        state_1_in = tensor([self.w1, x1])
        state_2_in = tensor([self.w2, x2])
        state_3_in = tensor([self.w3, x3])
        state_in = tensor([state_1_in, state_2_in, state_3_in])
        state_out = self.U_layer1.matrix() @ state_in
        dm_out_layer1 = density_matrix(state_out, [0, 4, 8])

        dm_w = density_matrix(self.w4)
        dm_in = tensor([dm_w, dm_out_layer1])
        U_matrix = self.U_layer2.matrix()
        dm_out = U_matrix @ dm_in @ tf.math.conj(U_matrix)
        dm_output = trace(dm_out, [1, 2, 3])
        P0 = dm_output[..., 0, 0]
        return P0

# input_data = tf.data.experimental.
import tensorflow_datasets as tfds
print(tfds.list_builders())

model = x3y1_4_model()
print(model)