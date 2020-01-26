import tensorflow as tf
from tensorflow.keras import layers

from . import float_type
from .qc import *

class QFTULayer(layers.Layer):
    def __init__(self, n_qubits=4):
        super(QFTULayer, self).__init__()
        self.n_qubits = None

    def build(self, input_shape):
        self.n_qubits = input_shape[-2].bit_length() - 1
        theta_init = tf.random_uniform_initializer(0, 2*π)
        self.thetas = tf.Variable(initial_value=theta_init(shape=(self.n_qubits-1,), dtype=float_type),
                                 trainable=True, name='thetas')

    def call(self, inputs, thetas=None):
        self.qft_U = self.matrix(thetas)
        return tf.matmul(self.qft_U, inputs)

    def matrix(self, thetas=None):
        if thetas:
            return qft_U(self.n_qubits, I4, thetas)
        else:
            return qft_U(self.n_qubits, I4, self.thetas)


# TEST
# x = tf.ones((10000000, 2**4, 1), dtype=complex_type)
# qft_u_layer = QFTULayer()
# t = time.process_time()
# qft_u_layer(x)
# exit()


class U3Layer(layers.Layer):
    def __init__(self):
        super(U3Layer, self).__init__()
        self.U3 = None
        self.thetas = None

    def build(self, input_shape):
        theta_init = tf.random_uniform_initializer(0, 2 * π)
        n_qubits = input_shape[-2].bit_length() - 1
        self.thetas = [tf.Variable(initial_value=theta_init(shape=(3,), dtype=float_type),
                                   trainable=True,
                                   dtype=float_type) for _ in range(n_qubits)]

    def call(self, inputs, thetas=None):
        self.U3 = self.matrix(thetas)
        return self.U3 @ inputs

    def matrix(self, thetas=None):
        if thetas:
            return U3(*thetas)
        else:
            return U3(*self.thetas)


class QFTLayer(layers.Layer):
    def __init__(self):
        super(QFTLayer, self).__init__()
        self.n_qubits = None

    def build(self, input_shape):
        self.n_qubits = input_shape[-2].bit_length() - 1

    def call(self, inputs, **kwargs):
        self.QFT = qft(self.n_qubits, I4)
        return tf.matmul(self.QFT, inputs)

    def matrix(self):
        return qft(self.n_qubits, I4)


class IQFTLayer(layers.Layer):
    def __init__(self):
        super(IQFTLayer, self).__init__()
        self.n_qubits = None

    def build(self, input_shape):
        self.n_qubits = input_shape[-2].bit_length() - 1

    def call(self, inputs, **kwargs):
        self.IQFT = iqft(self.n_qubits, I4)
        return tf.matmul(self.IQFT, inputs)

    def matrix(self):
        return iqft(self.n_qubits, I4)