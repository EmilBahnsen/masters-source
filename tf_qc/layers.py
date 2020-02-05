import tensorflow as tf
from tensorflow.keras import layers
from abc import ABCMeta, abstractmethod
from functools import reduce

from . import float_type, utils, complex_type
import tf_qc.qc as qc
from .qc import H, U, qft_U, I4, π, U3, iqft, qft, gate_expand_1toN, gate_expand_2toN, tensor


class QCLayer(layers.Layer, metaclass=ABCMeta):
    @abstractmethod
    def matrix(self, **kwargs) -> tf.Tensor:
        pass


class QFTULayer(QCLayer):
    def __init__(self, n_qubits=4):
        super(QFTULayer, self).__init__()
        self.n_qubits = None
        self.thetas = None

    def build(self, input_shape):
        self.n_qubits = utils.intlog2(input_shape[-2])
        theta_init = tf.random_uniform_initializer(0, 2*π)
        self.thetas = [tf.Variable(initial_value=theta_init(shape=(1,), dtype=float_type),
                                   trainable=True,
                                   name='thetas')
                       for _ in range(self.n_qubits-1)]

    def call(self, inputs, thetas=None):
        if thetas:
            return qft_U(self.n_qubits, inputs, thetas)
        else:
            return qft_U(self.n_qubits, inputs, self.thetas)

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


class HLayer(QCLayer):
    def __init__(self, target: int):
        super(HLayer, self).__init__()
        self.target = target
        self._matrix = None

    def build(self, input_shape: tf.TensorShape):
        n_qubits = utils.intlog2(input_shape[-2])
        self._matrix = gate_expand_1toN(H, n_qubits, self.target)

    def call(self, inputs, **kwargs):
        return self._matrix @ inputs

    def matrix(self, **kwargs) -> tf.Tensor:
        return self._matrix


class U3Layer(QCLayer):
    def __init__(self):
        super(U3Layer, self).__init__()
        self.U3 = None
        self.thetas = None

    def build(self, input_shape: tf.TensorShape):
        theta_init = tf.random_uniform_initializer(0, 2*π)  # It's an angle
        n_qubits = utils.intlog2(input_shape[-2])
        self.thetas = [tf.Variable(initial_value=theta_init(shape=(3,), dtype=float_type),
                                   trainable=True,
                                   dtype=float_type) for _ in range(n_qubits)]

    def call(self, inputs, thetas=None):
        return self.matrix(thetas) @ inputs

    def matrix(self, thetas=None):
        if thetas:
            return U3(*thetas)
        else:
            return U3(*self.thetas)


class ISWAPLayer(QCLayer):
    def __init__(self, targets, parameterized=False):
        super(ISWAPLayer, self).__init__(name='iSWAP_' + str(targets[0]) + '_' + str(targets[1]))
        self.targets = targets
        self._matrix = None
        self.parameterized = parameterized

    def build(self, input_shape):
        self.n_qubits = utils.intlog2(input_shape[-2])
        mi = tf.complex(0., -1.)
        if self.parameterized:
            theta_init = tf.random_uniform_initializer(0, 2 * π)
            self.t = tf.Variable(initial_value=theta_init((1,), dtype=float_type),
                                 trainable=True,
                                 dtype=float_type)
        else:
            i_swap = tf.convert_to_tensor([
                [1,  0,  0, 0],
                [0,  0, mi, 0],
                [0, mi,  0, 0],
                [0,  0,  0, 1]
            ], complex_type)
            self._matrix = gate_expand_2toN(i_swap, self.n_qubits, targets=self.targets)

    def call(self, inputs, **kwargs):
        return self.matrix() @ inputs

    def matrix(self, **kwargs) -> tf.Tensor:
        if self.parameterized:
            t = self.t[0]
            mi = tf.complex(0., -1.)
            i_swap = tf.convert_to_tensor([
                [1, 0, 0, 0],
                [0, tf.cos(t), mi * tf.cast(tf.sin(t), tf.complex64), 0],
                [0, mi * tf.cast(tf.sin(t), tf.complex64), tf.cos(t), 0],
                [0, 0, 0, 1]
            ], complex_type)
            return gate_expand_2toN(i_swap, self.n_qubits, targets=self.targets)
        else:
            return self._matrix


class SWAPLayer(QCLayer):
    def __init__(self, targets):
        super(SWAPLayer, self).__init__()
        self.targets = targets
        self._matrix = None

    def build(self, input_shape):
        n_qubits = utils.intlog2(input_shape[-2])
        i_swap = tf.convert_to_tensor([
            [1,  0,  0, 0],
            [0,  0, 1, 0],
            [0, 1,  0, 0],
            [0,  0,  0, 1]
        ], complex_type)
        self._matrix = gate_expand_2toN(i_swap, n_qubits, targets=self.targets)

    def call(self, inputs, **kwargs):
        return self._matrix @ inputs

    def matrix(self, **kwargs) -> tf.Tensor:
        return self._matrix


class ULayer(QCLayer):
    def __init__(self):
        super(ULayer, self).__init__()
        self.thetas: tf.Variable = None
        self.n_qubits = None

    def build(self, input_shape: tf.TensorShape):
        self.n_qubits = utils.intlog2(input_shape[-2])
        assert self.n_qubits % 4 == 0, \
            'Input tensor is not a multiple of 4, i.e. cannot apply diamond U-gate (TODO: specify what qubits to apply to)'
        theta_init = tf.random_uniform_initializer(0, 2 * π)
        self.thetas = tf.Variable(initial_value=theta_init(shape=(self.n_qubits//4,), dtype=float_type),
                                  trainable=True,
                                  dtype=float_type)

    def call(self, inputs, **kwargs):
        return self.matrix() @ inputs

    def matrix(self, **kwargs) -> tf.Tensor:
        Us = []
        for i in range(self.thetas.shape[0]):
            Us.append(U(self.thetas[i]))
        return tensor(Us)


class QFTLayer(QCLayer):
    def __init__(self):
        super(QFTLayer, self).__init__()
        self.n_qubits = None

    def build(self, input_shape):
        self.n_qubits = input_shape[-2].bit_length() - 1

    def call(self, inputs, **kwargs):
        return qft(self.n_qubits, inputs)

    def matrix(self):
        return qft(self.n_qubits, I4)


class QFTCrossSwapLayer(QCLayer):
    def __init__(self):
        super(QFTCrossSwapLayer, self).__init__()

    def build(self, input_shape):
        n_qubits = utils.intlog2(input_shape[-2])
        self._matrix = reduce(lambda a,b: a@b, [qc.SWAP[n_qubits][n][(n_qubits - 1) - n]
                                                for n in range(n_qubits//2)])

    def call(self, inputs, **kwargs):
        return self._matrix @ inputs

    def matrix(self, **kwargs) -> tf.Tensor:
        return self._matrix


class IQFTLayer(QCLayer):
    def __init__(self):
        super(IQFTLayer, self).__init__()
        self.n_qubits = None

    def build(self, input_shape):
        self.n_qubits = input_shape[-2].bit_length() - 1

    def call(self, inputs, **kwargs):
        return iqft(self.n_qubits, inputs)

    def matrix(self):
        return iqft(self.n_qubits, I4)


class ILayer(QCLayer):
    def __init__(self):
        super(ILayer, self).__init__()

    def build(self, input_shape):
        self._matrix = tf.eye(input_shape[-2], dtype=complex_type)

    def call(self, inputs, **kwargs):
        return self._matrix @ inputs

    def matrix(self, **kwargs) -> tf.Tensor:
        return self._matrix