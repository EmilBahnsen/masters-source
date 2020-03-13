import tensorflow as tf
from functools import reduce
from typing import *
from abc import ABCMeta, abstractmethod

import tf_qc.qc
from tf_qc import float_type, utils, complex_type
import tf_qc.qc as qc
from tf_qc.qc import H, U, qft_U, I4, π, U3, iqft, qft, gate_expand_1toN, gate_expand_2toN, gate_expand_toN, gates_expand_toN, tensor

_uniform_theta = tf.random_uniform_initializer(0, 2*π)
_normal_theta = tf.random_normal_initializer(0, π/4)


class QCLayer(tf.keras.layers.Layer, metaclass=ABCMeta):
    def __init__(self, *args, init_nearly_eye=False, **kwargs):
        super(QCLayer, self).__init__(*args, **kwargs)
        self.init_nearly_eye = init_nearly_eye

    @abstractmethod
    def matrix(self, **kwargs) -> tf.Tensor:
        pass

    def build(self, input_shape):
        if input_shape == None:
            raise TypeError('input_shape is None')
        self.n_qubits = tf_qc.qc.intlog2(input_shape[-2])


class QFTULayer(QCLayer):
    def __init__(self, n_qubits=4):
        super(QFTULayer, self).__init__()
        self.n_qubits = None
        self.thetas = None

    def build(self, input_shape):
        super().build(input_shape)
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
        super().build(input_shape)
        self._matrix = gate_expand_1toN(H, self.n_qubits, self.target)

    def call(self, inputs, **kwargs):
        return self._matrix @ inputs

    def matrix(self, **kwargs) -> tf.Tensor:
        return self._matrix


class U3Layer(QCLayer):
    def __init__(self, targets: Optional[List[Union[List[int], int]]] = None):
        super(U3Layer, self).__init__()
        self.U3 = None
        self.thetas = None
        # Make sure that we have Optional[List[List[int]]] no matter what
        self.targets = targets
        if self.targets is not None:
            self.targets = list(map(lambda t: [t] if isinstance(t, int) else t, self.targets))

    def build(self, input_shape: tf.TensorShape):
        super().build(input_shape)
        n_thetas = len(self.targets) if self.targets is not None else self.n_qubits
        self.thetas = [tf.Variable(initial_value=_uniform_theta(shape=(3,), dtype=float_type),
                                   trainable=True,
                                   dtype=float_type,
                                   name=f'var_{j}') for j in range(n_thetas)]

    def call(self, inputs, **kwargs):
        m = self.matrix()
        return m @ inputs

    def matrix(self):
        if self.targets is not None:
            U3s = [U3(t) for t in self.thetas]
            return gates_expand_toN(U3s, self.n_qubits, self.targets)
        else:
            return U3(*self.thetas)


class ISWAPLayer(QCLayer):
    number = 0

    def __init__(self, targets, parameterized=False):
        super(ISWAPLayer, self).__init__(name=f'iSWAP_{targets[0]}_{targets[1]}_{ISWAPLayer.number}')
        ISWAPLayer.number += 1
        self.targets = targets
        self._matrix = None
        self.parameterized = parameterized

    def build(self, input_shape):
        super().build(input_shape)
        mi = tf.complex(0., -1.)
        if self.parameterized:
            self.t = tf.Variable(initial_value=_uniform_theta((1,), dtype=float_type),
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
        super().build(input_shape)
        self._matrix = qc.SWAP(self.n_qubits, self.targets[0], self.targets[1])

    def call(self, inputs, **kwargs):
        return self._matrix @ inputs

    def matrix(self, **kwargs) -> tf.Tensor:
        return self._matrix


class ULayer(QCLayer):
    def __init__(self, targets: Optional[List[int]] = None):
        super(ULayer, self).__init__()
        self.thetas: tf.Variable
        self.n_qubits: int
        if targets:
            if targets != sorted(targets):
                raise Exception('ULayer targets must be sorted')
        self.targets = targets

    def build(self, input_shape: tf.TensorShape):
        super().build(input_shape)
        n_thetas = len(self.targets)//4 if self.targets else self.n_qubits//4
        self.thetas = tf.Variable(initial_value=_uniform_theta(shape=(n_thetas,), dtype=float_type),
                                  trainable=True,
                                  dtype=float_type)
        if self.targets:
            pre_identity_qubits = self.targets[0]
            post_identity_qubits = self.n_qubits-1 - self.targets[-1]
            self.pre_identity = tf.eye(2**pre_identity_qubits, dtype=complex_type)
            self.post_identity = tf.eye(2**post_identity_qubits, dtype=complex_type)

    def call(self, inputs, **kwargs):
        return self.matrix() @ inputs

    def matrix(self, **kwargs) -> tf.Tensor:
        Us = []
        for i in range(self.thetas.shape[0]):
            Us.append(qc.U(self.thetas[i]))
        if self.targets:
            return tensor([self.pre_identity, *Us, self.post_identity])
        else:
            return tensor(Us)


class QFTLayer(QCLayer):
    def __init__(self):
        super(QFTLayer, self).__init__()
        self.n_qubits = None

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return qft(self.n_qubits, inputs)

    def matrix(self):
        return qft(self.n_qubits, tf.eye(2**self.n_qubits, dtype=complex_type))


class QFTCrossSwapLayer(QCLayer):
    def __init__(self, targets=None):
        super(QFTCrossSwapLayer, self).__init__()
        self.targets = targets

    def build(self, input_shape):
        super().build(input_shape)
        n_targets = len(self.targets) if self.targets is not None else self.n_qubits
        swap_matrices = [qc.SWAP(n_targets, n, (n_targets - 1) - n) for n in range(n_targets//2)]
        self._matrix = reduce(lambda a, b: a@b, swap_matrices)
        self._matrix = gate_expand_toN(tf.cast(self._matrix, complex_type), self.n_qubits, self.targets)

    def call(self, inputs, **kwargs):
        return self._matrix @ inputs

    def matrix(self, **kwargs) -> tf.Tensor:
        return self._matrix


qft_cross_iswap_layers = lambda targets: [ISWAPLayer([targets[n], targets[(len(targets)-1) - n]]) for n in range(len(targets)//2)]
qft_cross_swap_layers = lambda targets: [SWAPLayer([targets[n], targets[(len(targets)-1) - n]]) for n in range(len(targets)//2)]


class IQFTLayer(QCLayer):
    def __init__(self, targets: List[int] = None):
        super(IQFTLayer, self).__init__()
        self.targets = targets
        self._matrix = None

    def build(self, input_shape):
        super().build(input_shape)
        # if no target specified, then just act on all qubits
        if not self.targets:
            self.targets = list(range(self.n_qubits))
        n_qft = len(self.targets)
        I = tf.eye(2**n_qft, dtype=complex_type)
        self._matrix = gate_expand_toN(iqft(n_qft, I), self.n_qubits, self.targets)

    def call(self, inputs, **kwargs):
        return self.matrix() @ inputs

    def matrix(self):
        return self._matrix


class CPLayer(QCLayer):
    def __init__(self, control: int, target: int, phase: float):
        super(CPLayer, self).__init__()
        self.control = control
        self.target = target
        self.phase = phase

    def build(self, input_shape):
        super().build(input_shape)
        self._matrix = gate_expand_2toN(tf.convert_to_tensor([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, tf.exp(-1j * self.phase)]], dtype=complex_type),
                                        self.n_qubits,
                                        self.control,
                                        self.target)

    def call(self, inputs, **kwargs):
        return self.matrix() @ inputs

    def matrix(self):
        return self._matrix


class ILayer(QCLayer):
    def __init__(self):
        super(ILayer, self).__init__()
        self._matrix = None

    def build(self, input_shape):
        self._matrix = tf.eye(input_shape[-2], dtype=complex_type)

    def call(self, inputs, **kwargs):
        return inputs

    def matrix(self, **kwargs) -> tf.Tensor:
        return self._matrix


if __name__ == '__main__':
    from tf_qc.utils import random_pure_states
    from txtutils import ndtotext_print
    data = random_pure_states((2, 2**2, 1))
    l_u3 = U3Layer()
    l_u3(data)
    l_u3.variables[0].assign([0])
    l_u3.variables[1].assign([0])
    l_u3.variables[2].assign([0])
    l_u3.variables[3].assign([0])
    l_u3.variables[4].assign([π/2])
    l_u3.variables[5].assign([π])
    print(*l_u3.variables, sep='\n')
    ndtotext_print(l_u3.matrix)

    data_data = lambda N: tf.keras.layers.Input((2**N, 1), batch_size=2, dtype=complex_type)
    data = lambda N: random_pure_states((2, 2**N, 1))
    l1 = ULayer()
    l2 = ULayer([[0,1,2,3], [4,5,6,7]])
    l1(data(8))
    l2(data(8))
    l1.thetas = l2.thetas
    assert tf.reduce_all(l1.matrix() == l2.matrix())

    l1 = ULayer()
    l_I1 = ILayer()
    l2 = ULayer([[0, 1, 2, 3], [4, 5, 6, 7]])
    l1(data(8))
    l_I1(data(1))
    l2(data(9))
    l1.thetas = l2.thetas
    assert tf.reduce_all(tensor([l1.matrix(), l_I1.matrix()]) == l2.matrix())

    l1 = ULayer()
    l_I1 = ILayer()
    l2 = ULayer([[1, 2, 3, 4], [5, 6, 7, 8]])
    l1(data(8))
    l_I1(data(1))
    l2(data(10))
    l1.thetas = l2.thetas
    assert tf.reduce_all(tensor([l_I1.matrix(), l1.matrix(), l_I1.matrix()]) == l2.matrix())

    l1 = ULayer()
    l_I2 = ILayer()
    l2 = ULayer([0, 1, 2, 3])
    l1(data(4))
    l_I2(data(2))
    l2(data(6))
    l1.thetas = l2.thetas
    assert tf.reduce_all(tensor([l1.matrix(), l_I2.matrix()]) == l2.matrix())

    l1 = IQFTLayer([1,2,3,4])
    l1(data(5))
    ndtotext_print(l1.matrix())
