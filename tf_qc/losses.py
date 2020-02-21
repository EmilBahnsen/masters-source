import tensorflow as tf
import math
from typing import *

from tf_qc import float_type, complex_type
from tf_qc.utils import partial_trace, random_pure_states
from tf_qc.qc import intlog2
from tf_qc.models import ApproxUsingInverse, QCModel, U3Layer, ILayer, OneMemoryDiamondQFT


class MeanNorm(tf.losses.Loss):
    def call(self, y_true, y_pred):
        # y_pred = tf.convert_to_tensor(y_pred)
        diff = y_true - y_pred
        norms = tf.cast(tf.norm(diff, axis=[-2, -1]), dtype=float_type)
        mean_norm = tf.reduce_mean(norms)
        return mean_norm


class Mean1mFidelity(tf.losses.Loss):
    def call(self, y_true, y_pred):
        outer_product = tf.matmul(y_true, y_pred, adjoint_a=True)
        # tf.transpose(y_true, perm=[0, 2, 1]) @ tf.math.conj(y_pred)
        fidelities = tf.abs(outer_product)**2
        meanFilelity = tf.reduce_mean(fidelities)
        return 1 - meanFilelity


class Mean1mUhlmannFidelity(tf.losses.Loss):
    """
    1807.01640
    """
    def __init__(self, subsystem: List[int], n_qubits: int):
        """
        Mean of 1 - Uhlmann fidelity of states
        :param subsystem: The subsystem to measure the fidelity of
        """
        super(Mean1mUhlmannFidelity, self).__init__()
        self.subsystem = subsystem
        self.subsys_to_trace = [i for i in range(n_qubits) if i not in self.subsystem]
        self.n_qubits = n_qubits

    def call(self, y_true, y_pred):
        rho_true = partial_trace(y_true, self.subsys_to_trace, self.n_qubits)
        rho_pred = partial_trace(y_pred, self.subsys_to_trace, self.n_qubits)
        sqrtm = tf.linalg.sqrtm
        # Square to be compatible with defn. of Mean1mFidelity
        fids = tf.linalg.trace(sqrtm(sqrtm(rho_true) @ rho_pred @ sqrtm(rho_true)))**2
        return 1 - tf.cast(tf.reduce_mean(fids), float_type)


class StdFidelity(tf.losses.Loss):
    def call(self, y_true, y_pred):
        norm_squares = tf.transpose(y_true, perm=[0, 2, 1]) @ tf.math.conj(y_pred)
        fidelities = tf.square(tf.abs(norm_squares))
        stdFilelity = tf.keras.backend.std(fidelities)
        return stdFilelity

# TEST
x = tf.constant([1/math.sqrt(3), 1/math.sqrt(3), 1/math.sqrt(3), 1/math.sqrt(2), 1/math.sqrt(2), 0], shape=(2,3,1), dtype=complex_type)
y = tf.constant([1, 0, 0, 1, 0, 0], shape=(2,3,1), dtype=complex_type)
assert round(Mean1mFidelity()(x, y).numpy(), 5) - round(1 - (1/3 + 1/2)/2, 5) < 1e-4
# TEST END

# TEST
# x = tf.constant([1,2,3,4], shape=(2,2,1), dtype=complex_type)
# assert round(MeanNorm()(x, 2*x).numpy(), 5) == 3.61803
# TEST END

if __name__ == '__main__':
    N = 6
    targets = [0, 1, 2, 3]
    class Model(ApproxUsingInverse):
        def __init__(self):
            model = QCModel(layers=[
                tf.keras.Input((2**N, 1), dtype=complex_type, name='input_state'),
                U3Layer(targets)
            ])
            target = QCModel(layers=[
                ILayer()
            ])
            super(Model, self).__init__(model, target, 'test')

    data = random_pure_states((10, 2**N, 1), post_zeros=2, seed=0)
    m = Model()
    m(data)
    loss1 = Mean1mFidelity()
    loss2 = Mean1mUhlmannFidelity(targets, N)
    output = m.matrix() @ data
    print('fideliy', loss1(data, output))
    print('Uhlmann', loss2(data, output))

    m2 = OneMemoryDiamondQFT()
    print(m2(data))

