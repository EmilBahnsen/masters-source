import tensorflow as tf
import math
from typing import *
import warnings

from tf_qc import float_type, complex_type
from tf_qc.utils import partial_trace, random_pure_states
from tf_qc.qc import intlog2, density_matrix, partial_trace_last, fidelity
from tf_qc.models import ApproxUsingInverse, QCModel, U3Layer, ILayer, OneMemoryDiamondQFT


class MeanNorm(tf.losses.Loss):
    def call(self, y_true, y_pred):
        # y_pred = tf.convert_to_tensor(y_pred)
        diff = y_true - y_pred
        norms = tf.cast(tf.norm(diff, axis=[-2, -1]), dtype=float_type)
        mean_norm = tf.reduce_mean(norms)
        return mean_norm


class MeanTraceDistance(tf.losses.Loss):
    """
    This is good because it's convex in both inputs
    https://en.wikipedia.org/wiki/Trace_distance
    or in Nielsen and Chuang
    """
    def __init__(self, subsystem: List[int] = None):
        super(MeanTraceDistance, self).__init__()
        self.subsystem = subsystem

    def call(self, y_true, y_pred):
        delta = density_matrix(y_true, self.subsystem) - density_matrix(y_pred, self.subsystem)
        result = tf.linalg.trace(tf.linalg.sqrtm(tf.linalg.adjoint(delta) @ delta))/2
        return tf.cast(tf.reduce_mean(result), float_type)


class Mean1mFidelity(tf.losses.Loss):
    def __init__(self, subsystem=None, true_is_pure_on_sub=False, pred_is_pure_on_sub=False):
        super(Mean1mFidelity, self).__init__()
        self.subsystem = subsystem
        self.true_is_pure_on_sub = true_is_pure_on_sub
        self.pred_is_pure_on_sub = pred_is_pure_on_sub

    def call(self, y_true, y_pred):
        fidelities = fidelity(y_true, y_pred, self.subsystem, self.true_is_pure_on_sub, self.pred_is_pure_on_sub)
        meanFilelity = tf.reduce_mean(fidelities)
        return 1 - meanFilelity


class Mean1mUhlmannFidelity(tf.losses.Loss):
    """
    1807.01640
    """
    def __init__(self, subsystem: List[int], n_qubits: int, optimized=False):
        """
        Mean of 1 - Uhlmann fidelity of states
        :param subsystem: The subsystem to measure the fidelity of
        """
        warnings.warn('Mean1mUhlmannFidelity is just the same as Mean1mFidelity now.', PendingDeprecationWarning)
        super(Mean1mUhlmannFidelity, self).__init__()
        self.subsystem = subsystem
        self.subsys_to_trace = [i for i in range(n_qubits) if i not in self.subsystem]
        self.n_qubits = n_qubits
        # If it's the last qubits we ignore (trace away), then use more efficient implemetation of trace
        self.last_trace = self.subsys_to_trace[-1] == n_qubits-1 and \
                          sum(self.subsys_to_trace) == sum(range(self.subsys_to_trace[0], n_qubits)) and \
                          optimized

    def call(self, y_true, y_pred):
        if self.last_trace:
            n2trace = len(self.subsys_to_trace)
            rho_true = partial_trace_last(y_true, n2trace, self.n_qubits)
            rho_pred = partial_trace_last(y_pred, n2trace, self.n_qubits)
        else:
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
    import time
    def time_oper(f):
        t1 = time.time()
        res = f()
        res = f()
        res = f()
        res = f()
        res = f()
        return res, f'{round(time.time() - t1, 10)}s'
    N = 12
    targets = [0, 1, 2, 3, 4, 5, 6, 7]
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

    data = random_pure_states((16, 2**N, 1), post_zeros=2, seed=0)
    m = Model()
    m(data)
    loss1 = Mean1mFidelity()
    loss2 = Mean1mUhlmannFidelity(targets, N, optimized=True)
    loss3 = Mean1mUhlmannFidelity(targets, N, optimized=False)
    output = m.matrix() @ data
    l1, t1 = time_oper(lambda: loss1(data, output))
    l2, t2 = time_oper(lambda: loss2(data, output))
    l3, t3 = time_oper(lambda: loss3(data, output))
    print('fidelity', t1, l1)
    print('Uhlmann0', t2, l2)
    print('Uhlmann1', t3, l3)
    assert l1 - l2 < 1e-6

    m2 = OneMemoryDiamondQFT()
    # print(m2(data))
