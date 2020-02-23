from unittest import TestCase
from tf_qc.qc import append_zeros, tensor, s0, density_matrix, fidelity, density_matrix_trace_from_state, trace, partial_trace_last
from tf_qc.utils import random_pure_states
import tensorflow as tf


def almost_equal(a: tf.Tensor, b: tf.Tensor):
    return almost_zero(a - b)


def almost_zero(a: tf.Tensor):
    return tf.reduce_all(tf.abs(a) < 1e-5)


class Test(TestCase):
    def test_append_zeros(self):
        def naive_impl(states, n):
            tensor([states] + [s0] * n)

        states = random_pure_states((10, 2 ** 12, 1))
        states = append_zeros(states, 2)
        print(density_matrix(states))


class TestFidelity(TestCase):
    def setUp(self) -> None:
        self.states1 = random_pure_states((10, 2 ** 12, 1), post_zeros=2)
        self.states2 = random_pure_states((10, 2 ** 12, 1), post_zeros=2)

    def test_fidelity(self):
        targets = list(range(10))
        fid1 = fidelity(self.states1, self.states1)
        self.assertTrue(almost_zero(1 - fid1))
        fid2a = fidelity(self.states1, self.states2, targets)
        fid2b = fidelity(self.states1, self.states2, targets, False, True)
        fid2c = fidelity(self.states1, self.states2, targets, True, False)
        fid2d = fidelity(self.states1, self.states2, targets, True, True)
        self.assertTrue(almost_equal(fid2a, fid2b))
        self.assertTrue(almost_equal(fid2a, fid2c))
        self.assertTrue(almost_equal(fid2a, fid2d))


class TestDensityMatrixTraceFromState(TestCase):
    def setUp(self) -> None:
        with tf.device('cpu'):
            self.states1 = random_pure_states((32, 2 ** 12, 1))

    def test_density_matrix_trace_from_state(self):
        with tf.device('cpu'):
            # This method can take 500 states at once!
            dm1 = density_matrix(self.states1, list(range(10)))  # Density matrix of forst 10 qubits
            # This con only take ~40 states at once...
            dm2 = partial_trace_last(density_matrix(self.states1), 2, 12)
            self.assertTrue(tf.reduce_all(tf.abs(dm1 - dm2) < 1e-3))
