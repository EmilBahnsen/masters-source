from unittest import TestCase
from tf_qc.qc import append_zeros, tensor, s0, s1, density_matrix, fidelity, density_matrix_trace_from_state, trace, \
    partial_trace_last, inner_product, gate_expand_2toN, SWAP, I1
from tf_qc.utils import random_pure_states
import tensorflow as tf
from tf_qc import float_type
from txtutils import ndtotext_print
import numpy as np


def almost_equal(a: tf.Tensor, b: tf.Tensor, threshold=1e-5):
    return almost_zero(a - b, threshold)


def almost_zero(a: tf.Tensor, threshold=1e-5):
    return tf.reduce_all(tf.abs(a) < threshold)


class Test(TestCase):
    def test_append_zeros(self):
        def naive_impl(states, n):
            tensor([states] + [s0] * n)

        states = random_pure_states((10, 2 ** 12, 1))
        states = append_zeros(states, 2)
        print(density_matrix(states))


class TestFidelity(TestCase):
    def setUp(self) -> None:
        self.states1 = random_pure_states((10, 2 ** 4, 1), post_zeros=2)
        self.states2 = random_pure_states((10, 2 ** 4, 1), post_zeros=2)
        self.states3 = random_pure_states((10, 2 ** 4, 1))
        self.targets = list(range(2))

    @staticmethod
    def naive_alg(a, b):
        sqrtm = tf.linalg.sqrtm
        dm_a = density_matrix(a)
        dm_b = density_matrix(b)
        result = tf.linalg.trace(sqrtm(sqrtm(dm_a) @ dm_b @ sqrtm(dm_a))) ** 2
        return tf.cast(result, float_type)

    def test_basic_fidelity(self):
        states1 = random_pure_states((5, 2 ** 4, 1))
        states2 = random_pure_states((5, 2 ** 4, 1))
        fid1 = TestFidelity.naive_alg(states1, states2)
        fid2 = fidelity(states1, states2)
        # This is not precisely the same bc. of the numeric alg of the naive alg.
        self.assertTrue(almost_equal(fid1, fid2, 1e-3))

    def test_zero(self):
        fid = fidelity(self.states1, self.states1)
        self.assertTrue(almost_zero(1 - fid))

    def test_subsystem(self):
        fid1 = fidelity(self.states1, self.states2)
        fid2 = fidelity(self.states1, self.states2, self.targets)
        self.assertTrue(almost_equal(fid1, fid2, 1e-3), 'Does not extract subsystem.')

    def test_fidelity_subsystem_pure(self):
        fid0 = fidelity(self.states1, self.states3)
        fid1 = fidelity(self.states1, self.states3, self.targets, True)
        fid2 = fidelity(self.states3, self.states1, self.targets, False, True)
        self.assertTrue(almost_equal(fid0, fid1))
        self.assertTrue(almost_equal(fid1, fid2))


class TestDensityMatrixTraceFromState(TestCase):
    def setUp(self) -> None:
        with tf.device('cpu'):
            self.states1 = random_pure_states((10, 2 ** 4, 1))

    def test_density_matrix_trace_from_state(self):
        with tf.device('cpu'):
            # This method can take 500 states of 2**12 at once!
            dm1 = density_matrix(self.states1, [0, 1])  # Density matrix of forst 10 qubits
            dm1_0 = density_matrix(self.states1[0], [0, 1])
            # This con only take ~40 states of 2**12 at once...
            dm2 = trace(density_matrix(self.states1), [2, 3])
            dm2_0 = trace(density_matrix(self.states1[0]), [2, 3])
            # print('dm')
            # ndtotext_print(density_matrix(self.states1))
            # print('dm1')
            # ndtotext_print(dm1)
            # print('dm1_0')
            # ndtotext_print(dm1_0)
            # print('dm2')
            # ndtotext_print(dm2)
            # print('dm2_0')
            # ndtotext_print(dm2_0)
            self.assertTrue(almost_equal(dm1[0], dm1_0, 1e-3), 'density_matrix mixes states!')
            self.assertTrue(almost_equal(dm2[0], dm2_0, 1e-3), 'trace->density_matrix mixes states!')
            self.assertTrue(almost_equal(dm1, dm2, 1e-3))

    def test_non_last_qubits_trace(self):
        # Test trace for two conseg. non-last qubits
        state1 = tensor([s0, s1])
        state2 = tensor([(s0 + s1)/np.sqrt(2), (s0 - s1)/np.sqrt(2)])
        state = tensor([state1, state2])
        dm1 = trace(density_matrix(state), [0, 1])
        self.assertTrue(almost_equal(dm1, density_matrix(state2)))

        # Test trace for two separated non-last qubits
        state_test = tensor([s0, (s0 + s1)/np.sqrt(2)])
        dm2 = trace(density_matrix(state), [1, 3])
        self.assertTrue(almost_equal(dm2, density_matrix(state_test)))


class TestUtils(TestCase):
    def test_inner_product(self):
        a = tf.constant([[[1 + 1j]] * 4, [[1 + 2j]] * 4])
        b = tf.constant([[[2 + 1j]] * 4, [[3 + 2j]] * 4])
        self.assertTrue(tf.reduce_all(inner_product(a, b) == [12. - 4.j, 28. - 16.j]))


class TestGateExpand(TestCase):
    def test_gate_expand_2to_n(self):
        swap2 = SWAP(2, 0, 1)
        self.assertTrue(almost_equal(gate_expand_2toN(swap2, 3, targets=[0, 1]), tensor([swap2, I1])))
        self.assertTrue(almost_equal(gate_expand_2toN(swap2, 4, targets=[1, 2]), tensor([I1, swap2, I1])))
        self.assertTrue(almost_equal(gate_expand_2toN(swap2, 4, targets=[2, 3]), tensor([I1, I1, swap2])))
