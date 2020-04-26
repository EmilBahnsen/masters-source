from unittest import TestCase
from tf_qc.layers import SWAPLayer
from tf_qc.utils import random_pure_states
from txtutils import ndtotext_print
from qutip.qip.operations import swap as qt_swap
import tensorflow as tf


def almost_equal(a: tf.Tensor, b: tf.Tensor, threshold=1e-5):
    return almost_zero(a - b, threshold)


def almost_zero(a: tf.Tensor, threshold=1e-5):
    return tf.reduce_all(tf.abs(a) < threshold)


class TestSWAPLayer(TestCase):
    def test_swap(self):
        N = 10
        for i in range(5):
            for j in range(5):
                if i == j: continue
                targets = [i, j]
                data = random_pure_states((2, 2**N, 1))
                l = SWAPLayer(targets)
                l(data)
                self.assertTrue(almost_equal(l.matrix(), qt_swap(N, targets).full()), f'N = {N}, [{i},{j}]')

    def test_N5_0_3(self):
        N = 5
        targets = [0, 3]
        data = random_pure_states((2, 2 ** N, 1))
        l = SWAPLayer(targets)
        l(data)
        m1 = l.matrix()
        m2 = qt_swap(N, targets).full()
        self.assertTrue(almost_equal(m1, m2))
