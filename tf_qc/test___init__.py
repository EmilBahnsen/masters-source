from unittest import TestCase
import math
import tensorflow as tf

from tf_qc import Mean1mFidelity

complex_type = tf.complex128

class TestMean1mFidelity(TestCase):
    def test_call(self):
        x = tf.constant([1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
                        shape=(2, 3, 1), dtype=complex_type)
        y = tf.constant([1, 0, 0, 1, 0, 0], shape=(2, 3, 1), dtype=complex_type)
        # assert round(self(x, y).numpy(), 5) == round(1 - (1 / 3 + 1 / 2) / 2, 5)
        self.assertAlmostEqual(Mean1mFidelity(x, y).numpy(), 1 - (1 / 3 + 1 / 2) / 2)
