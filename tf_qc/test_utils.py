from unittest import TestCase
from tf_qc.utils import random_pure_states


class TestStateGeneration(TestCase):
    def test_random_pure_states(self):
        states = random_pure_states((10, 2 ** 12, 1), post_zeros=2)
        print(states)


class Test(TestCase):
    def test_normalize_state_vectors(self):
        self.fail()
