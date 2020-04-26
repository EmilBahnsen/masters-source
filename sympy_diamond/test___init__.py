from unittest import TestCase
from sympy_diamond import partial_trace, density_matrix
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy_diamond import s0, s1
from sympy import pprint, tensorcontraction


class Test(TestCase):
    def test_partial_trace(self):
        state1 = TensorProduct(s0, s1, s0, s1)
        state2 = TensorProduct(s0, s0)
        dm1 = partial_trace(density_matrix(state1), 4, [1, 3])
        dm2 = density_matrix(state2)
        self.assertEqual(dm1, dm2)

    def test_partial_trace2(self):
        import sympy as sp
        dm = sp.randMatrix(2**4, 2**4)
        dm1 = partial_trace(dm, 4, [0, 2, 3])
        dm2 = partial_trace(dm, 4, [0, 1, 2])
        dm3 = partial_trace(dm, 4, [0, 2])

        T1_0 = dm1[0, 0]
        T3_0 = dm2[0, 0]
        T13_00 = dm3[0, 0]
        T13_01 = dm3[1, 1]
        T13_10 = dm3[2, 2]
        T13_11 = dm3[3, 3]

        self.assertEqual(T1_0, T13_00 + T13_01)
        self.assertEqual(T3_0, T13_00 + T13_10)
