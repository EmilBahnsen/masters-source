from sympy_diamond import *
import sympy as sp
from typing import *
from qutip.qip.operations import swap

class xCTyCT:
    def __init__(self):
        # We start out with all zeros |0000>
        self._dm = sp.ImmutableDenseMatrix([
            [1] + [0] * (2**4 - 1),
            *( [[0]*(2**4)]*(2**4 - 1) )
        ])
        self._swap_matrix = swap(4, [1, 2])

    def set_C1T1_state(self, state: sp.Matrix):
        self.dm_C1T1 = density_matrix(state)

    def set_C2T2_state(self, state: sp.Matrix):
        self.dm_C2T2 = density_matrix(state)

    def density_matrix(self, subspace:List[int] = None):
        if subspace is None:
            return self._dm.subs(1.0, 1)
        else:
            index2trace = list(set(range(4)).difference(set(subspace)))
            return partial_trace(self.density_matrix(), 4, index2trace)

    @property
    def dm_C1T1(self):
        return self.density_matrix([0, 2])

    @dm_C1T1.setter
    def dm_C1T1(self, dm_C1T1):
        self._dm = sp.kronecker_product(dm_C1T1, self.dm_C2T2)
        self.swapC2T1()

    @property
    def dm_C2T2(self):
        return self.density_matrix([1, 3])

    @dm_C2T2.setter
    def dm_C2T2(self, dm_C2T2):
        self._dm = sp.kronecker_product(self.dm_C1T1, dm_C2T2)
        self.swapC2T1()

    @property
    def normalization_factor(self):
        return 1/sp.sqrt(sp.trace(self.density_matrix()).simplify())

    def apply_operator(self, operator: sp.Matrix):
        self._dm = operator @ self.density_matrix() @ operator.adjoint()

    def apply_operator_C1T1(self, operator: sp.Matrix):
        self.swapC2T1()
        I2 = sp.eye(4)
        operator = sp.kronecker_product(operator, I2)
        self._dm = operator @ self.density_matrix() @ operator.adjoint()
        self.swapC2T1()

    def apply_operator_C2T2(self, operator: sp.Matrix):
        self.swapC2T1()
        I2 = sp.eye(4)
        operator = sp.kronecker_product(I2, operator)
        self._dm = operator @ self.density_matrix() @ operator.adjoint()
        self.swapC2T1()

    def swapC2T1(self):
        # Swaps C1 C2 T1 T2 -> C1 T1 C2 T2
        self._dm = self._swap_matrix @ self.density_matrix() @ self._swap_matrix
