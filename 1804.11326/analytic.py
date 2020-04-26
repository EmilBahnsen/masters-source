import sympy as sp
from sympy.physics.quantum.gate import H
from sympy import pprint
from sympy_diamond import *
from scipy.stats import unitary_group
import  scipy.linalg.special_matrices as sm

sm.

def rand_SU(n):
    unitary_group.rvs(n)

I = sp.I
H = H().get_target_matrix()
HH = sp.kronecker_product(H, H)
a = sp.symbols('a:10')
U3U3 = sp.kronecker_product(U3(a[0], a[1], a[2]), U3(a[3], a[4], a[5]))
pprint(U3U3 @ HH @ s00)
