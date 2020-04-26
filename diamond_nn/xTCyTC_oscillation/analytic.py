from sympy_diamond import *
from diamond_nn.x2y2 import xCTyCT
from sympy import pprint, expand_trig
import matplotlib.pyplot as plt
import numpy as np
import sympy.utilities.codegen
import pickle
import cloudpickle

# Here we are looking at the diamond in the config.
#
# xa (C1) – ya (T2)
# |          |
# xb (T1) – yb (C2)
#
# So that's 2 of the qubits as an input and 2 as output
# We aim to make the state oscillate btw. the input and output, setting the old input with
# new weights, and the back and fourth.

π = sp.pi

t0, t1, t2 = sp.symbols('t:3', real=True, nonnegative=True)
Ut0 = U(t0)
Ut1 = U(t1)
Ut2 = U(t2)

# Make the initial state of the diamond
diamond = xCTyCT()
state_x, xs = state(2, 'x')
state_w, ws = state(2, 'w')
state_v, vs = state(2, 'v')
state_u, us = state(2, 'u')

def replace_symbols(expr, value, *args):
    if len(args) == 0:
        return expr
    return reduce(lambda acc, arg: acc.subs(arg, value), args[0], replace_symbols(expr, value, *args[1:]))


# Entangling gate
def gamma_gate(a0, a1, b0, b1, t, c1, c2, c3, d1, d2, d3):
    o1 = sp.kronecker_product(U3(a0, a1, 0), U3(b0, b1, 0))
    o2 = iSWAP(t)
    o3 = sp.kronecker_product(U3(c1, c2, c3), U3(d1, d2, d3))
    return o3 @ o2 @ o1


print('--- Algorithm execution ---')
diamond.set_C1T1_state(state_x)  # REPLACE W apply_operator_C1T1 gamma_gate
diamond.set_C2T2_state(state_w)  # REPLACE W apply_operator_C2T2 gamma_gate
diamond.apply_operator(Ut0)
# diamond.set_C1T1_state(state_v)  # REPLACE W apply_operator_C1T1 gamma_gate
# diamond.apply_operator(Ut1)
# diamond.set_C2T2_state(state_u)
# diamond.apply_operator(Ut2)
dm = diamond.dm_C2T2  # This is where the output is read from

print('--- Output probabilities ---')
def simplify_dm(expr: sp.Expr):
    # expr = expr.expand(complex=True)
    # a, b, c = map(sp.Wild, 'abc')
    # expr = expr.replace(sp.cos(a) ** 2, 1 - sp.sin(a) ** 2)
    # expr = expr.expand()
    # expr = expr.collect((sp.cos(t0), sp.cos(t1)))
    # expr = expr.collect((sp.sin(t0), sp.sin(t1)))
    return expr.expand(complex=True)

P00 = simplify_dm(dm[0, 0])  # LOCK
P01 = simplify_dm(dm[1, 1])
P10 = simplify_dm(dm[2, 2])
P11 = simplify_dm(dm[3, 3])

print('P00')
pprint(P00)

print('--- lambda creation ---')
input_params = [*xs, *ws, *vs, t0, t1]

# pprint(replace_symbols(P00_real_input, 0, *input_params_real_input))

norm_factor = diamond.normalization_factor
P00_numpy = sp.lambdify(input_params, norm_factor**2 * P00, modules='numpy')
P01_numpy = sp.lambdify(input_params, norm_factor**2 * P01, modules='numpy')
P10_numpy = sp.lambdify(input_params, norm_factor**2 * P10, modules='numpy')
P11_numpy = sp.lambdify(input_params, norm_factor**2 * P11, modules='numpy')

P00_tensorflow = sp.lambdify(input_params, norm_factor**2 * P00, modules='tensorflow')
P01_tensorflow = sp.lambdify(input_params, norm_factor**2 * P01, modules='tensorflow')
P10_tensorflow = sp.lambdify(input_params, norm_factor**2 * P10, modules='tensorflow')
P11_tensorflow = sp.lambdify(input_params, norm_factor**2 * P11, modules='tensorflow')

with open(r"analytic_P.pickle", "wb") as output_file:
    cloudpickle.dump((P00_numpy, P01_numpy, P10_numpy, P11_numpy), output_file)

with open(r"analytic_P_tensorflow.pickle", "wb") as output_file:
    cloudpickle.dump((P00_tensorflow, P01_tensorflow, P10_tensorflow, P11_tensorflow), output_file)

