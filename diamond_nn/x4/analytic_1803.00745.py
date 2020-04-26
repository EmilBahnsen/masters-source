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
# xa(y) (C1) – xd (T2)
# |          |
# xc (T1) – xb (C2)
#
# So that's all qubits as input, and also qubit 0 as output
# As per the article 1803.00745

s0000 = sp.kronecker_product(s00, s00)

π = sp.pi
a, b, c = map(sp.Wild, 'abc')

# This is the input parameter
x = sp.Symbol('x', real=True)

# Free parameters
t0, t1, t2 = sp.symbols('t:3', real=True, nonnegative=True)
Ut0 = U(t0)
Ut1 = U(t1)
Ut2 = U(t2)
w = sp.symbols('w:12', real=True)


def replace_symbols(expr, value, *args):
    if len(args) == 0:
        return expr
    return reduce(lambda acc, arg: acc.subs(arg, value), args[0], replace_symbols(expr, value, *args[1:]))


# --- Simplification helpers ---
def clear_conjugates_of_x(expr):
    a, b, c = map(sp.Wild, 'abc')
    expr = expr.replace(sp.conjugate(sp.asin(x ** a)), sp.asin(x ** a))  # We can do this bc. x is in [-1, 1]
    expr = expr.replace(sp.conjugate(sp.acos(x ** a)), sp.acos(x ** a))
    expr = expr.replace(sp.conjugate(sp.sqrt(a*x**b +c)), sp.sqrt(a*x**b +c))
    return expr

def simplify_trig_arctrig(expr):
    expr = expr.replace(sp.sin(sp.asin(x) / 2), (sp.sqrt(1 + x) - sp.sqrt(1 - x)) / 2)  # https://www.wolframalpha.com/input/?i=sin%28asin%28x%29%2F2%29
    expr = expr.replace(sp.cos(sp.asin(x) / 2), (sp.sqrt(1 + x) + sp.sqrt(1 - x)) / 2)  # https://www.wolframalpha.com/input/?i=cos%28asin%28x%29%2F2%29
    expr = expr.replace(sp.cos(sp.acos(x**2) / 2), sp.sqrt((1+x**2)/2))  # https://www.wolframalpha.com/input/?i=simplify+cos%28acos%28x%29%2F2%29
    expr = expr.replace(sp.sin(sp.acos(x**2) / 2), sp.sqrt((1-x**2)/2))  # https://www.wolframalpha.com/input/?i=simplify+sin%28acos%28x%29%2F2%29
    return expr

def simplify_exp_acos(expr):
    expr = expr.replace(sp.exp( 1j*sp.acos(x ** 2)), x**2 + 1j*sp.sqrt(1 - x**4))  # https://www.wolframalpha.com/input/?i=simplify+exp%28i*acos%28x**2%29%29, https://www.wolframalpha.com/input/?i=simplify+x%5E2+%2B+i+sqrt%281+-+x%29+sqrt%281+%2B+x%29+sqrt%281+%2B+x%5E2%29
    expr = expr.replace(sp.exp(-1j*sp.acos(x ** 2)), x**2 - 1j*sp.sqrt(1 - x**4))  # https://www.wolframalpha.com/input/?i=simplify+exp%28-i*acos%28x**2%29%29
    expr = expr.replace(sp.exp( 2j*sp.acos(x ** 2)), 2*x**4 + 2j*sp.sqrt(1 - x**4)*x**2 - 1)  # https://www.wolframalpha.com/input/?i=simplify+exp%282i*acos%28x**2%29%29, https://www.wolframalpha.com/input/?i=simplify+-1+%2B+2+x%5E4+%2B+2+i+sqrt%281+-+x%29+x%5E2+sqrt%281+%2B+x%29+sqrt%281+%2B+x%5E2%29
    expr = expr.replace(sp.exp(-2j*sp.acos(x ** 2)), 2*x**4 - 2j*sp.sqrt(1 - x**4)*x**2 - 1)  # https://www.wolframalpha.com/input/?i=simplify+exp%28-2i*acos%28x**2%29%29
    return expr

def expand_complex_exp(expr):
    a, b, c = map(sp.Wild, 'abc')
    return expr.replace(sp.exp(1j*a), sp.cos(a) + 1j*sp.sin(a))

def measure_Z_exp_0(state):
    dm = density_matrix(state)
    P_q1 = partial_trace_last_n_qubits(dm, 4, 3)  # Trace away the last qubits to measure on the first
    Z_exp = 2 * P_q1[0, 0] - 1  # Calc. the expectation value of the first qubit
    return Z_exp

# --- For intermediate results ---
def print_Z_exp_encoding(state):
    Z_exp = measure_Z_exp_0(state)
    Z_exp = clear_conjugates_of_x(Z_exp)  # x is not complex nor any of the roots of this
    Z_exp = Z_exp.simplify()
    print('Z_exp after encoding')
    pprint(Z_exp)

def print_Z_exp_enc_U0(state):
    Z_exp = measure_Z_exp_0(state)
    Z_exp = clear_conjugates_of_x(Z_exp)  # x is not complex nor any of the roots of this
    print('Z_exp after encoding->U0 (NOTICE: U after encoding has no effect, '
          'and that\'s the same no matter what qubit we measure), '
          'AND the same thing goes for parametzired iSWAP, BUT'
          'it does contribute with off-diagonal elements')
    pprint(Z_exp.simplify())

def print_Z_exp_enc_U0_XZX(state):
    Z_exp = measure_Z_exp_0(state)
    Z_exp = clear_conjugates_of_x(Z_exp)
    # Z_exp = Z_exp.simplify()
    print('Z_exp after encoding->U0->RXZX')
    pprint(Z_exp)


print('--- Algorithm execution ---')
# Encoding
encoding_oper = RZ(sp.acos(x**2)) @ RY(sp.asin(x))
print('Encoding matrix on 1 qubit before simplification')
pprint(encoding_oper)

encoding_oper = expand_complex_exp(encoding_oper)
encoding_oper = simplify_trig_arctrig(encoding_oper)
encoding_oper = encoding_oper.simplify()
print('Encoding matrix on 1 qubit after simplification')
pprint(encoding_oper)
encoding_opers = sp.kronecker_product(*(4*[encoding_oper]))
state = encoding_opers @ s0000  # The same on all 4 qubits
state = state.simplify()
print_Z_exp_encoding(state)  # Intermediate result
state = Ut0 @ state  # THIS DOES NOTDING?! (mathematically)
print_Z_exp_enc_U0(state)
exit()
rot_opers = sp.kronecker_product(
    RXZX(w[0], w[1], w[2]),
    RXZX(w[3], w[4], w[5]),
    RXZX(w[6], w[7], w[8]),
    RXZX(w[9], w[10], w[11])
)
state = rot_opers @ state
print_Z_exp_enc_U0_XZX(state)
exit()

print('--- Output probabilities ---')

Z_exp = measure_Z_exp_0(state)

print('Z_exp')
pprint(Z_exp)
exit()

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

