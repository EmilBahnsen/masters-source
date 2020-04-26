from sympy_diamond import *
from diamond_nn.x2y2 import xCTyCT
from sympy import pprint, expand_trig
import matplotlib.pyplot as plt
import numpy as np

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
state_x, xs, phis = cp_state(2, ['x', 'phi'])
state_w, ws, omegas = cp_state(2, ['w', 'omega'])
state_v, vs, rhos = cp_state(2, ['v', 'rho'])
state_u, us, mus = cp_state(2, ['u', 'mu'])

def replace_symbols(expr, value, *args):
    if len(args) == 0:
        return expr
    return reduce(lambda acc, arg: acc.subs(arg, value), args[0], replace_symbols(expr, value, *args[1:]))

print('--- Algorithm execution ---')
diamond.set_C1T1_state(state_x)
diamond.set_C2T2_state(state_w)
diamond.apply_operator(Ut0)
diamond.set_C1T1_state(state_v)
diamond.apply_operator(Ut1)
# diamond.set_C2T2_state(state_u)
# diamond.apply_operator(Ut2)

print('--- Output probabilities ---')
dm = diamond.dm_C1T1
P00 = dm[0, 0]  # LOCK
P00_real_input = replace_symbols(P00, 0, phis, omegas, rhos, mus)  # LOCK
P01 = dm[1, 1]
P01_real_input = replace_symbols(P01, 0, phis, omegas, rhos, mus)  # LOCK
P10 = dm[2, 2]
P10_real_input = replace_symbols(P10, 0, phis, omegas, rhos, mus)  # LOCK
P11 = dm[3, 3]
P11_real_input = replace_symbols(P11, 0, phis, omegas, rhos, mus)  # LOCK

print('P00')
pprint(P00)
print('P00 real')
pprint(P00_real_input)
print('P00 real input only')
# P00_input_only = replace_symbols(P00_real_input.subs(t0, π/4).subs(t0, π/4), 1, ws, vs).expand(complex=True).simplify()
# P00_input_only = P00_input_only.collect(xs)
def simplify_wo_angles(expr):
    expr = expr.expand(complex=True)
    a, b, c = map(sp.Wild, 'abc')
    expr = expr.replace(sp.cos(a)**2, 1 - sp.sin(a)**2)
    expr = expr.expand()
    expr = expr.collect((sp.cos(t0), sp.cos(t1))).collect((sp.cos(t0), sp.cos(t1)))
    expr = expr.collect((sp.sin(t0), sp.sin(t1))).collect((sp.sin(t0), sp.sin(t1)))
    # expr = expr.subs(norm_square_sum, 1)
    # expr = expr.collect((sp.sin(t0)**2 + sp.sin(t1)**2)/16)
    # pprint(expr)
    return expr
P00_real_input = simplify_wo_angles(P00_real_input)
P01_real_input = simplify_wo_angles(P01_real_input)
P10_real_input = simplify_wo_angles(P10_real_input)
P11_real_input = simplify_wo_angles(P11_real_input)
pprint(P00_real_input)
# exit()
# print('P01')
# pprint(P01)
# print('P10')
# pprint(P10)
# print('P11')
# pprint(P11)


print('--- lambda creation ---')
input_params = [*xs, *phis, *ws, *omegas, *vs, *rhos, *us, *mus, t0, t1, t2]
input_params_real_input = [*xs, *ws, *vs, *us, t0, t1, t2]

# pprint(replace_symbols(P00_real_input, 0, *input_params_real_input))

norm_factor = diamond.normalization_factor
P00_nympy = sp.lambdify(input_params, norm_factor**2 * P00, modules='numpy')
P00_real_input_nympy = sp.lambdify(input_params_real_input, norm_factor**2 * P00_real_input)
P01_real_input_nympy = sp.lambdify(input_params_real_input, norm_factor**2 * P01_real_input)
P10_real_input_nympy = sp.lambdify(input_params_real_input, norm_factor**2 * P10_real_input)
P11_real_input_nympy = sp.lambdify(input_params_real_input, norm_factor**2 * P11_real_input)

print('--- Plotting ---')
x = np.linspace(0, 1)
x_rev = list(reversed(x))
y00 = np.zeros(len(x))
y01 = np.zeros(len(x))
y10 = np.zeros(len(x))
y11 = np.zeros(len(x))
ysum = np.zeros(len(x))
for i in range(len(x)):
    # y[i] = P00_real_input_nympy(x[i], 1, 1, 1, 1, 1/2, 1/3, 1/2, 1/3, 2/3, 2/3, 1/3, np.pi/2, np.pi/2)
    input = [1/2, x[i], 1/2, 1/2,  # Input x
             1, 1/3, 1, 1/3,           # Weights 1: w
             1, 1, 1/2, 1,           # Weights 1: v
             1/2, 2/3, 1/2, 1,           # Weights 1: u (may not be used at the moment: see above)
             np.pi, np.pi, np.pi] # U-time: t0, t1
    y00[i] = P00_real_input_nympy(*input)
    y01[i] = P01_real_input_nympy(*input)
    y10[i] = P10_real_input_nympy(*input)
    y11[i] = P11_real_input_nympy(*input)
    ysum[i] = y00[i] + y01[i] + y10[i] + y11[i]

plt.figure()
ax = plt.gca()
plt.title('P00, P01, P10, P11')
plt.plot(x, y00, label='$P_{00}$')
plt.plot(x, y01, label='$P_{01}$')
plt.plot(x, y10, label='$P_{10}$')
plt.plot(x, y11, label='$P_{11}$')
plt.plot(x, ysum)
plt.xlim(min(x), max(x))
# plt.ylim(0, 1)
ax.legend()
plt.show()
