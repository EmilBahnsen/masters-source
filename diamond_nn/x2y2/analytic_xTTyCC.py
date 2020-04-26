import sympy as sp
from sympy import pprint
from sympy import Q
from sympy.physics.quantum import TensorProduct, tensor_product_simp, Ket, UnitaryOperator, HermitianOperator, OuterProduct
from sympy.physics.quantum.dagger import Dagger
from tf_qc.qc import intlog2
from functools import reduce
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sympy_diamond import *

# Here we are looking at the diamond in the config.
#
# xa (T1) – ya (C1)
#         X             (There is a cross-over bwt. qubits natually, that's not so beneficial pratically)
# xb (T2) – yb (C2)
#
# So that's 2 of the qubits as an input and 2 as output
# NOTES:
# In this config, there's no change if we init C1 and C2 in 00
# But if we init them in 01, we get a result ... os this is what we use

π = np.pi

t = sp.symbols('t', real=True)
Ut = U(t)

xs = sp.symbols('x00 x01 x10 x11', real=True)
x1, x2, x3, x4 = xs
state = sp.Matrix([
    [x1],
    [x2],
    [x3],
    [x4]
])

ws = sp.symbols('w00 w01 w10 w11', real=True)
w1, w2, w3, w4 = ws
init_w_state = sp.Matrix([
    [w1],
    [w2],
    [w3],
    [w4]
])

state = TensorProduct(init_w_state, state)#.subs(w1, 0).subs(w4, 0)
norm_factor = normalization_factor(state)
sum_xs2 = sum(map(lambda x: x**2, xs))
# state = norm_factor * state
# state = normalize_state(state)
state = Ut @ state
dm = density_matrix(state)
dm = partial_trace_last_n_qubits(dm, 4, 2)
# Measurement of C1
dm_C1 = partial_trace_last_n_qubits(dm, 2, 1)
P0_C1 = dm_C1[0, 0].simplify()
P1_C1 = dm_C1[1, 1].simplify()
P01_C1 = (P0_C1 + P1_C1).simplify()
# Measurement of C2
dm_C2 = partial_trace(dm, 2, [0])
P0_C2 = dm_C2[0, 0].simplify()
P1_C2 = dm_C2[1, 1].simplify()
P01_C2 = (P0_C2 + P1_C2).simplify()
# Measurement of C1 and C2 at once
P00 = dm[0, 0].simplify()
P01 = dm[1, 1].simplify()
P10 = dm[2, 2].simplify()
P11 = dm[3, 3].simplify()


print('state')
pprint(state)
print('norm_factor')
pprint(norm_factor)
print('sum_xs2')
pprint(sum_xs2)
print('P0')
pprint(P0_C1.subs(sum_xs2, 1))
print('P1')
pprint(P1_C1.collect(1/4))
print('P00')
pprint(P00)
print('P01')
pprint(P01)
print('P10')
pprint(P10)
print('P11')
pprint(P11)

print()

# --- Diff. of vars ---
for i, x in enumerate(xs):
    print(f'dP0_C1/dx{i}')
    pprint(sp.diff(P0_C1, x).simplify().collect(x))
    print(f'dP0_C2/dx{i}')
    pprint(sp.diff(P0_C2, x).simplify().collect(x))

print()

for i, x in enumerate(xs):
    print(f'dP00/dx{i}')
    pprint(sp.diff(P00, x).simplify().collect(x))
for i, x in enumerate(xs):
    print(f'dP01/dx{i}')
    pprint(sp.diff(P01, x).simplify().collect(x))
for i, x in enumerate(xs):
    print(f'dP10/dx{i}')
    pprint(sp.diff(P10, x).simplify().collect(x))
for i, x in enumerate(xs):
    print(f'dP11/dx{i}')
    pprint(sp.diff(P11, x).simplify().collect(x))

print('Measuring 00 or 11 is proportional to |w00|**2 and |w11|**2, respectively.')
print('Which is of no use ')

exit()
# --- Plotting ---

# Norm factor is not carried by P0, and is squared bc of the
P0_numpy = sp.lambdify(([x1, x2, x3, x4], t), norm_factor**2 * P0_C1, 'numpy')

x = np.linspace(.1, 1, 100)
t = np.linspace(0, 2*π, 100)
x_vector = [0, 0, 0, x]
y = P0_numpy(x_vector, 0.3)

matplotlib.use('TkAgg')
from matplotlib.widgets import Slider

plt.ion()
fig = plt.figure()

P0_ax = plt.axes([0.1, 0.2, 0.8, 0.65])
slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])

plt.axes(P0_ax) # select P0_ax
plt.title('P_0(t)')
text = plt.text(.2, .2, '0')
P0_plot, = plt.plot(x, y, 'r')
plt.ylim([0, 1])

slider = Slider(slider_ax, 't', min(t), max(t), valinit=3)


def update(t):
    x_vector = [x, x/5, x/4, 1/3, 1, 1, 1/2, 1]
    y = P0_numpy(x_vector, t)
    text.set_text(f'{round(max(y.real), 2)} \pm {round(np.std(y), 2)}')
    P0_plot.set_ydata(y)
    fig.canvas.draw_idle()


slider.on_changed(update)

update(3)
plt.show()
input('')
