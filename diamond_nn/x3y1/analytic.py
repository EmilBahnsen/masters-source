import sympy as sp
from sympy import pprint
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sympy_diamond import *

# Here we are looking at the diamond in the config.
#
# xa (C2)
# |       \
# xb (T1)  y (C1)
# |      /
# xc (T2)
#
# So that's 3 of the qubits as an input and one as output

def replace_symbols(expr, value, *args):
    if len(args) == 0:
        return expr
    return reduce(lambda acc, arg: acc.subs(arg, value), args[0], replace_symbols(expr, value, *args[1:]))

π = np.pi

t = sp.symbols('t', real=True, nonnegative=True)
Ut = U(t).subs(sp.exp(1j*t), sp.cos(t) + 1j*sp.sin(t))

xs = sp.symbols('x:2:2:2', real=True, nonnegative=True)
phis = sp.symbols('phi:2:2:2', real=True)
x1, x2, x3, x4, x5, x6, x7, x8 = xs
_, phi2, phi3, phi4, phi5, phi6, phi7, phi8 = phis
state = sp.ImmutableDenseMatrix([
    [x1],
    [x2 * sp.exp(1j*phi2)],
    [x3 * sp.exp(1j*phi3)],
    [x4 * sp.exp(1j*phi4)],
    [x5 * sp.exp(1j*phi5)],
    [x6 * sp.exp(1j*phi6)],
    [x7 * sp.exp(1j*phi7)],
    [x8 * sp.exp(1j*phi8)],
])

ws = sp.symbols('w:2', real=True, nonnegative=True)
omegas = sp.symbols('omega:2', real=True)
w1, w2 = ws
_, omega2 = omegas
init_w_state = sp.ImmutableDenseMatrix([
    [w1],
    [w2 * sp.exp(1j*omega2)]
])

state = sp.kronecker_product(init_w_state, state)
norm_factor = normalization_factor(state)
# state = norm_factor * state
# state = normalize_state(state)
state = Ut @ state
state = state.subs(1.0, 1).expand(basic=True, complex=True).simplify()
dm = density_matrix(state)
dm = partial_trace_last_n_qubits(dm, 4, 3)
P0 = dm[0, 0]
P1 = dm[1, 1]
P01 = (P0 + P1)
P0_real = replace_symbols(P0, 0, phis, omegas)
P1_real = replace_symbols(P1, 0, phis, omegas)


print('state')
pprint(state)
print('norm_factor')
pprint(norm_factor)
sum_xs2 = sum(map(lambda x: 1.0*x**2, xs))
print('sum_xs2')
pprint(sum_xs2)
print('P0')
pprint(P0.subs(sum_xs2, 1).simplify().collect(xs))
print('P1')
pprint(P1.subs(sum_xs2, 1).simplify().collect(xs))
print('P01')
pprint(P01.subs(sum_xs2, 1))

print('P0_real')
pprint(P0_real)
print('P1_real')
pprint(P1_real)

# --- Diff. of vars ---
for i, x in enumerate(xs):
    print(f'dP0/dx{i+1}')
    pprint(sp.diff(P0, x))

exit()
# --- Plotting ---

# Norm factor is not carried by P0, and is squared bc of the
P0_numpy = sp.lambdify(([x1, x2, x3, x4, x5, x6, x7, x8], [w1, w2], t), norm_factor**2 * P0, 'numpy')

x = np.linspace(.001, 1, 100)
t = np.linspace(0, π, 100)
x_vector = [0, 0, 0, x, x, x, 0, x]
w_vector = [1, 0]
y = P0_numpy(x_vector, w_vector, 0.3)

matplotlib.use('TkAgg')
from matplotlib.widgets import Slider

plt.ion()
fig = plt.figure()

P0_ax = plt.axes([0.1, 0.5, 0.8, 0.45])
slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])
slider2_ax = plt.axes([0.1, 0.15, 0.8, 0.05])
slider3_ax = plt.axes([0.1, 0.25, 0.8, 0.05])
slider4_ax = plt.axes([0.1, 0.35, 0.8, 0.05])

plt.axes(P0_ax) # select P0_ax
plt.title('P_0(x)')
text = plt.text(.2, .2, '0')
P0_plot, = plt.plot(x, y, 'r')
plt.xlim([0, 1])
plt.ylim([0, 1])

slider = Slider(slider_ax, 'w0', min(x), max(x), valinit=0.1)
slider2 = Slider(slider2_ax, 'w1', min(x), max(x), valinit=0.1)
slider3 = Slider(slider3_ax, 'x0', min(x), max(x), valinit=0.1)
slider4 = Slider(slider4_ax, 't', min(t), max(t), valinit=max(t))


def update_slider1(w1):
    update(w1=w1)

def update_slider2(w2):
    update(w2=w2)

def update_slider3(x1):
    update(x1=x1)

def update_slider4(t):
    update(t=t)


t_val = [0.1]
x1_val = [0.1]
w_vector = [slider.val, slider2.val]
def update(w1=None, w2=None, x1=None, t=None):
    if x1 is not None: x1_val[0] = x1
    x_vector = [x, x1_val[0], 0/4, 0/3, 0, 0, 0/2, 0]
    if w1 is not None: w_vector[0] = w1
    if w2 is not None: w_vector[1] = w2
    if t is not None: t_val[0] = t
    y = P0_numpy(x_vector, w_vector, t_val[0])
    text.set_text(f'{round(max(y.real), 2)} \pm {round(np.std(y), 2)}')
    P0_plot.set_ydata(y)
    fig.canvas.draw_idle()


slider.on_changed(update_slider1)
slider2.on_changed(update_slider2)
slider3.on_changed(update_slider3)
slider4.on_changed(update_slider4)

update(slider.val, slider2.val)
plt.show()
input('')
