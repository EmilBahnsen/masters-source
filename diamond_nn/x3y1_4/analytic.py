import sympy as sp
from sympy import pprint
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sympy_diamond import *

# Here we are looking at the diamond in the config.
#
# xa (C2)
# |       \(t1)
# xb (T1)  wa (C1)
# |      /        \
# xc (T2)          \
#                   \
# xd (C2)            wa' (C2)
# |       \(t2)      |        \(t4)
# xe (T1)  wb (C1) – wb' (T1)  y (C1) –> MEASURE
# |      /           |       /
# xf (T2)            wc' (T2)
#                   /
# xg (C2)          /
# |       \(t3)   /
# xh (T1)  wc (C1)
# |      /
# xi (T2)
#
# So that's 3 of the qubits as an input and one as output (in each diamond)

π = np.pi

t1, t2, t3, t4 = sp.symbols('t1 t2 t3 t4', real=True)
Ut1 = U(t1)
Ut2 = U(t2)
Ut3 = U(t3)
Ut4 = U(t4)

input_symbols = ' '.join([f'x__{int2bin(n, 9)}' for n in range(2**9)])
xs = sp.symbols(input_symbols, real=True)
input_state = sp.Matrix([[x] for x in xs])
input_dm = density_matrix(input_state)
pprint(input_dm)
dm1 = partial_trace(input_dm, 9, [3, 4, 5, 6, 7, 8])
dm2 = partial_trace(input_dm, 9, [0, 1, 2, 6, 7, 8])
dm3 = partial_trace(input_dm, 9, [0, 1, 2, 3, 4, 5])
pprint(dm1)

ws_a = sp.symbols('w__0_a w__1_a', real=True)
ws_b = sp.symbols('w__0_b w__1_b', real=True)
ws_c = sp.symbols('w__0_c w__1_c', real=True)

init_w_a_state = sp.Matrix([[w] for w in ws_a])
init_w_b_state = sp.Matrix([[w] for w in ws_b])
init_w_c_state = sp.Matrix([[w] for w in ws_c])

state = TensorProduct(init_w_state, state)
norm_factor = normalization_factor(state)
# state = norm_factor * state
# state = normalize_state(state)
state = Ut @ state
dm = density_matrix(state)
dm = partial_trace_last_n_qubits(dm, 4, 3)
P0 = dm[0, 0].simplify()
P1 = dm[1, 1].simplify()
P01 = (P0 + P1).simplify()


print('state')
pprint(state)
print('norm_factor')
pprint(norm_factor)
sum_xs2 = sum(map(lambda x: 1.0*x**2, xs))
print('sum_xs2')
pprint(sum_xs2)
print('P0')
pprint(P0.subs(sum_xs2, 1).simplify().collect(w1).collect(w2))
print('P1')
pprint(P1.subs(sum_xs2, 1).simplify().collect(w1).collect(w2))
print('P01')
pprint(P01.subs(sum_xs2, 1))

# --- Diff. of vars ---
for i, x in enumerate(xs):
    print(f'dP0/dx{i}')
    pprint(sp.diff(P0, x))

# exit()
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
