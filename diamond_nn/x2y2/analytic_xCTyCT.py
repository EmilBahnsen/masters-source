import sympy as sp
from sympy import pprint
from qutip.qip.operations import swap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sympy_diamond import *

# Here we are looking at the diamond in the config.
#
# xa (C1) – ya (T2)
# |          |
# xb (T1) – yb (C2)
#
# So that's 2 of the qubits as an input and 2 as output

π = sp.pi

t = sp.symbols('t', real=True, nonnegative=True)
# Ut = U(t).subs(t, π/4).subs(sp.exp(1j*π/4), (1+1j)/sp.sqrt(2))
Ut = U(t).subs(sp.exp(1j*t), sp.cos(t) + 1j*sp.sin(t))

xs = sp.symbols('x:2:2', real=True, nonnegative=True)
phis = sp.symbols('phi:2:2', real=True)
x1, x2, x3, x4 = xs
_, phi2, phi3, phi4 = phis
state = sp.ImmutableDenseMatrix([
    [x1],
    [x2 * sp.exp(1j*phi2)],
    [x3 * sp.exp(1j*phi3)],
    [x4 * sp.exp(1j*phi4)]
])

ws = sp.symbols('w:2:2', real=True, nonnegative=True)
omegas = sp.symbols('omega:2:2', real=True)
ws_conj = (sp.conjugate(w) for w in ws)
w1, w2, w3, w4 = ws
_, omega2, omega3, omega4 = omegas
init_w_state = sp.ImmutableDenseMatrix([
    [w1],
    [w2 * sp.exp(1j*omega2)],
    [w3 * sp.exp(1j*omega3)],
    [w4 * sp.exp(1j*omega4)]
])
xs2_sum = sum([1.0*xs[i]**2 for i in range(4)])
ws2_sum = sum([1.0*ws[i]**2 for i in range(4)])

state = sp.kronecker_product(state, init_w_state)
state = swap(4, [1, 2]) @ state
norm_factor = normalization_factor(state)
state2_sum = 1/norm_factor**2
# state = norm_factor * state
# state = normalize_state(state)
state = Ut @ state
state = state.subs(1.0, 1).expand(basic=True, complex=True).simplify()

dm = density_matrix(state)#.subs(xs[0]*sp.conjugate(xs[0]), sp.abs xs[0].norm()**2)
# Measurement of C2
dm_C2 = partial_trace(dm, 4, [0, 2, 3])
P0_C2 = dm_C2[0, 0].simplify()
P1_C2 = dm_C2[1, 1].simplify()
P01_C2 = (P0_C2 + P1_C2).simplify()
# Measurement of T2
dm_T2 = partial_trace(dm, 4, [0, 1, 2])
P0_T2 = dm_T2[0, 0].simplify()
P1_T2 = dm_T2[1, 1].simplify()
P01_T2 = (P0_T2 + P1_T2).simplify()
# Measurement of C1 and C2 at once
dm_C2T2 = partial_trace(dm, 4, [0, 2])
P00 = dm_C2T2[0, 0].simplify()#.collect(xs)
P01 = dm_C2T2[1, 1].simplify()#.collect(xs)
P10 = dm_C2T2[2, 2].simplify()#.collect(xs)
P11 = dm_C2T2[3, 3].simplify()#.collect(xs)

# Measurement of C1
dm_C1 = partial_trace(dm, 4, [1, 2, 3])
P0_C1 = dm_C1[0, 0].simplify()
P1_C1 = dm_C1[1, 1].simplify()
P01_C1 = (P0_C1 + P1_C1).simplify()
# Measurement of T1
dm_T1 = partial_trace(dm, 4, [0, 1, 3])
P0_T1 = dm_T1[0, 0].simplify()
P1_T1 = dm_T1[1, 1].simplify()
P01_T1 = (P0_T1 + P1_T1).simplify()
# Measurement of C1 and C2 at once
dm_C1T1 = partial_trace(dm, 4, [1, 3])
P00_C1T1 = dm_C1T1[0, 0].simplify()#.collect(xs)
P01_C1T1 = dm_C1T1[1, 1].simplify()#.collect(xs)
P10_C1T1 = dm_C1T1[2, 2].simplify()#.collect(xs)
P11_C1T1 = dm_C1T1[3, 3].simplify()#.collect(xs)


pprint('state')
pprint(state)
print('norm_factor')
pprint(norm_factor)
print('P00')
pprint(P00)
print('P01')
pprint(P01)
print('P10')
pprint(P10)
print('P11')
pprint(P11)
print('P00_C1T1')
pprint(P00_C1T1)
print('P01_C1T1')
pprint(P01_C1T1)
print('P10_C1T1')
pprint(P10_C1T1)
print('P11_C1T1')
pprint(P11_C1T1)

print()

print('P00 + P00_C1T1')
pprint(P00 + P00_C1T1)
print('P01 + P01_C1T1')
pprint(P01 + P01_C1T1)
print('P10 + P10_C1T1')
pprint(P10 + P10_C1T1)
print('P11 + P11_C1T1')
pprint(P11 + P11_C1T1)

print()

# --- Diff. of vars ---
for i, x in enumerate(xs):
    print(f'dP0_T2/dx{i+1}')
    pprint(sp.diff(P0_T2, x).simplify().collect(x))
    print(f'dP0_C2/dx{i+1}')
    pprint(sp.diff(P0_C2, x).simplify().collect(x))
    print(f'dP0_C1/dx{i+1}')
    pprint(sp.diff(P0_C1, x).simplify().collect(x))
    print(f'dP0_T1/dx{i+1}')
    pprint(sp.diff(P0_T1, x).simplify().collect(x))

print('--- OUTPUT ---')

for i, x in enumerate(xs):
    print(f'dP00/dx{i+1}')
    pprint(sp.diff(P00, x).simplify().collect(x))
for i, x in enumerate(xs):
    print(f'dP01/dx{i+1}')
    pprint(sp.diff(P01, x).simplify().collect(x))
for i, x in enumerate(xs):
    print(f'dP10/dx{i+1}')
    pprint(sp.diff(P10, x).simplify().collect(x))
for i, x in enumerate(xs):
    print(f'dP11/dx{i+1}')
    pprint(sp.diff(P11, x).simplify().collect(x))

print('--- INPUT AS OUTPUT ---')

for i, x in enumerate(xs):
    print(f'dP00_C1T1/dx{i+1}')
    pprint(sp.diff(P00_C1T1, x).subs(ws2_sum, 1).simplify().collect(x))
for i, x in enumerate(xs):
    print(f'dP01_C1T1/dx{i+1}')
    pprint(sp.diff(P01_C1T1, x).subs(ws2_sum, 1).simplify().collect(x))
for i, x in enumerate(xs):
    print(f'dP10_C1T1/dx{i+1}')
    pprint(sp.diff(P10_C1T1, x).subs(ws2_sum, 1).simplify().collect(x))
for i, x in enumerate(xs):
    print(f'dP11_C1T1/dx{i+1}')
    pprint(sp.diff(P11_C1T1, x).subs(ws2_sum, 1).simplify().collect(x))

print('So as we can see the one where we actually use the output (T1, C2) as output of the calculation.')
print('There are more dependencies on the inputs, i.e. all the x-es are included in the probabilities for P00, P01, P10, P11.')
print('One the other hand if we use the input as output, then there is just dependence on 3 x-es at a time.')

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
