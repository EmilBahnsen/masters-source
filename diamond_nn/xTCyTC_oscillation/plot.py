import matplotlib.pyplot as plt
import numpy as np
import cloudpickle
from sympy_diamond import *

π = np.pi

with open(r"analytic_P.pickle", "rb") as input_file:
    P00_numpy, P01_numpy, P10_numpy, P11_numpy = cloudpickle.load(input_file)

print('--- Plotting ---')
data_len = 50
x_linspace = np.linspace(0, 1, data_len)
y_linspace = x_linspace

x0 = np.linspace(0, 2*π, data_len)
x1 = np.linspace(π, 2*π, data_len)
x2 = np.linspace(2*π, π, data_len)
x3 = np.linspace(1, 2*π, data_len)
x4 = np.linspace(2, 2*π, data_len)
x5 = np.linspace(0, 2*π, data_len)
x6 = np.linspace(0.5, 2*π, data_len)
x7 = np.linspace(2*π, 1, data_len)
x8 = np.linspace(0.1, 2*π, data_len)
x9 = np.linspace(0.2, π, data_len)
x10 = np.linspace(0.3, 2*π, data_len)

y00 = np.zeros(data_len)
y01 = np.zeros(data_len)
y10 = np.zeros(data_len)
y11 = np.zeros(data_len)
ysum = np.zeros(data_len)

# Two 1-qubit tensor state embedding
def embed_two_tensor_qubits(a, b, c, d):
    return np.array([
        np.cos(a / 2) * np.cos(c / 2),
        np.cos(a / 2) * np.sin(c / 2) * np.exp(1j * d),
        np.sin(a / 2) * np.cos(c / 2) * np.exp(1j * b),
        np.sin(a / 2) * np.sin(c / 2) * np.exp(1j * (b + d))
    ]).transpose()

# Two 1-qubit tensor state embedding with iSWAP
a, b, c, d, e = sp.symbols('a b c d e', real=True)
o1 = sp.kronecker_product(U3(a, b, 0), U3(c, d, 0))
o2 = iSWAP(e)
_state_lambda = sp.lambdify((a, b, c, d, e), (o2 @ o1 @ s00), modules='numpy')
def embed_two_tensor_qubits_w_iswap(a, b, c, d, e):
    return _state_lambda(a, b, c, d, e)[:, 0, ...].transpose()


# Two 1-qubit tensor state embedding with iSWAP
a = sp.symbols('a:12', real=True)
o1 = sp.kronecker_product(U3(a[0], a[1], 0), U3(a[2], a[3], 0))
o2 = iSWAP(a[4])
o3 = sp.kronecker_product(U3(a[5], a[6], a[7]), U3(a[8], a[9], a[10]))
_state_lambda2 = sp.lambdify(a[:11], (o3 @ o2 @ o1 @ s00), modules='numpy')
def embed_two_tensor_qubits_U3_ISWAP_U3(*args):
    return _state_lambda2(*args)[:, 0, ...].transpose()


r_embed = embed_two_tensor_qubits_U3_ISWAP_U3(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
r_embed_ij = lambda i,j: embed_two_tensor_qubits_U3_ISWAP_U3(x0[i], 1, 1, 1, x1[j], 1, 1, 1, 1, 1, 1)
w_embed = embed_two_tensor_qubits_U3_ISWAP_U3(1/2, 1/3, 1/2, 1/4, 1/6, 1, 1/3, 1/2, 1/2, 1/3, 1)
v_embed = embed_two_tensor_qubits_U3_ISWAP_U3(1/3, 1/2, 1/4, 1/2, 1/5, 1, 1/2, 1/3, 1/2, 1/6, 1/3)
t0 = π/2
t1 = 3*π/2

for i in range(data_len):
    # y[i] = P00_real_input_numpy(x[i], 1, 1, 1, 1, 1/2, 1/3, 1/2, 1/3, 2/3, 2/3, 1/3, np.pi/2, np.pi/2)
    input = np.array([
        *r_embed[i],     # Input x
        *w_embed,                 # Weights 1: w
        *v_embed,                   # Weights 2: v
        t0, t1                          # U-time: t0, t1
    ])
    # noice = np.array([*np.random.normal(0, 0.01, len(input)-2), 0, 0])
    input = input
    y00[i] = P00_numpy(*input)
    y01[i] = P01_numpy(*input)
    y10[i] = P10_numpy(*input)
    y11[i] = P11_numpy(*input)
    ysum[i] = y00[i] + y01[i] + y10[i] + y11[i]


plt.figure()
plt.plot(x_linspace, y00, 'r', label='$P_{00}$',)
plt.plot(x_linspace, y01, 'g', label='$P_{01}$',)
plt.plot(x_linspace, y10, 'b', label='$P_{10}$')
plt.plot(x_linspace, y11, 'k', label='$P_{11}$')
# plt.plot(x, ysum, label='P_{sum}')

ax = plt.gca()
plt.title('P00, P01, P10, P11')
plt.xlim(min(x_linspace), max(x_linspace))
# plt.ylim(0, 1)
ax.legend()
plt.show()
# exit()


print('--- x0 vs x1 plot ---')
out_size = (data_len, data_len)
y00 = np.zeros(out_size)
y01 = np.zeros(out_size)
y10 = np.zeros(out_size)
y11 = np.zeros(out_size)
ysum = np.zeros(out_size)
for i in range(out_size[0]):
    for j in range(out_size[1]):
        # y[i] = P00_real_input_numpy(x[i], 1, 1, 1, 1, 1/2, 1/3, 1/2, 1/3, 2/3, 2/3, 1/3, np.pi/2, np.pi/2)
        x = r_embed_ij(i,j)
        input = np.array([
            *x,  # Input x
            *w_embed,     # Weights 1: w
            *v_embed,     # Weights 2: v
            t0, t1        # U-time: t0, t1
        ])
        # noice = np.array([*np.random.normal(0, 0.01, len(input)-2), 0, 0])
        input = input
        y00[i][j] = P00_numpy(*input)
        y01[i][j] = P01_numpy(*input)
        y10[i][j] = P10_numpy(*input)
        y11[i][j] = P11_numpy(*input)
        ysum[i][j] = y00[i][j] + y01[i][j] + y10[i][j] + y11[i][j]

print('--- scatter plot ---')
P00_scatter = {'x0': [], 'x1': []}
P01_scatter = {'x0': [], 'x1': []}
P10_scatter = {'x0': [], 'x1': []}
P11_scatter = {'x0': [], 'x1': []}
P_scatter = [P00_scatter, P01_scatter, P10_scatter, P11_scatter]
for i in range(out_size[0]):
    for j in range(out_size[1]):
        idx_max = np.argmax([y00[i][j], y01[i][j], y10[i][j], y11[i][j]])
        P_scatter[idx_max]['x0'].append(x_linspace[i])
        P_scatter[idx_max]['x1'].append(y_linspace[j])

plt.figure()
plt.title('x vs phi plot with max P')
plt.scatter(P00_scatter['x0'], P00_scatter['x1'], color='r')
plt.scatter(P01_scatter['x0'], P01_scatter['x1'], color='g')
plt.scatter(P10_scatter['x0'], P10_scatter['x1'], color='b')
plt.scatter(P11_scatter['x0'], P11_scatter['x1'], color='k')
plt.show()

