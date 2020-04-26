import sympy as sp
from sympy import pprint
from sympy_diamond import *

# AS OF: 2003.09887
def amplitude_encoding(x, basis=None):
    if basis is None:
        state = sp.Matrix([
            *[[xi] for xi in x]
        ])
    else:
        state = sum([xi*b for xi, b in zip(x, basis)])
    return normalize_state(state)

# def product_encoding

π = sp.pi
I = sp.I

I1 = sp.eye(2)


print('--- 1-qubit state creation ---')
s0 = sp.ImmutableDenseMatrix([
    [1],
    [0]
])
state = s0
a = sp.symbols('a:20')
o1 = U3(a[0], a[1], a[2])
state = o1 @ state
pprint(state)


print('--- Two 1-qubit tensor state creation ---')
s0 = sp.ImmutableDenseMatrix([
    [1],
    [0]
])
state = sp.kronecker_product(s0, s0)
a = sp.symbols('a:20')
o1 = sp.kronecker_product(U3(a[0], a[1], a[2]), U3(a[3], a[4], a[5]))
state = o1 @ state
pprint(state)

print('--- 2-qubit state preparation ---')
s00 = sp.kronecker_product(s0, s0)
state = s00
a = sp.symbols('a:20')
b = sp.symbols('b:15')
x, y, z = map(sp.Wild, 'xyz')
I_coscos = (z*sp.cos(x)*sp.cos(y), z*(sp.cos(x - y) + sp.cos(x + y))/2)
I_sinsin = (z*sp.sin(x)*sp.sin(y), z*(sp.cos(x - y) - sp.cos(x + y))/2)
I_sincos = (z*sp.sin(x)*sp.cos(y), z*(sp.sin(x + y) + sp.sin(x - y))/2)
I_cossin = (z*sp.cos(x)*sp.sin(y), z*(sp.sin(x + y) - sp.sin(x - y))/2)
# No initial z-rotation, doesn't make a diff.
o1 = sp.kronecker_product(U3(a[0], b[0], 0), U3(a[1], b[1], 0))
o2 = iSWAP(a[2])
o3 = sp.kronecker_product(U3(a[3], b[3], b[4]), U3(a[4], b[5], b[6]))
state = o1 @ state
state = o2 @ state
state = o3 @ state
state = state.simplify()

f0 = sp.Function('f0')(a[0], a[1], a[2], b[0])
f1 = sp.Function('f1')(a[0], a[1], a[2], b[0])
pprint(f0)
state = state.replace(sp.exp(I*b[0]) * sp.sin(a[0]/2) * sp.cos(a[1]/2) * sp.cos(a[2]), f0)
state = state.replace(sp.exp(I*b[0]) * sp.sin(a[0]/2) * sp.cos(a[1]/2) * sp.sin(a[2]), f1)
# state = state.expand()
# state = state.replace(*I_coscos).replace(*I_sinsin).replace(*I_sincos).replace(*I_cossin)
# state = state.expand()
# state = state.replace(*I_coscos).replace(*I_sinsin).replace(*I_sincos).replace(*I_cossin)
# state = state.simplify()
pprint(state)
for i in range(4):
    pprint(state[i].free_symbols)