import sympy as sp
from sympy.physics.quantum import TensorProduct, Dagger
from functools import reduce
from typing import *
from sympy import tensorcontraction
import qutip as qt

# Construction of U
# Basis states
s0 = sp.Matrix([[1], [0]])
s1 = sp.Matrix([[0], [1]])
# s0 = Ket(0)
# s1 = Ket(1)
s00 = TensorProduct(s0, s0)
s01 = TensorProduct(s0, s1)
s10 = TensorProduct(s1, s0)
s11 = TensorProduct(s1, s1)

# Bell states
_sp = (s01 + s10)/sp.sqrt(2)  # (|01> + |10>)/√2
_sm = (s01 - s10)/sp.sqrt(2)  # (|01> - |10>)/√2

o0000 = TensorProduct(s00, Dagger(s00))
o1111 = TensorProduct(s11, Dagger(s11))
opp = TensorProduct(_sp, Dagger(_sp))
omm = TensorProduct(_sm, Dagger(_sm))
I = sp.I


def U00(t):
    return sp.Matrix([
        [1, 0, 0, 0],
        [0, (sp.exp(-I*t) + 1) / 2, (sp.exp(-I*t) - 1) / 2, 0],
        [0, (sp.exp(-I*t) - 1) / 2, (sp.exp(-I*t) + 1) / 2, 0],
        [0, 0, 0, sp.exp(-I*t)]
    ])
def U11(t):
    return sp.Matrix([
        [sp.exp(I*t), 0, 0, 0],
        [0, (sp.exp(I*t) + 1) / 2, (sp.exp(I*t) - 1) / 2, 0],
        [0, (sp.exp(I*t) - 1) / 2, (sp.exp(I*t) + 1) / 2, 0],
        [0, 0, 0, 1]
    ])
def Up(t):
    return sp.Matrix([
        [sp.exp(I*t), 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, sp.exp(-I*t)]
    ])
def Um(t):
    return sp.Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def U(t):
    U00_t = U00(t)
    U11_t = U11(t)
    Up_t = Up(t)
    Um_t = Um(t)

    return TensorProduct(o0000, U00_t) + \
           TensorProduct(o1111, U11_t) + \
           TensorProduct(opp, Up_t) + \
           TensorProduct(omm, Um_t)


def state(n_qubits:int, symbol:str):
    x0 = sp.symbols(symbol + ':1' * n_qubits, real=True)
    xrest = sp.symbols(symbol + ':2' * n_qubits, complex=True)
    xs = [*x0, *xrest[1:]]
    state = sp.ImmutableDenseMatrix([*xs])
    return state, xs


def cp_state(n_qubits:int, symbols:List[str]):
    rs = sp.symbols(symbols[0] + ':2' * n_qubits, real=True, nonnegative=True)
    phis = sp.symbols(symbols[1] + ':2' * n_qubits, real=True)
    return sp.ImmutableDenseMatrix([
        [rs[0]],
        *[[rs[i] * sp.exp(sp.I*phis[i])] for i in range(1, len(rs))]
    ]), rs, phis[1:]


U3 = lambda theta, phi, lamb: sp.ImmutableDenseMatrix([
    [sp.cos(theta/2), -sp.exp(I*lamb)*sp.sin(theta/2)],
    [sp.exp(I*phi)*sp.sin(theta/2), sp.exp(I*(lamb + phi))*sp.cos(theta/2)]
])

# https://www.researchgate.net/figure/Example-universal-set-of-quantum-gates-consisting-of-three-single-qubit-rotation-gates_fig3_327671865
# NIelsen and Chuang p. 174
def RX(angle):
    return sp.ImmutableDenseMatrix([
        [sp.cos(angle/2), -1j*sp.sin(angle/2)],
        [-1j*sp.sin(angle/2), sp.cos(angle/2)],
    ])

def RY(angle):
    return sp.ImmutableDenseMatrix([
        [sp.cos(angle/2), -sp.sin(angle/2)],
        [sp.sin(angle/2), sp.cos(angle/2)],
    ])

def RZ(angle):
    return sp.ImmutableDenseMatrix([
        [sp.exp(-1j*angle/2), 0],
        [0, sp.exp(1j*angle/2)],
    ])

def RXZX(x1, z, x2):
    return RX(x2) @ RZ(z) @ RX(x1)  # Slow but works (TODO: write explicitly)

# Defn. from cirq
SWAP = lambda t: sp.ImmutableDenseMatrix([
    [1, 0, 0, 0],
    [0, sp.exp(I*t) * sp.cos(t), -I * sp.exp(I*t) * sp.sin(t), 0],
    [0, -I * sp.exp(I*t) * sp.sin(t), sp.exp(I*t) * sp.cos(t), 0],
    [0, 0, 0, 1]
])

iSWAP = lambda t: sp.ImmutableDenseMatrix([
    [1, 0, 0, 0],
    [0, sp.cos(t), -I * sp.sin(t), 0],
    [0, -I * sp.sin(t), sp.cos(t), 0],
    [0, 0, 0, 1]
])

# iSWAP(pi/2)
def iSWAP_N_pi_2(N, targets):
    return sp.ImmutableDenseMatrix(qt.iswap(N, targets)).subs(1.0, 1).subs(I, -I)

# iSWAP(pi/2 + pi) = iSWAP(3*pi/2)
def iSWAP_N_3pi_2(N, targets):
    return sp.ImmutableDenseMatrix(qt.iswap(N, targets)).subs(1.0, 1)

CNOT = sp.ImmutableDenseMatrix([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
])

X = sp.ImmutableDenseMatrix([
    [0, 1],
    [1, 0]
])

Y = sp.ImmutableDenseMatrix([
    [0, -I],
    [I, 0]
])

Z = sp.ImmutableDenseMatrix([
    [1, 0],
    [0, -1]
])

CZ = lambda t: sp.ImmutableDenseMatrix([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,sp.exp(I*t)]
])

H = sp.ImmutableDenseMatrix([
    [1/sp.sqrt(2), 1/sp.sqrt(2)],
    [1/sp.sqrt(2), -1/sp.sqrt(2)]
])

def partial_trace(dm, n_qubits, subsystem):
    # This is the same calc. as for the tf_qc/qc.trace
    # First we find the last consecutive block to trace away
    block_end = subsystem[-1]
    block_start = block_end
    for idx in reversed(subsystem):
        if block_start - idx <= 1:
            block_start = idx
        else:
            break
    n_static1 = block_start  # First part of static qubits
    n_static2 = (n_qubits - 1) - block_end  # Second part of static qubits
    n_static = n_static1 + n_static2
    n_qubits2trace = block_end - block_start + 1  # Qubits to trace away

    new_shape = [2**n_static1, 2**n_qubits2trace, 2**n_static2, 2**n_static1, 2**n_qubits2trace, 2**n_static2]

    flat_dm = sp.flatten(dm)
    new_dm = sp.NDimArray(flat_dm, shape=new_shape)
    trace_result = tensorcontraction(new_dm, (1, 4))
    reshaped_result = trace_result.reshape(2**n_static, 2**n_static)
    # We must now recursively to the same to the lest of the subsystems
    idx_of_start = subsystem.index(block_start)
    new_subsystem = subsystem[:idx_of_start]
    return partial_trace(reshaped_result, n_static, new_subsystem) if len(new_subsystem) > 0 else reshaped_result.tomatrix()

def partial_trace_last_n_qubits(dm, n_qubits, n_qubits2trace):
    n_static = n_qubits - n_qubits2trace
    new_shape = [2**n_static, 2**n_qubits2trace, 2**n_static, 2**n_qubits2trace]
    flat_expr = sp.flatten(dm)
    new_expr = sp.NDimArray(flat_expr, shape=new_shape)
    return_expr = new_expr[:, 0, :, 0]
    for i in range(1, 2**n_qubits2trace):
        return_expr += new_expr[:, i, :, i]
    return return_expr


def density_matrix(state):
    return state @ Dagger(state)


def normalization_factor(state):
    sum = reduce(lambda acc, elem: acc + elem * sp.conjugate(elem), state, 0)
    return 1 / sp.sqrt(sum)


def normalize_state(state):
    return normalization_factor(state) * state


def int2bin(x, n=0):
    """
    Get the binary representation of x.

    Parameters
    ----------
    x : int
    n : int
        Minimum number of digits. If x needs less digits in binary, the rest
        is filled with zeros.

    Returns
    -------
    str
    """
    return format(x, 'b').zfill(n)


# --- Helpers ---
def hadamard_division(A: sp.Matrix, B: sp.Matrix) -> sp.Matrix:
    assert A.shape == B.shape, f'Rank must be the same got {A.rank()} and {B.rank()}'
    rows, cols = A.shape
    C = sp.MutableMatrix.zeros(rows, cols)
    for row in range(rows):
        for col in range(cols):
            C[row,col] = A[row,col] / B[row,col]
    return C

# --- Helpers ---
def hadamard_product(A: sp.Matrix, B: sp.Matrix):
    assert A.shape == B.shape, f'Rank must be the same got {A.rank()} and {B.rank()}'
    rows, cols = A.shape
    C = sp.MutableMatrix.zeros(rows, cols)
    for row in range(rows):
        for col in range(cols):
            C[row,col] = A[row,col] * B[row,col]
    return C
