import math
import tensorflow as tf
from tensorflow import linalg as la
from typing import *
from tf_qc import complex_type, float_type
import qutip as qt

# Basic QC-definitions (including that for the diamond)

# tf.compat.v1.disable_eager_execution()

π = math.pi
c1 = tf.cast(1, dtype=complex_type)

I1 = tf.convert_to_tensor([
    [1, 0],
    [0, 1]
], dtype=complex_type)

H = tf.convert_to_tensor([
    [1, 1],
    [1, -1]
], dtype=complex_type)/math.sqrt(2)

sigmaz = tf.convert_to_tensor(qt.sigmaz().full(), dtype=complex_type)

# SWAP matrices: SWAP[N][target1][target2]
SWAP = []
for N in range(10):
    if N < 2:
        SWAP.append(None)
        continue
    SWAP.append([])
    for i in range(N):
        SWAP[N].append([])
        for j in range(N):
            SWAP[N][i].append(tf.convert_to_tensor(qt.swap(N, [i, j]).full()) if i != j else None)


# Kronecker product, takes a list af tensors like QuTiP
def tensor(tensors: List[tf.Tensor]):
    if len(tensors) == 1:
        return tensors.pop()
    b = tensors.pop()
    a = tensors.pop()
    result = \
        tf.reshape(
            tf.reshape(a, [a.shape[0], 1, a.shape[1], 1]) *
            tf.reshape(b, [1, b.shape[0], 1, b.shape[1]]),
            [a.shape[0] * b.shape[0], a.shape[1] * b.shape[1]]
        )
    return tensor(tensors + [result]) if len(tensors) > 0 else result


# Also QuTiP
def gate_expand_1toN(U: tf.Tensor, N: int, target: int):
    assert target < N, 'target exceeds N: {target} vs {N}'.format(target=target, N=N)
    return tensor([I1] * target + [U] + [I1] * (N - target - 1))


# Also QuTiP
def gate_expand_2toN(U: tf.Tensor, N: int, control: int = None, target: int = None, targets: List[int] = None):
    if targets is not None:
        control, target = targets

    if control is None or target is None:
        raise ValueError("Specify value of control and target")

    if N < 2:
        raise ValueError("integer N must be larger or equal to 2")

    if control >= N or target >= N:
        raise ValueError("control and not target must be integer < integer N")

    if control == target:
        raise ValueError("target and not control cannot be equal")

    # Make the gate work on 0 and 1 as control and target, and then swap with real control and target
    gate = tensor([U] + [I1] * (N - 2))
    if control is not None and control is not 0:
        gate = SWAP[N][0][control] @ gate @ SWAP[N][0][control]
    if target is not None and target is not 1:
        gate = SWAP[N][1][target] @ gate @ SWAP[N][1][target]
    return gate


I2 = tensor([I1] * 2)
I3 = tensor([I1] * 3)
I4 = tensor([I1] * 4)
I = [None] + [tensor([I1] * N) for N in range(1, 10)]

# Construction of U
# Basis states
s0 = tf.convert_to_tensor([[c1], [0.]], dtype=complex_type)  # Column vector
s1 = tf.convert_to_tensor([[0.], [c1]], dtype=complex_type)
s00 = tensor([s0, s0])
s01 = tensor([s0, s1])
s10 = tensor([s1, s0])
s11 = tensor([s1, s1])

# Bell states
sp = (s01 + s10)/math.sqrt(2)  # (|01> + |10>)/√2
sm = (s01 - s10)/math.sqrt(2)  # (|01> - |10>)/√2

o0000 = tensor([s00, la.adjoint(s00)])
o1111 = tensor([s11, la.adjoint(s11)])
opp = tensor([sp, la.adjoint(sp)])
omm = tensor([sm, la.adjoint(sm)])


def U(t):
    it = tf.complex(tf.constant(0, float_type), t)
    # TODO: Why do we have to use 'exp + -1'?!
    U00 = tf.convert_to_tensor([
        [1, 0, 0, 0],
        [0, (tf.exp(-it) + 1) / 2, (tf.exp(-it) + -1) / 2, 0],
        [0, (tf.exp(-it) + -1) / 2, (tf.exp(-it) + 1) / 2, 0],
        [0, 0, 0, tf.exp(-it)]
    ])
    U11 = tf.convert_to_tensor([
        [tf.exp(it), 0, 0, 0],
        [0, (tf.exp(it) + 1) / 2, (tf.exp(it) + -1) / 2, 0],
        [0, (tf.exp(it) + -1) / 2, (tf.exp(it) + 1) / 2, 0],
        [0, 0, 0, 1]
    ])
    Up = tf.convert_to_tensor([
        [tf.exp(it), 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, tf.exp(-it)]
    ])
    Um = tf.convert_to_tensor([
        [c1, 0, 0, 0],
        [0, 1., 0, 0],
        [0, 0, 1., 0],
        [0, 0, 0, 1.]
    ])

    return tensor([o0000, U00]) + \
           tensor([o1111, U11]) + \
           tensor([opp, Up]) + \
           tensor([omm, Um])


# Check that tf-version of U is the same as that for qutip!
from diamond.definitions import U as U_qutip
from qutip import Qobj
# Not precicely the same!
#assert U_qutip(π/16) == Qobj(U(π/16).numpy(), dims=[[2, 2, 2, 2], [2, 2, 2, 2]]), 'TF U is not the same as QuTiP U. diff: {}' + ndtotext((U_qutip(π/16) - Qobj(U(π/16).numpy(), dims=[[2, 2, 2, 2], [2, 2, 2, 2]])).full())


def make_gate(N: int,
              gate: Union[str, tf.Tensor],
              targets: Union[int, List[int]] = None,
              controls: Union[int, List[int]] = None) -> tf.Tensor:
    if type(gate) is str:
        name = gate.lower()
        if name == 'swap':
            gate = SWAP[N][targets[0]][targets[1]]
        elif name == 'h':
            gate = H
        else:
            raise NotImplementedError('Gate not implemented')
    size = gate.shape[0]
    if size == 2 ** N:
        return gate
    elif size == 2:
        return gate_expand_1toN(gate, N, targets)
    elif size == 2 ** 2:
        return gate_expand_2toN(gate, N, controls, targets)
    else:
        raise NotImplementedError('Literal gates > 2**3 not implemented.')


def qft_U(N: int, oper: tf.Tensor, params: List[tf.Variable]) -> tf.Tensor:
    # # Execute QFT
    # tf.print('\n')
    # print_non_zero_unitary_test(oper)
    assert N == 4, 'N != 4 not implemented!'
    n_param = 0
    # H - U(t_1) - H - ... - H - U(t_n) - H
    for i in range(N):
        hadamard_gate = make_gate(N, 'h', i)
        oper = hadamard_gate @ oper
        if i != N-1:
            U_gate = U(params[n_param][0])
            oper = U_gate @ oper
            n_param += 1
    assert n_param == len(params)
    # Final swaps
    for n in range(N // 2):
        swap_gate = make_gate(N, 'swap', [n, N - n - 1])
        oper = swap_gate @ oper
        # tf.print('swap', n)
        # print_non_zero_unitary_test(oper)
    # print_non_zero_unitary_test(oper)
    return oper


def qft(N: int, oper: tf.Tensor) -> tf.Tensor:
    # Execute QFT
    for i in range(N):
        oper = make_gate(N, 'h', i) @ oper
        for n_j, j in enumerate(range(i+1, N)):
            R_k = tf.convert_to_tensor([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, tf.exp(2j*π/(2**(n_j+2)))]], dtype=complex_type)
            oper = gate_expand_2toN(R_k, N, j, i) @ oper
    # Final swaps
    for n in range(N//2):
        oper = make_gate(N, 'swap', [n, (N-1)-n]) @ oper
    return oper


def iqft(N: int, oper: tf.Tensor) -> tf.Tensor:
    # Final swaps (first)
    for n in range(N//2):
        oper = make_gate(N, 'swap', [n, (N-1)-n]) @ oper
    # Execute inverse QFT
    for i in reversed(range(N)):
        for n_j, j in enumerate(range(i+1, N)):
            R_k = tf.convert_to_tensor([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, tf.exp(-2j * π / (2 ** (n_j + 2)))]], dtype=complex_type)
            oper = gate_expand_2toN(R_k, N, j, i) @ oper
        oper = make_gate(N, 'h', i) @ oper
    return oper


# https://quantumcomputing.stackexchange.com/questions/6236/how-to-quickly-calculate-the-custom-u3-gate-parameters-theta-phi-and-lamb
def U3(t_xyz: Union[tf.Tensor, tf.Variable], *args) -> tf.Tensor:
    t_xyz = tf.cast(t_xyz, complex_type)
    gate = tf.convert_to_tensor([
        [tf.cos(t_xyz[0]/2), -tf.exp(1j * t_xyz[2]) * tf.sin(t_xyz[0]/2)],
        [tf.exp(1j * t_xyz[1]) * tf.sin(t_xyz[0]/2), tf.exp(1j * (t_xyz[1]+t_xyz[2])) * tf.cos(t_xyz[0]/2)]
    ], dtype=complex_type)
    if len(args) == 0:
        return gate
    else:
        return tensor([gate, U3(*args)])
