import math
from typing import Union, List

import opt_einsum as oe
import tensorflow as tf
from tensorflow import linalg as la
from typing import *
from tf_qc import complex_type, float_type, QubitState, QubitDensityMatrix, QubitStateOrDM, Matrix
import qutip as qt
from qutip.qip.operations import swap as qt_swap
from qutip.qip.operations import iswap as qt_iswap
from functools import reduce

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
# TODO: Don't use these, make swaps from scratch in SWAPLayer
# SWAP = []
# for N in range(13):
#     if N < 2:
#         SWAP.append(None)
#         continue
#     SWAP.append([])
#     for i in range(N):
#         SWAP[N].append([])
#         for j in range(N):
#             SWAP[N][i].append(tf.convert_to_tensor(qt.swap(N, [i, j]).full(), complex_type) if i != j else None)


def SWAP(N: int, i: int, j: int):
    def make_swap(N: int, i: int, j: int):
        # Swap them if i > j
        if i > j:
            i, j = j, i
        if j - i == 1:
            swap_tensor = tf.convert_to_tensor([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ], tf.int32)
            # Squeeze it as 'tensor' returns an extra batch dim.
            return tf.squeeze(tensor([tf.eye(2**i, dtype=tf.int32), swap_tensor, tf.eye(2**(N-1-j), dtype=tf.int32)]))
        side_swap = make_swap(N, i, j-1)
        adjacent_swap = make_swap(N, j-1, j)
        return tf.matmul(
            tf.matmul(side_swap, adjacent_swap, a_is_sparse=True, b_is_sparse=True),
            side_swap, a_is_sparse=True, b_is_sparse=True)
    return tf.cast(make_swap(N, i, j), complex_type)


def iSWAP(t):
    mi = tf.complex(0., -1.)
    return tf.convert_to_tensor([
        [1, 0, 0, 0],
        [0, tf.cos(t), mi * tf.cast(tf.sin(t), tf.complex64), 0],
        [0, mi * tf.cast(tf.sin(t), tf.complex64), tf.cos(t), 0],
        [0, 0, 0, 1]
    ], complex_type)


# Kronecker product, takes a list af tensors like QuTiP
def tensor(tensors: List[tf.Tensor], outshape: Optional[Tuple] = None):
    if len(tensors) == 1:
        if outshape is None:
            return tensors.pop()
        else:
            return tf.reshape(tensors.pop(), outshape)
    b = tensors.pop()
    a = tensors.pop()
    result = \
        tf.reshape(
            tf.reshape(a, [-1, a.shape[-2], 1, a.shape[-1], 1]) *
            tf.reshape(b, [-1, 1, b.shape[-2], 1, b.shape[-1]]),
            [-1, a.shape[-2] * b.shape[-2], a.shape[-1] * b.shape[-1]]
        )
    return tensor(tensors + [result], outshape)


def commutator(a: tf.Tensor, b: tf.Tensor):
    return a@b - b@a


def product(tensors: List[tf.Tensor]):
    return reduce(lambda U1, U2: U1 @ U2, tensors)


def inner_product(a: QubitState, b: QubitState):
    return tf.squeeze(tf.matmul(a, b, adjoint_a=True))


def outer_product(a: tf.Tensor, b: tf.Tensor):
    return tf.matmul(a, b, adjoint_b=True)


def density_matrix(states: QubitStateOrDM, subsystem: List[int] = None):
    if states.shape[-1] != 1:
        return states
    if subsystem is not None:
        return density_matrix_trace_from_state(states, subsystem)
    return outer_product(states, states)


def purity(a: QubitStateOrDM):
    rho = a
    if isinstance(a, QubitState):
        rho = density_matrix(a)
    return tf.linalg.trace(rho**2)


def trace(matrices: Matrix,
          subsystem: List[int] = None):
    n_qubits = intlog2(matrices.shape[-1])
    if subsystem is None:
        return tf.linalg.trace(matrices)
    else:
        # Make sure it's sorted
        subsystem = sorted(subsystem)
    # If we trace over the last qubits in the sequence it's quite straight forward
    trace_is_over_last_qubits = max(subsystem) == n_qubits-1 and \
                                min(subsystem) == n_qubits - len(subsystem) and \
                                sum(subsystem) == sum(range(min(subsystem), n_qubits))
    if trace_is_over_last_qubits:
        # If it's a partial trace (over last qubits) then we divide
        # the system into [batch_dims, static, trace_sys, static, trace_sys]
        # and then do the trace over '...bab' with an Einstein sum, and then reshape into the old shape
        n_qubits2trace = len(subsystem)
        n_static = n_qubits - n_qubits2trace
        batch_shape = matrices.shape[:-2]
        matrices = tf.reshape(matrices, [-1, 2**n_static, 2**n_qubits2trace, 2**n_static, 2**n_qubits2trace])
        trace_result = tf.einsum('...bab', matrices)  # Watch the alphabetic order of subscripts!
        return tf.reshape(trace_result, [*batch_shape, 2**n_static, 2**n_static])
    else:
        # Now this is done in more steps... as TF doesn't support einsum with rank > 6
        # We divide the system into [batch_dims, static1, trace_sys, static2, static1, trace_sys, static2],
        # where 'static1 + static2' accounts of all the static entries on either side of the last
        # consecutive block of qubits that is to be traced. That is, we do the partial trace over the
        # last consecutive qubits to be traced and then recursively the rest.

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
        batch_shape = matrices.shape[:-2]
        # This shape is what we wound have used, but this has rank 7
        # new_shape = [-1, 2**n_static1, 2**n_qubits2trace, 2**n_static2, 2**n_static1, 2**n_qubits2trace, 2**n_static2]
        new_shape = [-1, 2**n_qubits2trace, 2**n_static2, 2**n_static1, 2**n_qubits2trace, 2**n_static2]
        matrices = tf.reshape(matrices, new_shape)
        trace_result = tf.einsum('...abcad', matrices)  # Watch the alphabetic order of subscripts!
        reshaped_result = tf.reshape(trace_result, [*batch_shape, 2**n_static, 2**n_static])
        # We must now recursively to the same to the lest of the subsystems
        idx_of_start = subsystem.index(block_start)
        new_subsystem = subsystem[:idx_of_start]
        return trace(reshaped_result, new_subsystem) if len(new_subsystem) > 0 else reshaped_result


def density_matrix_trace_from_state(states: QubitState, subsystem: List[int] = None):
    """
    Take the (partial) trace of a density matrix calculated from the fiven state,
    i.e. \rho_subsystem = Tr_subsystem^C(states @ states^\dag). Where ^C denotes
    the 'complimentory' set.
    :param states:
    :param subsystem:
    :return:
    """
    # If we trace over the last qubits in the sequence it quite straight forward
    n_qubits = intlog2(states.shape[-2])
    subsystem2trace = list(set(range(n_qubits)).difference(set(subsystem)))
    n_qubits2trace = len(subsystem2trace)
    n_qubits_static = n_qubits - n_qubits2trace
    subsys_is_last = max(subsystem2trace) == n_qubits - 1 and \
                     min(subsystem2trace) == n_qubits - len(subsystem2trace) and \
                     sum(subsystem2trace) == sum(range(min(subsystem2trace), n_qubits))
    if subsystem2trace and subsys_is_last:
        matrix_size = 2**n_qubits_static
        result = 0.0  # tf.zeros(shape, complex_type)
        # Do the trace over the last elements
        new_ket_shape = [-1, 2 ** n_qubits_static, 2 ** n_qubits2trace, 1]
        ket = tf.reshape(states, new_ket_shape)
        # new_bra_shape = [-1, 1, 2 ** n_qubits_static, 2 ** n_qubits2trace]
        # bra = tf.reshape(tf.linalg.adjoint(ket), new_bra_shape)
        for n in range(2**n_qubits2trace):
            new_bra_idx_shape = [-1, 1, 2 ** n_qubits_static]
            ket_idx = ket[:, :, n, :]
            bra_idx = tf.reshape(tf.linalg.adjoint(ket_idx), new_bra_idx_shape)
            result += ket_idx @ bra_idx
            # result += ket[:, :, n, :] @ bra[:, :, :, n]
        return result
    else:
        print(Warning('Not optimized not to trace away last qubits.'))
        return trace(density_matrix(states), subsystem2trace)


def measure(states: QubitState, subsystem: List[int] = None):
    """
    Get the probabilities for each outcome, maybe on a subsystem
    :param states:
    :param subsystem:
    :return:
    """
    probs = states * tf.math.conj(states)
    if subsystem is None:
        return probs
    # We must trace away qubits that does not influence the probabilities
    subsystem = sorted(subsystem)
    n_qubits = intlog2(probs.shape[-2])
    # batch_shape = tf.shape(probs)[-2]
    probs_new = tf.reshape(probs, [-1, *([2]*n_qubits)])
    subsys_2_trace = list(set(range(n_qubits)).difference(set(subsystem)))
    subsys_2_trace_axis = list(map(lambda x:x+1, subsys_2_trace))  # Skip the batch axis
    trace_result = tf.reduce_sum(probs_new, axis=subsys_2_trace_axis)
    return tf.reshape(trace_result, [-1, 2**len(subsystem)])




def fidelity(a: QubitState,
             b: QubitState,
             subsystem: List[int] = None,
             a_subsys_is_pure=False,
             b_subsys_is_pure=False) -> tf.Tensor:
    """
    Fidelity of qubit states. https://en.wikipedia.org/wiki/Fidelity_of_quantum_states#Definition
    :param a:
    :param b:
    :param subsystem:
    :param a_subsys_is_pure:
    :param b_subsys_is_pure:
    :return:
    """
    sqrtm = tf.linalg.sqrtm
    if subsystem is None or (a_subsys_is_pure and b_subsys_is_pure):
        res = tf.abs(inner_product(a, b))**2
    elif a_subsys_is_pure:
        dm_b = density_matrix(b)
        res = inner_product(a, dm_b @ a)
    elif b_subsys_is_pure:
        dm_a = density_matrix(a)
        res = inner_product(b, dm_a @ b)
    else:
        dm_a = density_matrix(a, subsystem)
        dm_b = density_matrix(b, subsystem)
        res = trace(sqrtm(sqrtm(dm_a) @ dm_b @ sqrtm(dm_a))) ** 2
    return tf.cast(res, float_type)


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
    # If control and/or target must take place of target and/or control
    must_swap_control_and_or_target = (control == 1 or target == 0)
    if must_swap_control_and_or_target:
        gate = SWAP(N, 0, 1) @ gate @ SWAP(N, 0, 1)
        # Now c = 1 and t = 0
        if control != 1:
            gate = SWAP(N, 1, control) @ gate @ SWAP(N, 1, control)
        if target != 0:
            gate = SWAP(N, 0, target) @ gate @ SWAP(N, 0, target)
    # Control and target does not mix
    else:
        if control != 0:
            gate = SWAP(N, 0, control) @ gate @ SWAP(N, 0, control)
        if target != 1:
            gate = SWAP(N, 1, target) @ gate @ SWAP(N, 1, target)
    return gate


def gate_expand_toN(U: tf.Tensor, N: int, targets: List[int]):
    if targets is None or len(targets) == 0:
        return U
    assert targets[-1] - targets[0] == len(targets)-1, 'Target qubits must be consecutive'
    tensor_list = []
    add_eye = lambda n: tensor_list.append(tf.eye(2 ** n, dtype=complex_type)) if n != 0 else None
    add_eye(targets[0])
    tensor_list.append(U)
    add_eye(N - 1 - targets[-1])
    return tensor(tensor_list)


def gates_expand_toN(U: Union[List[tf.Tensor], tf.Tensor], N: int, targets: Union[List[List[int]], List[int]]):
    """
    Tensor one or more tensors with eye. padding in-between
    :param U: List of or tensor to tensor
    :param N: Total number of qubits
    :param targets: List of or list of consecutive target qubits
    :return: Final tensor product
    """
    targets = list(map(lambda t: [t] if isinstance(t, int) else t, targets))
    # TODO: Fast. This is slow but correct solution.
    if isinstance(U, list):
        return product([gate_expand_toN(_U, N, _targets) for _U, _targets in zip(U, targets)])
    else:
        return gate_expand_toN(U, N, targets)


def append_zeros(states: tf.Tensor, n: int):
    # idx = tf.meshgrid(tf.range(states.shape[0]))
    # print(idx)
    result = tensor([states] + [s0] * n)
    # idx = tf.where(tf.abs(result) < 1e-4)
    # # result = tf.gather_nd(result, idx)
    # out_shape = tf.cast(result.shape, tf.int64)
    # sparse_data = tf.gather_nd(result, idx)
    # result = tf.SparseTensor(idx, sparse_data, out_shape)
    return result


def intlog2(x: int):
    return x.bit_length() - 1


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

# Extra states
s0000 = tensor([s00, s00])


def make_gate(N: int,
              gate: Union[str, tf.Tensor],
              targets: Union[int, List[int]] = None,
              controls: Union[int, List[int]] = None) -> tf.Tensor:
    if type(gate) is str:
        name = gate.lower()
        if name == 'swap':
            gate = SWAP(N, targets[0], targets[1])
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
        oper = SWAP(N, n, (N-1)-n) @ oper
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
# NOT THE SAME AS ZXZ ROTATION: https://qiskit.org/documentation/stubs/qiskit.extensions.U3Gate.html
# THIS IS WHAT U3 REALLY IS ZYZ-transfrom: https://arxiv.org/pdf/1707.03429.pdf
def U3(t_xyz: Union[tf.Tensor, tf.Variable], *args) -> tf.Tensor:
    t_xyz = tf.cast(t_xyz, complex_type)
    gate = tf.convert_to_tensor([
        [tf.cos(t_xyz[..., 0]/2), -tf.exp(1j * t_xyz[..., 2]) * tf.sin(t_xyz[..., 0]/2)],
        [tf.exp(1j * t_xyz[..., 1]) * tf.sin(t_xyz[..., 0]/2), tf.exp(1j * (t_xyz[..., 1]+t_xyz[..., 2])) * tf.cos(t_xyz[..., 0]/2)]
    ], dtype=complex_type)
    if len(args) == 0:
        return gate
    else:
        return tensor([gate, U3(*args)])

# https://www.researchgate.net/figure/Example-universal-set-of-quantum-gates-consisting-of-three-single-qubit-rotation-gates_fig3_327671865
# NIelsen and Chuang p. 174
def RX(angle):
    angle = tf.cast(angle, complex_type)
    return tf.convert_to_tensor([
        [tf.cos(angle/2), -1j*tf.sin(angle/2)],
        [-1j*tf.sin(angle/2), tf.cos(angle/2)],
    ])

def RY(angle):
    angle = tf.cast(angle, complex_type)
    return tf.convert_to_tensor([
        [tf.cos(angle/2), -tf.sin(angle/2)],
        [tf.sin(angle/2), tf.cos(angle/2)],
    ])

def RZ(angle):
    angle = tf.cast(angle, complex_type)
    return tf.convert_to_tensor([
        [tf.exp(-1j*angle/2), 0],
        [0, tf.exp(1j*angle/2)],
    ])

def RXZX(x1, z, x2):
    return RX(x2) @ RZ(z) @ RX(x1)  # Slow but works (TODO: write explicitly)


def partial_trace_v1(states: tf.Tensor, subsystem: Union[int, List[int]], n_qubits: int):
    """
    Partial trace
    :param states: States in vector or density matrix from to trace
    :param subsystem: Subsystem to trace away
    :return: Remaining states
    """
    if isinstance(subsystem, int):
        subsystem = [subsystem]
    # Convert to density matrices
    if states.shape[-1] == 1:
        states = density_matrix(states)
    n_qubits = intlog2(states.shape[-1])
    # Construct Einstein sum-equation, inspired by:
    # https://github.com/rigetti/quantumflow/blob/bf965f0ca70cd69b387f9ca8407ab38da955e925/quantumflow/qubits.py#L201
    import string
    # EINSTEIN_SUBSCRIPTS = string.ascii_lowercase
    subscripts = list(string.ascii_lowercase)[0:n_qubits*2]
    # Make a copy of the same index n_qubits later to trace out that entry
    for i in subsystem:
        subscripts[n_qubits + i] = subscripts[i]
    subscript_str = 'z' + ''.join(subscripts)
    batch_shape = states.shape[:-2]
    states_reshaped = tf.reshape(states, batch_shape + [2]*2*n_qubits)
    expr = oe.contract_expression(subscript_str, tf.shape(states_reshaped))
    result = expr(states_reshaped, backend='tensorflow')
    result = tf.einsum(subscript_str, states_reshaped)  # FIXME: einsum in tf only supports up to rank 6!
    return tf.reshape(result, batch_shape + [2**n_qubits, 2**n_qubits])


def partial_trace_v2(states: tf.Tensor, subsystem: Union[int, List[int]], n_qubits: int):
    if isinstance(subsystem, int):
        subsystem = [subsystem]
    # Convert to density matrices
    if states.shape[-1] == 1:
        states = density_matrix(states)
    # Flatten the tensor to one batch dimension, and then a series of 2d indexes
    # that represent the states on the from
    # C_n1..._m1... = <n1...|C|m1...>
    states = tf.reshape(states, [-1, *([2] * 2 * n_qubits)])
    # Transpose the the indices of the subsystem we will trace away to the end of the tensor
    subsystem_idx = list(map(lambda i:i+1, subsystem))  # Must account for the batch index at the beginning
    static_indices = list(filter(lambda i: not ((i in subsystem_idx) or (i-n_qubits in subsystem_idx)), range(1, 2*n_qubits+1)))
    subsys_indices = tf.reshape([[i, i+n_qubits] for i in subsystem_idx], [-1])
    perm_indices = tf.concat([[0], static_indices, subsys_indices], axis=-1)
    # Do the transpose
    # with tf.device('cpu'):
    states = tf.transpose(states, perm=perm_indices)
    # Now trace over the len(subsystem_idx) number of pars of dimensions in the states
    for _ in range(len(subsystem_idx)):
        states = tf.linalg.trace(states)
    # Now we have fewer qubits!
    n_qubits_new = n_qubits - len(subsystem_idx)
    return tf.reshape(states, [-1, 2**n_qubits_new, 2**n_qubits_new])


def partial_trace_last(states: tf.Tensor, n_qubits2trace: int, n_qubits: int):
    # Convert to density matrices
    if states.shape[-1] == 1:
        states = density_matrix(states)
    subscripts = 'xyaza'
    # Reshape to the form of subscripts
    n_static = n_qubits - n_qubits2trace
    states = tf.reshape(states, [-1, 2**n_static, 2**n_qubits2trace, 2**n_static, 2**n_qubits2trace])
    expr = oe.contract_expression(subscripts, tf.shape(states))
    return expr(states, backend='tensorflow')


def apply_operators2state(operators: List[tf.Tensor], state: tf.Tensor):
    final_state = state
    for i in range(len(operators)):
        final_state = operators.pop(0) @ final_state
    return final_state
