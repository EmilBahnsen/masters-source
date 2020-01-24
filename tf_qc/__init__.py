import tensorflow as tf
from tensorflow import linalg as la
from tensorflow.keras import layers
import math
from typing import *
import qutip as qt

tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.set_soft_device_placement(True)

float_type = tf.float64
complex_type = tf.complex128

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

    # tf.print('t')
    # print_complex(t)
    # tf.print('tf.exp(-it)')
    # print_complex(tf.exp(-it))
    # tf.print('tf.exp(-it) + 1')
    # print_complex(tf.exp(-it) + 1)
    # tf.print('tf.exp(-it) - 1')
    # print_complex(tf.exp(-it) - 1)
    # tf.print('tf.exp(-it) + -1')
    # print_complex(tf.exp(-it) + -1)
    # print_matrix(U00)
    # print_unitary_test(U00)

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


def qft_U(N: int, oper: tf.Tensor, params: tf.Variable) -> tf.Tensor:
    # # Execute QFT
    # tf.print('\n')
    # print_non_zero_unitary_test(oper)
    assert N == 4, 'N != 4 not implemented!'
    n_param = 0
    for i in range(N):
        hadamard_gate = make_gate(N, 'h', i)
        oper = hadamard_gate @ oper
        # tf.print('h', i)
        # print_non_zero_unitary_test(oper)
        for j in range(i + 1, N):
            U_gate = U(params[n_param])
            oper = U_gate @ oper
            # tf.print('U', j, tf.cast(params[n_param], float_type))
            # print_non_zero_unitary_test(oper)
            # print_non_zero_unitary_test(U_gate)
            # print_unitary_test(oper)
            # print_unitary_test(U_gate)
            n_param += 1
    assert n_param == params.shape[0]
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

def print_matrix(matrix):
    tf.print(tf.math.real(matrix), summarize=100)

def print_non_zero_matrix(matrix):
    print_matrix(tf.greater(tf.math.real(matrix), 1e-5))

def unitary_test(matrix):
    return tf.transpose(matrix, conjugate=True) @ matrix

def print_unitary_test(matrix):
    tf.print(tf.math.real(unitary_test(matrix)), summarize=100)

def print_non_zero_unitary_test(matrix):
    print_non_zero_matrix(unitary_test(matrix))

def print_complex(*args):
    string = tf.as_string(tf.math.real(args[0])) + ' ' + tf.as_string(tf.math.imag(args[0])) + 'j'
    if len(args) > 1:
        tf.print(string, print_complex(*args[1:]))
    else:
        tf.print(string)


# Keral Layers and Models
class QFTULayer(layers.Layer):
    def __init__(self, n_qubits=4):
        super(QFTULayer, self).__init__()
        self.n_qubits = None

    def build(self, input_shape):
        self.n_qubits = input_shape[-2].bit_length() - 1
        theta_init = tf.random_uniform_initializer(0, 2*π)
        n_thetas = sum(range(self.n_qubits))
        self.thetas = tf.Variable(initial_value=theta_init(shape=(n_thetas,), dtype=float_type),
                                 trainable=True, name='thetas')

    def call(self, inputs, thetas=None):
        self.qft_U = self.matrix(thetas)
        return tf.matmul(self.qft_U, inputs)

    def matrix(self, thetas=None):
        if thetas:
            return qft_U(self.n_qubits, I4, thetas)
        else:
            return qft_U(self.n_qubits, I4, self.thetas)


# TEST
# x = tf.ones((10000000, 2**4, 1), dtype=complex_type)
# qft_u_layer = QFTULayer()
# t = time.process_time()
# qft_u_layer(x)
# exit()


class U3Layer(layers.Layer):
    def __init__(self):
        super(U3Layer, self).__init__()
        self.U3 = None
        self.thetas = None

    def build(self, input_shape):
        theta_init = tf.random_uniform_initializer(0, 2 * π)
        n_qubits = input_shape[-2].bit_length() - 1
        self.thetas = [tf.Variable(initial_value=theta_init(shape=(3,), dtype=float_type),
                                   trainable=True,
                                   dtype=float_type) for _ in range(n_qubits)]

    def call(self, inputs, thetas=None):
        self.U3 = self.matrix(thetas)
        return self.U3 @ inputs

    def matrix(self, thetas=None):
        if thetas:
            return U3(*thetas)
        else:
            return U3(*self.thetas)


class IQFTLayer(layers.Layer):
    def __init__(self):
        super(IQFTLayer, self).__init__()
        self.n_qubits = None

    def build(self, input_shape):
        self.n_qubits = input_shape[-2].bit_length() - 1

    def call(self, inputs, **kwargs):
        self.IQFT = iqft(self.n_qubits, I4)
        return tf.matmul(self.IQFT, inputs)

    def matrix(self):
        return iqft(self.n_qubits, I4)


# Models
class PrePostQFTUIQFT(tf.keras.Model):
    def __init__(self, nn=False):
        super(PrePostQFTUIQFT, self).__init__()
        if nn:
            tf.keras.layers.Dense()
        self.U3_in = U3Layer()
        self.QFT_U = QFTULayer()
        self.U3_out = U3Layer()
        self.IQFT = IQFTLayer()

    def call(self, inputs, training=None, mask=None):
        x = self.U3_in(inputs)
        x = self.QFT_U(x)
        x = self.U3_out(x)
        x = self.IQFT(x)
        return x

    def matrix(self):
        return self.IQFT.matrix() @ self.U3_out.matrix() @ self.QFT_U.matrix() @ self.U3_in.matrix()


# class PrePostQFTUIQFT(tf.keras.Model):
#     pass

# Losses
class MeanNorm(tf.losses.Loss):
    def call(self, y_true, y_pred):
        # y_pred = tf.convert_to_tensor(y_pred)
        diff = y_true - y_pred
        norms = tf.cast(tf.norm(diff, axis=[-2, -1]), dtype=float_type)
        mean_norm = tf.reduce_mean(norms)
        return mean_norm


class Mean1mFidelity(tf.losses.Loss):
    def call(self, y_true, y_pred):
        norm_squares = tf.transpose(y_true, perm=[0,2,1]) @ y_pred
        fidelies = tf.square(tf.abs(norm_squares))
        meanFilelity = tf.reduce_mean(fidelies)
        return 1 - meanFilelity
# TEST
x = tf.constant([1/math.sqrt(3), 1/math.sqrt(3), 1/math.sqrt(3), 1/math.sqrt(2), 1/math.sqrt(2), 0], shape=(2,3,1), dtype=complex_type)
y = tf.constant([1, 0, 0, 1, 0, 0], shape=(2,3,1), dtype=complex_type)
assert round(Mean1mFidelity()(x, y).numpy(), 5) == round(1 - (1/3 + 1/2)/2, 5)
# TEST END


# Utils
def random_unifrom_complex(shape: Any,
                           minval: int = 0,
                           maxval: Any = None,
                           dtype: tf.dtypes.DType = tf.dtypes.float64,
                           seed: Any = None,
                           name: Any = None):
    return tf.complex(tf.random.uniform(shape, minval, maxval, dtype, seed),
                      tf.random.uniform(shape, minval, maxval, dtype, seed),
                      name=name)

def normalize_state_vectors(state_vectors: tf.Tensor):
    '''
    Normalize state vectors
    :param state_vectors with shape (n_vectors, length, 1)
    :return:
    '''
    root_sum_norm_squares = tf.sqrt(tf.reduce_sum(tf.square(tf.abs(state_vectors)), axis=1, keepdims=True))
    return tf.math.xdivy(state_vectors, tf.cast(root_sum_norm_squares, state_vectors.dtype))

# TESTS
# TODO: make unitary testes on all new matrices
