import pennylane as qml
from pennylane import numpy as np
from pennylane.variable import Variable
import numpy as rnp
import matplotlib
import matplotlib.pyplot as plt

π = np.pi

# Unnessesary sympy-calculation of U-matrix (pennylane numpy wrapper has np.kron!!!)
# def calculate_U():
#     """
#     Calculate sympy representation of U matrix
#     :return:
#     """
#     from sympy import Matrix, exp, Symbol, sqrt, kronecker_product, pprint, latex
#     # from diamond import s00,s11,sp,sm
#     # print(s00.__array__())
#     # print(s11.__array__())
#     # print(sp.__array__())
#     # print(sm.__array__())
#
#     s00 = Matrix([[1, 0, 0, 0]])
#     s11 = Matrix([[0, 0, 0, 1]])
#     sp = Matrix([[0, 1, 1, 0]]) / sqrt(2)
#     sm = Matrix([[0, 1, -1, 0]]) / sqrt(2)
#
#     t = Symbol('t')
#
#     U00 = Matrix([
#         [1, 0, 0, 0],
#         [0, (exp(-1j*t) + 1)/2, (exp(-1j*t) - 1)/2, 0],
#         [0, (exp(-1j*t) - 1)/2, (exp(-1j*t) + 1)/2, 0],
#         [0, 0, 0, exp(-1j*t)],
#     ])
#
#     U11 = Matrix([
#         [exp(1j * t), 0, 0, 0],
#         [0, (exp(1j * t) + 1) / 2, (exp(1j * t) - 1) / 2, 0],
#         [0, (exp(1j * t) - 1) / 2, (exp(1j * t) + 1) / 2, 0],
#         [0, 0, 0, 1],
#     ])
#
#     Up = Matrix([
#         [exp(1j * t), 0, 0, 0],
#         [0, 1, 0, 0],
#         [0, 0, 1, 0],
#         [0, 0, 0, exp(-1j * t)],
#     ])
#
#     Um = Matrix([
#         [1, 0, 0, 0],
#         [0, 1, 0, 0],
#         [0, 0, 1, 0],
#         [0, 0, 0, 1],
#     ])
#
#     U = kronecker_product((s00.transpose() * s00), U00)\
#         + kronecker_product((s11.transpose() * s11), U11)\
#         + kronecker_product((sp.transpose() * sp), Up)\
#         + kronecker_product((sm.transpose() * sm), Um)
#
#     pprint(U)
#     print(latex(U))


# class Gates:
#     @staticmethod
#     def ControlledHadamard(wires: [int]):
#         matrix = np.array([
#                 [1, 0, 0, 0],
#                 [0, 1, 0, 0],
#                 [0, 0, 1/m.sqrt(2), 1/m.sqrt(2)],
#                 [0, 0, 1/m.sqrt(2), -1/m.sqrt(2)]
#         ])
#         qml.QubitUnitary(
#             matrix,
#             wires=wires
#         )
#
#
# def oper_A(wires):
#     qml.CNOT(wires=[wires[1], wires[0]])
#     Gates.ControlledHadamard(wires=wires)
#     qml.CNOT(wires=[wires[1], wires[0]])
#
#
# def oper_B(wires):
#     qml.CZ(wires=wires)
#     qml.SWAP(wires=wires)
#     qml.PauliZ(wires=wires[0])
#     qml.PauliZ(wires=wires[1])
#
#
# def oper_C(z, wires):
#     qml.Hadamard(wires=wires[2])
#     qml.Toffoli(wires=wires)
#     qml.Hadamard(wires=wires[2])
#     qml.CSWAP(wires=wires)
#     qml.CZ(wires=[wires[0], wires[1]])
#     qml.CZ(wires=[wires[0], wires[2]])
#     qml.RZ(z/2, wires=wires[0])
#
#
# def oper_D(z, wires):
#     qml.Hadamard(wires=wires[3])
#     qml.Toffoli(wires=[wires[0], wires[1], wires[3]])
#     qml.Hadamard(wires=wires[3])
#     qml.CSWAP(wires=[wires[0], wires[1], wires[3]])
#     qml.PauliZ(wires=wires[0])
#     qml.RZ(-z/2, wires=wires[0])
#
#
# def U(z, wires):
#     oper_A(wires=[wires[0], wires[1]])
#     oper_B(wires=[wires[2], wires[3]])
#     oper_C(z, wires=[wires[1], wires[2], wires[3]])
#     oper_D(z, wires=wires)
#     oper_A(wires=[wires[0], wires[1]])


# Construction of U
# Basis states
s0 = np.asanyarray([[1, 0]])
s1 = np.asanyarray([[0, 1]])
s00 = np.kron(s0, s0)
s01 = np.kron(s0, s1)
s10 = np.kron(s1, s0)
s11 = np.kron(s1, s1)

# Bell states
sp = (s01 + s10)/np.sqrt(2)      # (|01> + |10>)/√2
sm = (s01 - s10)/np.sqrt(2)      # (|01> - |10>)/√2

o0000 = s00.transpose() * s00
o1111 = s11.transpose() * s11
opp = sp.transpose() * sp
omm = sm.transpose() * sm


def U0_(t, wires):
    if type(t) is Variable:
        t = t.val
    U00 = np.array([
        [1,0,0,0],
        [0, (np.exp(-1j*t)+1)/2, (np.expm1(-1j*t))/2, 0],
        [0, np.expm1(-1j*t)/2, (np.exp(-1j*t)+1)/2, 0],
        [0,0,0,np.exp(-1j*t)]
    ])
    U11 = np.array([
        [np.exp(1j * t), 0, 0, 0],
        [0, (np.exp(1j * t) + 1) / 2, np.expm1(1j * t) / 2, 0],
        [0, np.expm1(1j * t) / 2, (np.exp(1j * t) + 1) / 2, 0],
        [0, 0, 0, 1]
    ])
    Up = np.array([
        [np.exp(1j * t), 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, np.exp(-1j * t)]
    ])
    Um = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    matrix = np.kron(o0000, U00) + np.kron(o1111, U11) + np.kron(opp, Up) + np.kron(omm, Um)
    # if type(matrix) is not np.ndarray:
    #     matrix: np.numpy_boxes.ArrayBox = matrix
    #     qml.QubitUnitary(matrix._value, wires=wires)
    # else:
    #     qml.QubitUnitary(matrix, wires=wires)
    qml.QubitUnitary(matrix, wires=wires)


def U0_matrix(t):
    if type(t) is Variable:
        t = t.val
    U00 = np.array([
        [1, 0, 0, 0],
        [0, (np.exp(-1j * t) + 1) / 2, (np.expm1(-1j * t)) / 2, 0],
        [0, np.expm1(-1j * t) / 2, (np.exp(-1j * t) + 1) / 2, 0],
        [0, 0, 0, np.exp(-1j * t)]
    ])
    U11 = np.array([
        [np.exp(1j * t), 0, 0, 0],
        [0, (np.exp(1j * t) + 1) / 2, np.expm1(1j * t) / 2, 0],
        [0, np.expm1(1j * t) / 2, (np.exp(1j * t) + 1) / 2, 0],
        [0, 0, 0, 1]
    ])
    Up = np.array([
        [np.exp(1j * t), 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, np.exp(-1j * t)]
    ])
    Um = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return np.kron(o0000, U00) + np.kron(o1111, U11) + np.kron(opp, Up) + np.kron(omm, Um)


def U0(t, wires):
    matrix = U0_matrix(t)
    qml.QubitUnitary(matrix, wires=wires)


n_qubits = 4
n_shots = 10 ** 6

device = qml.device("default.qubit", wires=n_qubits, shots=n_shots)


# all_states =

def prepare_state(input):
    qml.QubitStateVector(np.asanyarray(input), wires=[0, 1, 2, 3])


#@qml.qnode(device)
def qft_U(params):
    # Execute QFT
    n_param = 0
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
        for j in range(i+1, n_qubits):
            U0(params[n_param], wires=[0, 1, 2, 3])
            n_param += 1
    # Final swaps
    for n in range(n_qubits//2):
        qml.SWAP(wires=[n, n_qubits-n-1])


#@qml.qnode(device)
def qft():
    # Execute QFT
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
        for n_j,j in enumerate(range(i+1, n_qubits)):
            R_k = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, np.exp(2j*π/(2**(n_j+2)))]])
            qml.QubitUnitary(R_k, wires=[j,i])
    # Final swaps
    for n in range(n_qubits//2):
        qml.SWAP(wires=[n, (n_qubits-1)-n])


#@qml.qnode(device)
def iqft():
    # Final swaps (first)
    for n in range(n_qubits//2):
        qml.SWAP(wires=[n, (n_qubits-1)-n])
    # Execute inverse QFT
    for i in reversed(range(n_qubits)):
        for n_j,j in enumerate(range(i+1, n_qubits)):
            R_k = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, np.exp(-2j*π/(2**(n_j+2)))]])
            qml.QubitUnitary(R_k, wires=[j,i])
        qml.Hadamard(wires=i)


def loss(input, params):
    return np.sum(np.linalg.norm(qft_U(input=input, params=params) - np.fft.fft(input, norm='ortho'))**2)


def normalize(x):
    return x / np.sqrt(np.sum(np.linalg.norm(x)**2))


x = np.linspace(0, 2*π, 16)
input = normalize(np.sin(x))
print(input)
params = [π/2]*6
print('input', input)

# ifft = np.fft.ifft(input, norm='ortho')
# print('ifft', ifft)
# print(loss(input, params=params))
# plt.hist(np.random.randn(100))
# plt.show()

# print('qft_U', qft_U(input=input, params=params))
# print('qft', qft(input=input))


# --- Sanity check ---
# @qml.qnode(device)
# def qft_inv_test(input):
#     prepare_state(input)
#     qft()
#     iqft()
#
#     basis_obs = qml.Hermitian(np.diag(range(2 ** n_qubits)), wires=range(n_qubits))
#     return qml.sample(basis_obs)
#
# def test_qft(input):
#     samples = qft_inv_test(input=input).astype(int)
#     q_probs = np.bincount(samples, minlength=2 ** n_qubits) / n_shots
#     return q_probs
#
I16 = np.identity(2**4)
# for i in range(2**4):
#     q_probs = test_qft(I16[i])
#     print(q_probs)

# --- Optimize: state -> QFT (using U) -> QFT (traditional) -> state
def eigenvalues2basisvector(eigenvalues):
    n_basis_vector = sum([2**n if i==-1 else 0 for n,i in enumerate(reversed(eigenvalues))])
    return [*([0]*n_basis_vector), 1, *([0]*(2**n_qubits - n_basis_vector - 1))]

def basisvector2eigenvalues(basis_vector):
    n_basis_vector = -1
    for n_basis_vector, value in enumerate(basis_vector):
        if value == 1:
            break
    eigenvalues = [1] * n_qubits
    for i in reversed(range(n_qubits)):
        if n_basis_vector - 2**i >= 0:
            eigenvalues[i] = -1
            n_basis_vector -= 2**i
            if n_basis_vector == 0:
                break
    return list(reversed(eigenvalues))


basis_states = np.diag(range(2**n_qubits))
@qml.qnode(device)
def qft_U_iqft(thetas, input_state):
    # loss = 0
    # for i, basis_state in enumerate(basis_states):
    # i = 1
    prepare_state(input_state)
    # qft_U(thetas)
    # qft()
    qml.RZ(thetas[0], wires=0)
    iqft()
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


def qft_U_loss(thetas):
    print(thetas[0].val if type(thetas[0]) is Variable else thetas[0])
    loss = 0
    for i in range(2**n_qubits):
        input_state_vector = I16[i]
        exp_output_eigenvalues = basisvector2eigenvalues(input_state_vector)
        output_eigenvalues = qft_U_iqft(thetas, input_state=input_state_vector)
        # Calculate loss as the sum of the squared diff. btw. eigenvalues of expected output and actual (PauliZ)
        loss += np.sum((np.array(exp_output_eigenvalues) - np.array(output_eigenvalues))**2)
    return loss


# Actual optimazation
opt = qml.AdagradOptimizer()

# Init parameters in the U-gates
rng_seed = 0
np.random.seed(rng_seed)
# thetas = 2*π * np.random.randn(6)
thetas = np.array([π, π, π, π, π, π])

# Test run the loss function
print('Loss test:', qft_U_loss(thetas), qft_U_loss(thetas*0.2))

# Perform optimazation
cost_history = []
steps = 10
for it in range(steps):
    thetas = opt.step(qft_U_loss, thetas)
    _cost = qft_U_loss(thetas)
    print("Step {:3d}       Cost = {:9.7f}, thetas = {}".format(it, _cost, thetas))
    cost_history.append(_cost)
