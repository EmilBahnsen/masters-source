import numpy as np
import quantum_decomp as qd
import cirq
import qiskit as qk

np.set_printoptions(precision=3)
π = np.pi

U00 = lambda t: np.array([
    [1, 0, 0, 0],
    [0, (np.exp(-1j*t)+1)/2, np.expm1(-1j*t)/2, 0],
    [0, np.expm1(-1j*t)/2, (np.exp(-1j*t)+1)/2, 0],
    [0, 0, 0, np.exp(-1j*t)]
])


def print_gate(t):
    print('t =', t)
    matrix = U00(t)
    gates = qd.matrix_to_gates(matrix)
    circuit = qd.matrix_to_cirq_circuit(matrix)
    print(circuit)
    print()

# for t in np.arange(0, π, π/16):
#     print_gate(t)


def U0_matrix(t):
    # Construction of U
    # Basis states
    s0 = np.asanyarray([[1, 0]])
    s1 = np.asanyarray([[0, 1]])
    s00 = np.kron(s0, s0)
    s01 = np.kron(s0, s1)
    s10 = np.kron(s1, s0)
    s11 = np.kron(s1, s1)

    # Bell states
    sp = (s01 + s10) / np.sqrt(2)  # (|01> + |10>)/√2
    sm = (s01 - s10) / np.sqrt(2)  # (|01> - |10>)/√2

    o0000 = s00.transpose() * s00
    o1111 = s11.transpose() * s11
    opp = sp.transpose() * sp
    omm = sm.transpose() * sm

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


# for i,t in enumerate(np.arange(0, 2*π+0.1, π/8)):
#     print('t =', str(i) + 'π/8')
#     print(qd.matrix_to_cirq_circuit(U0_matrix(t), optimize=True))

# for matrix in qd.two_level_decompose(U0_matrix(π/8)):
#     print(matrix)

def keep_fuction(oper: cirq.Operation) -> bool:
    print(oper, oper.gate.num_qubits())
    return oper.gate.num_qubits() <= 2

print(qd.matrix_to_cirq_circuit(U0_matrix(π/8)))
decomp_circuit = cirq.decompose(qd.matrix_to_cirq_circuit(U0_matrix(π/8)), keep=keep_fuction)
print(cirq.Circuit(decomp_circuit))
