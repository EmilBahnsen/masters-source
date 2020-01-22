from qiskit import *
from qiskit.extensions import *
import numpy as np
import math

qr = QuantumRegister(2)
qc = QuantumCircuit(qr)
qc.cu1(math.pi, qr[0], qr[1])
print(qc.decompose().decompose().draw())

# H(N,3) * X(N,1) * X(N,0)
qr = QuantumRegister(4)
qc = QuantumCircuit(qr)
qc.x(qr[0])
qc.x(qr[1])
qc.h(qr[3])
print(qc.decompose().decompose().draw())

# X(N,1) * X(N,0) * H(N,2)
qr = QuantumRegister(4)
qc = QuantumCircuit(qr)
qc.h(qr[2])
qc.x(qr[0])
qc.x(qr[1])
print(qc.decompose().decompose().draw())

# https://arxiv.org/pdf/1807.01703.pdf
# u1(tz), u2(ty, tz), u3(tx, ty, tz)


print('--- U as sequence of gates ---')
from qiskit.extensions import *

qr = QuantumRegister(4)
qc = QuantumCircuit(qr)


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


matrix = U0_matrix(0)
qc.append(UnitaryGate(matrix), qargs=[0,1,2,3])
print(qc.decompose().decompose().draw())

qr = QuantumRegister(2)
qc = QuantumCircuit(qr)
qc.append(UnitaryGate([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,np.exp(1j*np.pi/2)]
]), qargs=[0,1])
print(qc.decompose().draw())

qr = QuantumRegister(2)
qc = QuantumCircuit(qr)
qc.append(UnitaryGate([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,np.exp(1j*np.pi/4)]
]), qargs=[0,1])
print(qc.decompose().draw())