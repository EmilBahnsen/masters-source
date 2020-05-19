import tensorflow_quantum as tfq
import cirq
from tfq_diamond import U
import sympy as sp
import numpy as np
from txtutils import ndtotext_print

π = np.pi

# t = sp.Symbol('t', real=True)
t = 1  # Just to test equality



def A2B_basis(qc, q1, q2):
    qc.append(cirq.XX.on(q1, q2))
    qc.append(cirq.CNOT.on(q2, q1))
    qc.append(cirq.H.on(q2).controlled_by(q1))
    qc.append(cirq.CNOT.on(q1, q2))
    qc.append(cirq.X.on(q2))
    return qc

def B2A_basis(qc, q1, q2):
    qc.append(cirq.X.on(q2))
    qc.append(cirq.CNOT.on(q1, q2))
    qc.append(cirq.H.on(q2).controlled_by(q1))
    qc.append(cirq.CNOT.on(q2, q1))
    qc.append(cirq.XX.on(q1, q2))
    return qc

def X_on_all(circuit, qubits):
    for q in qubits:
        circuit.append(cirq.X.on(q))
    return circuit

qc = cirq.Circuit()
q1, q2 = cirq.LineQubit.range(2)
qc = A2B_basis(qc, q1, q2)
qc = B2A_basis(qc, q1, q2)
print(qc)
ndtotext_print(qc.unitary())
# exit()

# class ISWAP(cirq.ISwapPowGate):
#     def __init__(self, target):
#         super(ISWAP, self).__init__()

qubits = cirq.GridQubit.rect(2, 2)
C1, C2, T1, T2 = qubits
qc = cirq.Circuit()

# qc = B2A_basis(qc, C1, C2)  # Change output matrix to B-basis (easier to check for mistakes)
# qc = B2A_basis(qc, T1, T2)
# --- Very first one ---
qc.append(cirq.XX(T1, T2))
qc.append(cirq.ZPowGate().on(T2).controlled_by(C1, C2, T1)**(t/π)) # NOTICE: Cancel out π
qc.append(cirq.XX(T1, T2))

qc.append(cirq.XX(C1, C2))
qc.append(cirq.ZPowGate().on(C1).controlled_by(C2, T1, T2)**(-t/π))
qc.append(cirq.XX(C1, C2))
# --- Very first one end ---
# --- Second two ---
A2B_basis(qc, C1, C2)  # This is to do the oper |11> <-> |+>, |00> <-> |->
A2B_basis(qc, T1, T2)
X_on_all(qc, qubits)
B2A_basis(qc, C1, C2)
B2A_basis(qc, T1, T2)
# CCiSWAP(t) then CCiSWAP(-t)
qc.append(cirq.ISwapPowGate().on(T1, T2).controlled_by(C1, C2)**(-2*t/π))  # NOTICE: Swapped signs and cancel out π/2
qc.append(cirq.ISwapPowGate().on(C1, C2).controlled_by(T1, T2)**(2*t/π))

A2B_basis(qc, C1, C2)  # This is to do the oper |11> <-> |+>, |00> <-> |-> (i.e. go back)
A2B_basis(qc, T1, T2)
X_on_all(qc, qubits)
B2A_basis(qc, C1, C2)
B2A_basis(qc, T1, T2)
# --- Second two end ---

# qc = A2B_basis(qc, C1, C2)  # Change output matrix to B-basis (i.e. change back)
# qc = A2B_basis(qc, T1, T2)

print(qc)
# ndtotext_print(qc.unitary())

qc_true = cirq.Circuit()
# qc2 = B2A_basis(qc2, C1, C2)
# qc2 = B2A_basis(qc2, T1, T2)
qc_true.append(U(t).on(*qubits))
# qc2 = A2B_basis(qc2, C1, C2)
# qc2 = A2B_basis(qc2, T1, T2)
# ndtotext_print(qc2.unitary())

same = (qc.unitary() - qc_true.unitary() < 1e-5).all()
print(f'Check if their unitary is the same: {same}')
assert same

# TODO: Simplify this circuit by hand and show that it's the same
# Simplification of transform circuit
print('Simplification!')
qc_simple = cirq.Circuit()
def A2B_XX_B2A(qc, q1, q2):
    qc.append(cirq.X(q2))
    qc.append(cirq.X.on(q1).controlled_by(q2))
    qc.append(cirq.X(q1))
    qc.append(cirq.H.on(q2))
    qc.append(cirq.X.on(q1).controlled_by(q2))
    qc.append(cirq.X(q2))

def A2B_XX_B2A_swapped(qc, q1, q2):
    A2B_XX_B2A(qc, q2, q1)

def A2B_XX_B2A_after_CCCZ(qc, q1, q2):
    qc.append(cirq.X.on(q1).controlled_by(q2))
    qc.append(cirq.H.on(q2))
    qc.append(cirq.X.on(q1).controlled_by(q2))
    qc.append(cirq.X(q2))

def A2B_XX_B2A_after_CCCZ_swapped(qc, q1, q2):
    A2B_XX_B2A_after_CCCZ(qc, q2, q1)

# --- First one ---
qc_simple.append(cirq.X(T1))
qc_simple.append(cirq.X(T2))
qc_simple.append(cirq.ZPowGate().on(T2).controlled_by(C1, C2, T1)**(t/π)) # NOTICE: Cancel out π
qc_simple.append(cirq.X(T1))
qc_simple.append(cirq.X(T2))

qc_simple.append(cirq.X(C1))
qc_simple.append(cirq.X(C2))
qc_simple.append(cirq.ZPowGate().on(C1).controlled_by(C2, T1, T2)**(-t/π))
# --- First one end ---
# --- Last two ---
A2B_XX_B2A_after_CCCZ(qc_simple, C1, C2)  # These shall be called [XX]_B
A2B_XX_B2A(qc_simple, T1, T2)
qc_simple.append(cirq.ISwapPowGate().on(T1, T2).controlled_by(C1, C2)**(-2*t/π))  # NOTICE: Swapped signs and cancel out π/2
qc_simple.append(cirq.ISwapPowGate().on(C1, C2).controlled_by(T1, T2)**(2*t/π))
A2B_XX_B2A(qc_simple, C1, C2)
A2B_XX_B2A(qc_simple, T1, T2)
# --- Last two end ---
print(qc_simple)
# ndtotext_print(qc_simple.unitary())

equal = (qc_simple.unitary() - qc_true.unitary() < 1e-5).all()
print(f'Simple circuit equal U: {equal}')
assert equal
# TODO: print to latex
# latex = cirq.contrib.qcircuit.circuit_to_latex_using_qcircuit(qc_simple)
# print(latex)


# --- Alternative with SWAP instead of CZ ---
qc2 = cirq.Circuit()
# --- All 3 ---
A2B_XX_B2A(qc2, T1, T2)  # This is to do the oper |11> <-> |+>, |00> <-> |->
# CCSWAP(t/2)
qc2.append(cirq.SwapPowGate().on(T1, T2).controlled_by(C1, C2)**(t/π))

A2B_XX_B2A(qc2, C1, C2)

qc2.append(cirq.ISwapPowGate().on(T1, T2).controlled_by(C1, C2)**(-2*t/π))  # NOTICE: Swapped signs and cancel out π/2
qc2.append(cirq.ISwapPowGate().on(C1, C2).controlled_by(T1, T2)**(2*t/π))

A2B_XX_B2A(qc2, T1, T2)
# CCSWAP(-t/2)
qc2.append(cirq.SwapPowGate().on(C1, C2).controlled_by(T1, T2)**(-t/π))

A2B_XX_B2A(qc2, C1, C2)  # This is to do the oper |11> <-> |+>, |00> <-> |-> (i.e. go back)
# --- All 3 end ---
print(qc2)
equal = (qc2.unitary() - qc_true.unitary() < 1e-5).all()
print(f'New circuit equal U: {equal}')
assert equal
print('This is somewhat more simple:')
print('"XX"_T -> CCSWAP -> "XX"_C -> CCiSWAP -> CCiSWAP -> "XX"_T -> CCSWAP -> "XX"_C')

# --- Alternative with SWAP instead of CZ ---
print('Same swapped C1 <-> C2 and T1 <-> T2')
qc2 = cirq.Circuit()
# --- All 3 ---
A2B_XX_B2A_swapped(qc2, T1, T2)  # This is to do the oper |11> <-> |+>, |00> <-> |->
# CCSWAP(t/2)
qc2.append(cirq.SwapPowGate().on(T1, T2).controlled_by(C1, C2)**(t/π))

A2B_XX_B2A_swapped(qc2, C1, C2)

qc2.append(cirq.ISwapPowGate().on(T1, T2).controlled_by(C1, C2)**(-2*t/π))  # NOTICE: Swapped signs and cancel out π/2
qc2.append(cirq.ISwapPowGate().on(C1, C2).controlled_by(T1, T2)**(2*t/π))

A2B_XX_B2A_swapped(qc2, T1, T2)
# CCSWAP(-t/2)
qc2.append(cirq.SwapPowGate().on(C1, C2).controlled_by(T1, T2)**(-t/π))

A2B_XX_B2A_swapped(qc2, C1, C2)  # This is to do the oper |11> <-> |+>, |00> <-> |-> (i.e. go back)
# --- All 3 end ---
print(qc2)
equal = (qc2.unitary() - qc_true.unitary() < 1e-5).all()
print(f'New swapped circuit equal U: {equal}')
assert equal

# --- Alternative with SWAP instead of CZ ---
print('Same swapped C1 <-> C2')
qc2 = cirq.Circuit()
# --- All 3 ---
A2B_XX_B2A(qc2, T1, T2)  # This is to do the oper |11> <-> |+>, |00> <-> |->
# CCSWAP(t/2)
qc2.append(cirq.SwapPowGate().on(T1, T2).controlled_by(C1, C2)**(t/π))

A2B_XX_B2A_swapped(qc2, C1, C2)

qc2.append(cirq.ISwapPowGate().on(T1, T2).controlled_by(C1, C2)**(-2*t/π))  # NOTICE: Swapped signs and cancel out π/2
qc2.append(cirq.ISwapPowGate().on(C1, C2).controlled_by(T1, T2)**(2*t/π))

A2B_XX_B2A(qc2, T1, T2)
# CCSWAP(-t/2)
qc2.append(cirq.SwapPowGate().on(C1, C2).controlled_by(T1, T2)**(-t/π))

A2B_XX_B2A_swapped(qc2, C1, C2)  # This is to do the oper |11> <-> |+>, |00> <-> |-> (i.e. go back)
# --- All 3 end ---
print(qc2)
equal = (qc2.unitary() - qc_true.unitary() < 1e-5).all()
print(f'New swapped circuit equal U: {equal}')
assert equal

print('--- Check that the symmetries work ---')
qc_x4_U_x4 = cirq.Circuit()
X_on_all(qc_x4_U_x4, qubits)
qc_x4_U_x4.append(U(t).on(*qubits))
X_on_all(qc_x4_U_x4, qubits)

equal = (qc_x4_U_x4.unitary() - qc_true.unitary() < 1e-5).all()
print(f'x4 U x4 == U: {equal}')
assert equal
