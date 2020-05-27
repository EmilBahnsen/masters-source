import cirq
from txtutils import ndtotext_print
import numpy as np
import sympy
from tfq_diamond import U
from typing import *

π = np.pi

U_pi = U(π)

iswap_1 = cirq.riswap(3*π/2)  # Apposite sign to sympy!!!
iswap_2 = cirq.riswap(π/2)
print('iSWAP(π/2) [NOT THE ONE WE USE, BUT MIGHT WORK ANYWAY!]')
ndtotext_print(iswap_1._unitary_())
print('iSWAP(3π/2)')
ndtotext_print(iswap_2._unitary_())

def Rn(n):
    return cirq.ZPowGate()**(2**(1-n))

qc_test = cirq.Circuit()
q = cirq.GridQubit(1,1)
qc_test.append(Rn(5).on(q))
print(qc_test)
print(qc_test.unitary())
print('Should be:', np.exp(2*π*1j/2**5))
assert np.isclose(qc_test.unitary()[1,1], np.exp(2*π*1j/2**5))

n_qft = 5
A = [cirq.GridQubit(0, i) for i in range(n_qft)]
B = [cirq.GridQubit(1, i) for i in range(n_qft)]
'''
A1 - B1
|    |
A2 - B2
|    |
A3 - B3
.
.
. (But turned on its side...)
'''
qc = cirq.Circuit()

# Input data
input_values = [1,1,1,1,1]
for i, value in enumerate(input_values):
    if value == 1:
        qc.append(cirq.X(A[i]))
    else:
        qc.append(cirq.I(A[i]))

sim = cirq.Simulator()
def sim_print(qc):
    res = sim.simulate(qc)
    print(res)

# Alg.
def post_qft_cross_swap(qc, qubits):
    n_qubits = len(qubits)
    for i in range(n_qubits//2):
        qc.append(cirq.SWAP(qubits[i], qubits[(n_qubits-1)-i]))

def CRn(qc, n, *, ctrl, targ, ctrl_opp, targ_opp):
    qc.append(cirq.X(ctrl_opp))  # Must have |1> opposite to ctrl
    qc.append(U_pi.on(targ, targ_opp, ctrl, ctrl_opp))
    qc.append(Rn(n).on(targ_opp))
    qc.append(U_pi.on(targ, targ_opp, ctrl, ctrl_opp))
    qc.append(cirq.X(ctrl_opp))  # ... rm that again

def dQFT(qc: cirq.Circuit, A: List[cirq.GridQubit], B: List[cirq.GridQubit]):
    # For every input qubit...
    for i in range(len(A)):
        print('i', i)
        # Make Rn on the qubits above, starting from this qubit up
        n = 2
        for j in reversed(range(i)):
            print('j', j)
            CRn(qc, n, ctrl=A[j+1], targ=A[j], ctrl_opp=B[j], targ_opp=B[j+1])
            n += 1
            if j != 0:  # Don't need to iswap with the first input qubit
                qc.append(iswap_1.on(A[j+1], A[j]))
        # Make the swap back of A[i] to its correct place (through the qubits it did in the first place)
        for j in range(1,i):
            qc.append(iswap_2.on(A[j], A[j+1]))
        # Do the H on this qubit
        qc.append(cirq.H.on(A[i]))
    post_qft_cross_swap(qc, A)

# Do QFT
dQFT(qc, A, B)

sim_print(qc)

print(qc.to_text_diagram(qubit_order=A+B))

ndtotext_print(qc.unitary(qubit_order=B+A)[:2**n_qft, :2**n_qft])

qc.append(cirq.QFT(*A)**(-1))
print('This should be the input encoding only!')
ndtotext_print(qc.unitary(qubit_order=B+A)[:2**n_qft, :2**n_qft])