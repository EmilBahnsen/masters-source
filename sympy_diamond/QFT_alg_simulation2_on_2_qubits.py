import cirq
from txtutils import ndtotext_print
import numpy as np
import sympy
from tfq_diamond import U

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

n_qft = 2
A = [cirq.GridQubit(0, i) for i in range(n_qft)]
B = [cirq.GridQubit(1, i) for i in range(n_qft)]
'''
A1 - B1
|    |
A2 - B2
|    |
A3 - B3
...
'''
qc = cirq.Circuit()
for i in range(n_qft):
    qc.append(cirq.I.on(A[i]))  # To make A show up
    if i % 2 == 0:
        qc.append(cirq.X.on(B[i]))  # Opposite to control qubit
    else:
        qc.append(cirq.I.on(B[i]))  # Opposite to target qubit

# Input data
input_values = [1, 0]
for i, value in enumerate(input_values):
    if value == 1:
        qc.append(cirq.X(A[i]))
    else:
        qc.append(cirq.I(A[i]))

sim = cirq.Simulator()
def sim_print(qc):
    res = sim.simulate(qc)
    print('\t', res)

# Alg.
print('init')
sim_print(qc)
qc.append(cirq.H.on(A[0]))  # H on first target
print('H')
sim_print(qc)
qc.append(U_pi.on(A[0], B[1], A[1], B[0]))
print('U_pi')
sim_print(qc)
qc.append(Rn(2).on(B[1]))
print('R(2)')
sim_print(qc)
qc.append(U_pi.on(A[0], B[1], A[1], B[0]))
print('U_pi')
sim_print(qc)
print('H')
qc.append(cirq.H.on(A[1]))
sim_print(qc)

print('QFT cross swap! (tecnical readout order)')
qc.append(cirq.SWAP(A[0], A[1]))
sim_print(qc)

print(qc)

