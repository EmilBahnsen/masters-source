import cirq
from txtutils import ndtotext_print
import numpy as np
import sympy
from tfq_diamond import U
from typing import *
import qutip
from numpy_partial_trace import partial_trace

def Rn(n):
    return cirq.ZPowGate()**(2**(1-n))
def Rn_dag(n):
    return cirq.ZPowGate()**(-2**(1-n))

def CNS(qc: cirq.Circuit, q1, q2, ctrl, targ):
    qc.append(cirq.H(targ))
    qc.append(cirq.Z(ctrl))
    qc.append(cirq.Z(targ))
    qc.append(U(np.pi).on(q1,q2,ctrl,targ))
    qc.append(cirq.H(ctrl))

n_qft = 4
n_B = 2*(n_qft-1)
A = [cirq.GridQubit(0, i) for i in range(n_qft)]
B = [cirq.GridQubit(1, i) for i in range(n_B)]
'''
   B1  B3  B5
  /  \/  \/  \ 
A1   A2  A3  B4
  \ / \ /  \/
   B2  B4  B6
'''
# Test QFT on the first input
qc = cirq.Circuit()
qc.append(cirq.H(A[0]))
qc.append(Rn(3).on(A[0]))
qc.append(Rn(3).on(A[1]))
CNS(qc, B[0], B[1], A[1], A[0])
qc.append(Rn_dag(3).on(A[1]))
CNS(qc, B[0], B[1], A[0], A[1])

qc.append(Rn(4).on(A[0]))
qc.append(Rn(4).on(A[2]))
CNS(qc, B[0], B[1], A[0], A[1]) ##
CNS(qc, B[2], B[3], A[2], A[1])
qc.append(Rn_dag(4).on(A[2]))
CNS(qc, B[2], B[3], A[1], A[2])
# CNS(qc, B[0], B[1], A[1], A[0]) ##

qc.append(Rn(5).on(A[1]))
qc.append(Rn(5).on(A[3]))
# CNS(qc, B[0], B[1], A[0], A[1]) ##
CNS(qc, B[0], B[1], A[1], A[2]) ##
CNS(qc, B[4], B[5], A[3], A[2])
qc.append(Rn_dag(5).on(A[3]))
CNS(qc, B[4], B[5], A[2], A[3])
CNS(qc, B[0], B[1], A[2], A[1]) ##
CNS(qc, B[0], B[1], A[1], A[0]) ##
matrix = qc.unitary(qubit_order=B+A)[:2**n_qft,:2**n_qft]
ndtotext_print(matrix) # Picked out for all B=0

qc_true = cirq.Circuit()
qc_true.append(cirq.H(A[0]))
qc_true.append(Rn(2).on(A[0]).controlled_by(A[1]))
qc_true.append(Rn(3).on(A[0]).controlled_by(A[2]))
qc_true.append(Rn(4).on(A[0]).controlled_by(A[3]))
matrix_true = qc_true.unitary(qubit_order=A)
ndtotext_print(matrix_true)

print('diff')
ndtotext_print(matrix_true - matrix)

# --- Reduced circuit ---
qc_reduced = cirq.Circuit()
qc_reduced.append(cirq.H(A[0]))
qc_reduced.append(Rn(3).on(A[0]))
qc_reduced.append(Rn(3).on(A[1]))
qc_reduced.append(Rn(4).on(A[0]))
qc_reduced.append(Rn(4).on(A[2]))
qc_reduced.append(Rn(5).on(A[0]))
qc_reduced.append(Rn(5).on(A[3]))
CNS(qc_reduced, B[0], B[1], A[0], A[1])
CNS(qc_reduced, B[2], B[3], A[1], A[2])
CNS(qc_reduced, B[4], B[5], A[3], A[2])
qc_reduced.append(Rn_dag(3).on(A[0]))
qc_reduced.append(Rn_dag(4).on(A[1]))
qc_reduced.append(Rn_dag(5).on(A[3]))
CNS(qc_reduced, B[4], B[5], A[2], A[3])
CNS(qc_reduced, B[2], B[3], A[2], A[1])
CNS(qc_reduced, B[0], B[1], A[1], A[0])
matrix_reduced = qc_reduced.unitary(qubit_order=B+A)[:2**n_qft,:2**n_qft]
print('reduced circuit diff')
ndtotext_print(matrix_true - matrix_reduced)

# qc_test3 = cirq.Circuit()
# qc_test3.append(cirq.CNOT(A[0], A[1]))
# qc_test3.append(cirq.ZPowGate().on(A[1]))
# ndtotext_print(qc_test3.unitary())
#
# qc_test3 = cirq.Circuit()
# qc_test3.append(cirq.CNOT(A[1], A[0]))
# qc_test3.append(cirq.ZPowGate().on(A[0]))
# qc_test3.append(cirq.CNOT(A[1], A[0]))
# qc_test3.append(cirq.CNOT(A[0], A[1]))
# ndtotext_print(qc_test3.unitary())
#
# qc_test3 = cirq.Circuit()
# qc_test3.append(cirq.ZPowGate().on(A[0]).controlled_by(A[1]))
# qc_test3.append(cirq.CNOT(A[0], A[1]))
# ndtotext_print(qc_test3.unitary())