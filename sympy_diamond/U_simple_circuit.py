from sympy_diamond import U
from sympy import *
import cirq
from txtutils import ndtotext_print
import numpy as np

def XX_B(qc:cirq.Circuit, q1, q2):
    qc.append(cirq.X(q2))
    qc.append(cirq.X.on(q1).controlled_by(q2))
    qc.append(cirq.X(q1))
    qc.append(cirq.H.on(q2))
    qc.append(cirq.X.on(q1).controlled_by(q2))
    qc.append(cirq.X(q2))

qc = cirq.Circuit()
C1, C2, T1, T2 = cirq.GridQubit(0,0), cirq.GridQubit(1,1), cirq.GridQubit(1,0), cirq.GridQubit(0,1)
qubit_order=[C1,C2,T1,T2]
# U00
qc.append(cirq.XX.on(C1, C2)) # --
qc.append(cirq.SWAP.on(T1, T2).controlled_by(C1, C2))
qc.append(cirq.Z.on(T2).controlled_by(C1, C2, T1))
qc.append(cirq.Z.on(T1).controlled_by(C1, C2))
qc.append(cirq.Z.on(T2).controlled_by(C1, C2))
qc.append(cirq.XX.on(C1, C2)) # --
# # U11
qc.append(cirq.SWAP.on(T1, T2).controlled_by(C1, C2))
qc.append(cirq.Z.on(T2).controlled_by(C1, C2, T1))
qc.append(cirq.Z.on(C2).controlled_by(C1))

XX_B(qc, C1, C2)
# Up
qc.append(cirq.XX.on(C1, C2)) # --
qc.append(cirq.Z.on(T1).controlled_by(C1, C2))
qc.append(cirq.Z.on(T2).controlled_by(C1, C2))
qc.append(cirq.Z.on(C2).controlled_by(C1))
# PHASE
qc.append(cirq.XX.on(C1, C2)) # --
#Um
# PHASE
XX_B(qc, C1, C2)

ndtotext_print(qc.unitary(qubit_order=qubit_order))
U_pi = np.array(U(pi).tolist()).astype(np.float64)
ndtotext_print(U_pi)

ndtotext_print(U_pi - qc.unitary(qubit_order=qubit_order))
