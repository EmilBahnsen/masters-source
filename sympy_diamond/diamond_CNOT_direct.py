import cirq
from txtutils import ndtotext_print
import numpy as np
import sympy
from tfq_diamond import U
from typing import *

π = np.pi

# ---- GOOD -----
qc = cirq.Circuit()
C1, C2, T1, T2 = cirq.GridQubit(0,0), cirq.GridQubit(1,1), cirq.GridQubit(1,0), cirq.GridQubit(0,1)

qc.append(cirq.SWAP.on(T1, T2))

qc.append(cirq.H.on(T1))
qc.append(cirq.Z.on(T1))
qc.append(cirq.Z.on(T2))
qc.append(U(π).on(C1, C2, T1, T2))
qc.append(cirq.H.on(T2))

ndtotext_print(qc.unitary(qubit_order=[C1, C2, T1, T2]))

# ---- BAD -----
qc2 = cirq.Circuit()

qc2.append(cirq.XX.on(C1,C2))
qc2.append(cirq.H.on(T2))
qc2.append(U(π).on(C1, C2, T1, T2))
qc2.append(cirq.H.on(T1))
qc2.append(cirq.XX.on(C1,C2))
qc2.append(U(π).on(C1, C2, T1, T2))

ndtotext_print(qc2.unitary(qubit_order=[C1, C2, T1, T2]))
