import cirq
from tfq_diamond import U
import numpy as np
from txtutils import ndtotext_print

π = np.pi

qubits = cirq.GridQubit.rect(2, 2)
C1, C2, T1, T2 = qubits
qc = cirq.Circuit()

t = 1

qc.append(U(π).on(*qubits))
qc.append(cirq.XX.on(T1, T2)**t)
qc.append(U(π).on(*qubits))

ndtotext_print(qc.unitary())

qc = cirq.Circuit()

t = 1
qc.append(cirq.I.on(C1))
qc.append(cirq.I.on(C2))
qc.append(cirq.XX.on(T1, T2)**t)

ndtotext_print(qc.unitary())
