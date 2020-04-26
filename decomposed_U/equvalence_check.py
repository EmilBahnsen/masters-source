import tensorflow_quantum as tfq
import cirq
from tfq_diamond import U
import sympy as sp

qubits = cirq.GridQubit.rect(2, 2)
C1, C2, T1, T2 = qubits
qc = cirq.Circuit()

t = sp.Symbol('t', real=True)
qc.append(U(t).on(*qubits))

print(qc)
