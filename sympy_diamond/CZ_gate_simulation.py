import cirq
from txtutils import ndtotext_print
import numpy as np
import sympy
from tfq_diamond import U
from typing import *
import itertools

π = np.pi
# t = sympy.Symbol('t', real=True)
t = 1

U_pi = U(π)

C1, C2, T1, T2 = cirq.GridQubit(0,0), cirq.GridQubit(1,1), cirq.GridQubit(1,0), cirq.GridQubit(0,1)
qubit_order=[C1,C2,T1,T2]
qc = cirq.Circuit()

qc.append(U_pi.on(C1,C2,T1,T2))
qc.append(cirq.ZPowGate(exponent=t/π).on(T2))  # Z(t)
qc.append(U_pi.on(C1,C2,T1,T2))
ndtotext_print(qc.unitary(qubit_order=qubit_order))

sim = cirq.Simulator()

seq = itertools.product([False, True], repeat=4)

for i,j,k,l in seq:
    qc = cirq.Circuit()  # Clear circuit
    if i:
        qc.append(cirq.X.on(C1))
    else:
        qc.append(cirq.I.on(C1))
    if j:
        qc.append(cirq.X.on(C2))
    else:
        qc.append(cirq.I.on(C2))
    if k:
        qc.append(cirq.X.on(T1))
    else:
        qc.append(cirq.I.on(T1))
    if l:
        qc.append(cirq.X.on(T2))
    else:
        qc.append(cirq.I.on(T2))
    qc.append(U_pi.on(C1,C2,T1,T2))
    qc.append(cirq.ZPowGate(exponent=t/π).on(T2))  # Z(t)
    qc.append(U_pi.on(C1,C2,T1,T2))

    # Simulate
    res = sim.simulate(qc,qubit_order=qubit_order)
    print(f'|C1 C2 T1 T2> = |{int(i), int(j), int(k), int(l)}>')
    print(res)

# --- Now with SymPy ---
from sympy_diamond import *
from sympy import *

π = pi

t1, t2, t3, t4 = sympy.symbols('t1 t2 t3 t4', real=True)
Z_T2 = sympy.kronecker_product(eye(2), eye(2), eye(2), U3(0,0,t1))
Z_T1 = sympy.kronecker_product(eye(2), eye(2), U3(0,0,t2), eye(2))
Z_C2 = sympy.kronecker_product(eye(2), U3(0,0,t3), eye(2), eye(2))
Z_C1 = sympy.kronecker_product(U3(0,0,t4), eye(2), eye(2), eye(2))

oper = (U(π) @ Z_T2 @ U(π)).simplify()
print('With sympy')
pprint(oper)

oper = (U(π) @ Z_T2 @ Z_T1 @ Z_C2 @ Z_C1 @ U(π)).simplify()
print('Phases on all qubits:')
pprint(oper)

# swap_T1C2 = kronecker_product(eye(2), SWAP(π/2), eye(2))
# oper = swap_T1C2 @ oper @ swap_T1C2
# print('Now with swapped such that the order is : C1 T1 C2 T2')
# pprint(oper)
