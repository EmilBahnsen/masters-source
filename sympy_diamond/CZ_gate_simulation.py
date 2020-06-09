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
Z_C1 = sympy.kronecker_product(U3(0,0,t1), eye(2), eye(2), eye(2))
Z_C2 = sympy.kronecker_product(eye(2), U3(0,0,t2), eye(2), eye(2))
Z_T1 = sympy.kronecker_product(eye(2), eye(2), U3(0,0,t3), eye(2))
Z_T2 = sympy.kronecker_product(eye(2), eye(2), eye(2), U3(0,0,t4))

oper = (U(π) @ Z_T2 @ U(π)).simplify()
print('With sympy')
pprint(oper)

oper = (U(π) @ Z_C1 @ Z_C2 @ Z_T1 @ Z_T2 @ U(π)).simplify()
print('Phases on all qubits:')
pprint(oper)

oper_noU = (Z_C1 @ Z_C2 @ Z_T1 @ Z_T2).simplify()
print('Phases on all qubits (NO U):')
pprint(oper_noU)

print('U phase U TIMES (phase with opposite sign)')
pprint(simplify(oper @ oper_noU.subs(t1, -t1).subs(t2, -t2).subs(t3, -t3).subs(t4, -t4)))

# swap_T1C2 = kronecker_product(eye(2), SWAP(π/2), eye(2))
# oper = swap_T1C2 @ oper @ swap_T1C2
# print('Now with swapped such that the order is : C1 T1 C2 T2')
# pprint(oper)


# --- Again with Cirq ---
print('CIRCUIT')
qc2 = cirq.Circuit()
_t1, _t2, _t3, _t4 = 1,2,3,4
qc2.append(cirq.SWAP(T1, T2).controlled_by(C1, C2))
qc2.append(cirq.SWAP(T1, T2).controlled_by(C1, C2, control_values=[0,0]))
qc2.append(cirq.SWAP(C1, C2).controlled_by(T1, T2))
qc2.append(cirq.SWAP(C1, C2).controlled_by(T1, T2, control_values=[0,0]))
qc2.append(cirq.ZPowGate().on(C1)**(_t1/np.pi))
qc2.append(cirq.ZPowGate().on(C2)**(_t2/np.pi))
qc2.append(cirq.ZPowGate().on(T1)**(_t3/np.pi))
qc2.append(cirq.ZPowGate().on(T2)**(_t4/np.pi))
qc2.append(cirq.SWAP(T1, T2).controlled_by(C1, C2))
qc2.append(cirq.SWAP(T1, T2).controlled_by(C1, C2, control_values=[0,0]))
qc2.append(cirq.SWAP(C1, C2).controlled_by(T1, T2))
qc2.append(cirq.SWAP(C1, C2).controlled_by(T1, T2, control_values=[0,0]))
criq_unitary = qc2.unitary(qubit_order=qubit_order)
ndtotext_print(criq_unitary)

print('The one from nympy with same values for t:')
sympy_array = oper.subs(t1, _t1).subs(t2, _t2).subs(t3, _t3).subs(t4, _t4)
numpy_array = np.array(sympy_array).astype(np.complex)
ndtotext_print(numpy_array)
print('diff')
ndtotext_print(criq_unitary - numpy_array)

# --- again with rewritten circuit ---
print('CIRCUIT rewritten!')
qc3 = cirq.Circuit()
qc3.append(cirq.ZPowGate().on(C2).controlled_by(T1, T2, control_values=[0,0])**(_t1/np.pi))
qc3.append(cirq.ZPowGate().on(C2).controlled_by(T1, T2, control_values=[1,1])**(_t1/np.pi))
qc3.append(cirq.ZPowGate().on(C1).controlled_by(T1, T2, control_values=[1,0])**(_t1/np.pi))
qc3.append(cirq.ZPowGate().on(C1).controlled_by(T1, T2, control_values=[0,1])**(_t1/np.pi))

qc3.append(cirq.ZPowGate().on(C1).controlled_by(T1, T2, control_values=[0,0])**(_t2/np.pi))
qc3.append(cirq.ZPowGate().on(C1).controlled_by(T1, T2, control_values=[1,1])**(_t2/np.pi))
qc3.append(cirq.ZPowGate().on(C2).controlled_by(T1, T2, control_values=[1,0])**(_t2/np.pi))
qc3.append(cirq.ZPowGate().on(C2).controlled_by(T1, T2, control_values=[0,1])**(_t2/np.pi))

qc3.append(cirq.ZPowGate().on(T2).controlled_by(C1, C2, control_values=[0,0])**(_t3/np.pi))
qc3.append(cirq.ZPowGate().on(T2).controlled_by(C1, C2, control_values=[1,1])**(_t3/np.pi))
qc3.append(cirq.ZPowGate().on(T1).controlled_by(C1, C2, control_values=[1,0])**(_t3/np.pi))
qc3.append(cirq.ZPowGate().on(T1).controlled_by(C1, C2, control_values=[0,1])**(_t3/np.pi))

qc3.append(cirq.ZPowGate().on(T1).controlled_by(C1, C2, control_values=[0,0])**(_t4/np.pi))
qc3.append(cirq.ZPowGate().on(T1).controlled_by(C1, C2, control_values=[1,1])**(_t4/np.pi))
qc3.append(cirq.ZPowGate().on(T2).controlled_by(C1, C2, control_values=[1,0])**(_t4/np.pi))
qc3.append(cirq.ZPowGate().on(T2).controlled_by(C1, C2, control_values=[0,1])**(_t4/np.pi))

print(qc3.to_text_diagram(qubit_order=qubit_order))
criq_unitary = qc3.unitary(qubit_order=qubit_order)
ndtotext_print(criq_unitary)

print('The one from nympy with same values for t:')
ndtotext_print(numpy_array)
print('diff')
ndtotext_print(criq_unitary - numpy_array)