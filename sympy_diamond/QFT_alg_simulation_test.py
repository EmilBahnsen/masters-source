import cirq
from txtutils import ndtotext_print
import numpy as np
import sympy

π = np.pi

print('iSWAP(π/2) [NOT THE ONE WE USE, BUT MIGHT WORK ANYWAY!]')
ndtotext_print(cirq.riswap(π/2)._unitary_())
print('iSWAP(π/2) @ iSWAP(3π/2)')
ndtotext_print(cirq.riswap(π/2)._unitary_() @ cirq.riswap(3*π/2)._unitary_())

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
.
'''
qc = cirq.Circuit()

# Input data
input_values = [1, 1, 1]
for i, value in enumerate(input_values):
    if value == 1:
        qc.append(cirq.X(A[i]))
    else:
        qc.append(cirq.I(A[i]))

# Alg.
qc.append(cirq.QFT(*A))
print(qc)

# --- Simulator ---
sim = cirq.Simulator()

res = sim.simulate(qc)
print(res)
for i in range(n_qft):
    amp = res.final_state[2**i]
    print(f'|{2**i}>: {amp}')
    exp_sum = np.sum([value/(2**(j+1)) for j,value in enumerate(input_values[i:])])
    amp_check = np.exp(2*π*1j*exp_sum)/np.sqrt(2**n_qft)
    print('\t', amp_check)  # Divide by 1/sqrt(2**n)
    print(np.isclose(amp, amp_check))

# samples = sim.run(qc, repetitions=1000)
# print(samples.histogram(key='result'))
ndtotext_print(qc.unitary())

qc.append(cirq.QFT(*A)**(-1))
ndtotext_print(qc.unitary())