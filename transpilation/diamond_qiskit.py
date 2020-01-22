from diamondQiskit import *

U_8 = U(math.pi/8)

print('Block diagram:')
print(U_8.draw())

print('Decomposition:')
print(U_8.decompose().draw())
decomposed_U_8 = U_8.decompose().decompose().decompose().decompose().decompose()
print(decomposed_U_8.count_ops())

# --- simulate the 4 gates ---

cr = QuantumRegister(2, name='c')
tr = QuantumRegister(2, name='t')
#mea = ClassicalRegister(2, name='target')

qc = QuantumCircuit(cr, tr)
#qc.x(0)
#qc.x(1)
#qc.x(2)
#qc.x(3)
qc.append(U_8,[0,1,2,3])
#qc.measure([2,3], [0,1])
print(qc.draw())

# Import Aer
from qiskit import Aer

# --- simulation ---
# Run the quantum circuit on a statevector simulator backend
# backend = BasicAer.get_backend('qasm_simulator') # the device to run on
# result = execute(qc, backend, shots=10000).result()
# counts  = result.get_counts(qc)
# plot_histogram(counts).show()

# execute the quantum circuit
# backend = BasicAer.get_backend('statevector_simulator') # the device to run on
# result = execute(qc, backend).result()
# psi  = result.get_statevector(qc)
# plot_bloch_multivector(psi).show()