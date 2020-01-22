from diamond import *

qc = FourDiamondCircuit()
n = 4  # Effective qubits
qc.add_gate('FFT', [0,1,4,5])
qc.reduce(full=True)
num_U = sum(map(lambda u: u.name == r'U(\pi)',qc.gates))
print('#U', num_U)
save_circuit(qc,'FFT0145')
