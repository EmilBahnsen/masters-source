from diamond import *
import pprint
pp = pprint.PrettyPrinter()

# qc = FourDiamondCircuit()
# n = 4  # Effective qubits
# qc.add_gate('FFT', [0,1,4,5])
# qc.reduce(full=True)
# num_U = sum(map(lambda u: u.name == r'U(\pi)',qc.gates))
# print('#U', num_U)
# save_circuit(qc,'FFT0145')

qc = FourDiamondCircuit()
qc.add_gate('FFT', [0,1,2,3,4,5,6,7])
qc.reduce(full=True)
num_U = sum(map(lambda u: u.name == r'U(\pi)',qc.gates))
print('#gates', len(qc.gates))
print('#U', num_U)
pp.pprint(qc.gate_count_dict())
# save_circuit(qc,'FFT01234567')

def make_FFT_circuit(qubits):
    qc = FourDiamondCircuit()
    qc.add_gate('FFT', qubits)
    qc.reduce(full=True)
    return qc

for i in range(2, 9):
    qubits = list(range(i))
    qc = make_FFT_circuit(qubits)
    print('qubits', qubits)
    pp.pprint(qc.gate_count_dict())
    del qc
