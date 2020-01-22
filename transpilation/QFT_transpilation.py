import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as  plt
from diamondQiskit import *


N = 4
qr = QuantumRegister(N)
qc: QuantumCircuit = QuantumCircuit(qr)
qft(qc,N)
tp_qc: QuantumCircuit = transpile(qc,
                  coupling_map=[[0,2], [0,3], [1,2], [1,3]],
                  optimization_level=3)
qc.draw(output='mpl')
print(tp_qc.draw())
