from qutip import *
from qutip.qip.circuit import QubitCircuit
from diamond.DiamondCircuit import save_circuit
import numpy as np

qc = QubitCircuit(4, reverse_states=False)
qc.add_gate('X', 0)
qc.add_gate('X', 1)
qc.add_gate('R_z', 2, [0,1,3], 0, r'\mp \frac{\pi}{2}')
qc.add_gate('R_y', 2, [0,1,3], 0, r'a')
qc.add_gate('R_z', 2, [0,1,3], 0, r'b')
qc.add_gate('A', [0,1,2,3])
qc.add_gate('R_z', 2, [0,1,3], 0, r'\mp \frac{\pi}{2}')
qc.add_gate('R_y', 2, [0,1,3], 0, r'a')
qc.add_gate('R_z', 2, [0,1,3], 0, r'c')
qc.add_gate('B', [0,1,2,3])
qc.add_gate('R_z', 3, [0,1,2], 0, r'\pm \frac{\pi}{2}')
qc.add_gate('R_y', 3, [0,1,2], 0, r'a')
qc.add_gate('R_z', 3, [0,1,2], 0, r'-b')
qc.add_gate('C', [0,1,2,3])
qc.add_gate('R_z', 1, [0,2,3], 0, r'\pm \frac{\pi}{2}')
qc.add_gate('R_y', 1, [0,2,3], 0, r'a')
qc.add_gate('R_z', 1, [0,2,3], 0, r'-c')
qc.add_gate('D', [0,1,2,3])
qc.add_gate('R_z', 3, [0,1,2], 0, r'\mp a')
qc.add_gate('R_y', 3, [0,1,2], 0, r'-\pi')
qc.add_gate('R_z', 3, [0,1,2], 0, r'\pm a')
qc.add_gate(r'R_{\phi}', 3, [0,1,2], 0, r't - \pi')
qc.add_gate('X', 1)
qc.add_gate('X', 2)
save_circuit(qc, 'U')

