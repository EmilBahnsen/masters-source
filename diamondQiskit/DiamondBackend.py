import math
from qiskit import *
from qiskit.visualization import *
from qiskit.providers.aer.backends import *
from qiskit.providers.models import *
from qiskit import Aer
import os
os.environ['PYOPENCL_CTX'] = '0'
from qiskit_qcgpu_provider.qasm_simulator import *

π = math.pi


# --- Construct U ---
# omega = t_g · J_C
def U(omega):
    # U_A
    UA = QuantumCircuit(2, name='U_A')
    UA.cx(1,0)
    UA.ch(0,1)
    UA.cx(1,0)

    # U_B
    UB = QuantumCircuit(2, name='U_B')
    UB.cz(0,1)
    UB.swap(0,1)
    UB.z(0)
    UB.z(1)

    # U_C
    UC = QuantumCircuit(3, name='U_C')
    UC.h(2)
    UC.ccx(0,1,2)
    UC.h(2)
    UC.cswap(0,1,2)
    UC.cz(0,1)
    UC.cz(0,2)
    UC.rz(omega,0)

    # U_D
    UD = QuantumCircuit(3, name='U_D')
    UD.h(2)
    UD.ccx(0,1,2)
    UD.h(2)
    UD.cswap(0,1,2)
    UD.z(0)
    UD.rz(-omega,0)

    cr = QuantumRegister(2, name='qc')
    tr = QuantumRegister(2, name='qt')

    U = QuantumCircuit(cr, tr)
    if π % omega == 0:
        U.name = 'U(π/{})'.format(round(π/omega))
    elif omega % π == 0:
        U.name = 'U({}π)'.format(round(omega/π))
    else:
        U.name = 'U({})'.format(omega)

    U.append(UA, [0,1])
    U.append(UB, [2,3])
    U.append(UC, [1,2,3])
    U.append(UD, [0,2,3])
    U.append(UA, [0,1])

    return U

U_gate = U(1).decompose()
qasm_def = U_gate.qasm()#.replace('1)', 'omega)')

cx_gate = QuantumCircuit(2)
cx_qasm_def = cx_gate.qasm()

UGateConfig = GateConfig(name='dU', parameters=['omega'], qasm_def=qasm_def)
# cxGateConfig = GateConfig(name='cx', parameters=[], qasm_def=cx_qasm_def)
#xGateConfig = GateConfig(name='X', parameters=[], qasm_def=QuantumCircuit().x(0))

class DiamondBackend(BaseBackend):
    def __init__(self):
        '''
            0
          /   \
        2       3
          \   /
            1
        '''

        cmap = [[0, 2], [0, 3], [1, 2], [1, 3]]

        configuration = QasmBackendConfiguration(
            backend_name='diamond',
            backend_version='0.1.0',
            n_qubits=4,
            basis_gates=['cx'],
            simulator=False,
            local=True,
            conditional=False,
            open_pulse=False,
            memory=False,
            max_shots=65536,
            gates=[UGateConfig],
            coupling_map=cmap,
        )

        super().__init__(configuration)

    def run(self, qobj):
        pass

# diamondConfiguration: BackendConfiguration = QCGPUQasmSimulator.DEFAULT_CONFIGURATION.copy()
# diamondConfiguration['backend_name'] = 'diamond'
# diamondConfiguration['backend_version'] = '0.1.0'
# diamondConfiguration['n_qubits'] = 4
# # diamondConfiguration['basis_gates'].append(['dU'])
# # diamondConfiguration['gates'].append([UGateConfig])
# diamondConfiguration['coupling_map'] = [[0,1], [0,2], [3,1], [3,2]]

backend = DiamondBackend()
new_QC: QuantumCircuit = transpile(U_gate, backend, basis_gates=['cx','u3'])
print(new_QC.decompose().draw())
# job = execute(U_gate, backend, shots=1024)
# result = job.result()
# unitary = result.get_unitary()
# plot_gate_map(backend)
# print(unitary)
# print(U_gate.draw())

from qiskit.test.mock.fake_backend import *