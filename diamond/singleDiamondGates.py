from diamond import *

qc = DiamondCircuit(4)

cnot_gate_list

cnot01_gate_list = [['H', 0], [r'\psi^{-}', [0, 1]], ['Z', 0], ['X', 1]]
cnot01 = DiamondCircuit(4)
add_gates(cnot01, cnot01_gate_list)

cnot10_gate_list = [['H', 1], [r'\psi^{-}', [0, 1]], ['Z', 0], ['X', 1], ['H', 0], ['H', 1]]
cnot10 = DiamondCircuit(4)
add_gates(cnot10, cnot10_gate_list)

cnot02_gate_list = [['SWAP', [1, 2]], *cnot01_gate_list , ['SWAP', [1, 2]]]
cnot02 = DiamondCircuit(4)
add_gates(cnot02, cnot02_gate_list)

cnot20_gate_list = [['SWAP', [1, 2]], *cnot10_gate_list , ['SWAP', [1, 2]]]
cnot20 = DiamondCircuit(4)
add_gates(cnot20, cnot20_gate_list)

gates = {'cnot01': cnot01, 'cnot10': cnot10, 'cnot02': cnot02, 'cnot20': cnot20}

for i,cnot in gates.items():
    print(i)
    oper = qc2matrix(cnot)
    apply_to_basis(oper, diamond_basis(4), print_non_changed=True)
