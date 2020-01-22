from diamond import *

cnot15 = FourDiamondCircuit()

save_swap_gate_list = [['SWAP',[5,15]], ['SWAP',[1,14]], ['SWAP',[4,11]], ['SWAP',[0,10]]]

add_gates(cnot15, save_swap_gate_list)
cnot15.add_gate(r'\psi^{-}', [0,1])  # Save the states of diamond a,b,c, it is not affected
cnot15.add_gate(r'\psi^{-}', [4,5])
cnot15.add_gate(r'\psi^{-}', [8,9])
add_gates(cnot15, basic_entanglement(12))
cnot15.add_gate('U(\pi)')
cnot15.add_gate(r'(\psi^{-})^{\dag}', [8,9])
cnot15.add_gate(r'(\psi^{-})^{\dag}', [4,5])
cnot15.add_gate(r'(\psi^{-})^{\dag}', [0,1])
add_gates(cnot15, reversed(save_swap_gate_list))

save_circuit(cnot15, 'cnot15')

cnot15_prop = qc2matrix(cnot15)
# apply_to_basis(cnot15_prop, four_diamond_basis, print_non_changed=False)


# Compare the effect of two operators up to a global phase
def compare_opers(oper1: Qobj, oper2: Qobj, basis: [Qobj]):
    for state in basis:
        state1 = oper1 * state
        state2 = oper2 * state
        print()
        # if fidelity(state1, state2) != 1:
        print('Operators differ:')
        print(state, '->', state1)
        print(state, '->', state2)
    return True


compare_opers(cnot15_prop, cnot(4*4, 1, 5), four_diamond_basis)