from diamond import *

N = 2*4
CNOT_23 = U2(π) * X(N,1) * X(N,0) * H(N,2) * U2(π) * H(N,3) * X(N,1) * X(N,0)
CNOT_32 = X(N,1) * X(N,0) * H(N,2) * U2(π) * H(N,3) * X(N,1) * X(N,0) * U2(π)
CNOT_67 = U2(π) * X(N,5) * X(N,4) * H(N,6) * U2(π) * H(N,7) * X(N,5) * X(N,4)
CNOT_76 = X(N,5) * X(N,4) * H(N,6) * U2(π) * H(N,7) * X(N,5) * X(N,4) * U2(π)
CNOT_62 = iSWAP36(N) * CNOT_32 * iSWAP36(N)
CNOT_26 = iSWAP36(N) * CNOT_23 * iSWAP36(N)
CNOT_73 = U2(π) * CNOT_62 * U2(π)
CNOT_37 = U2(π) * CNOT_26 * U2(π)

# SWAP65 = swap(N,[5,6])
# CNOT_25 = SWAP65 * CNOT_26 * SWAP65
opers = [
    ('CNOT_23', CNOT_23),
    ('CNOT_32', CNOT_32),
    ('CNOT_67', CNOT_67),
    ('CNOT_76', CNOT_76),
    ('CNOT_62', CNOT_62),
    ('CNOT_26', CNOT_26),
    ('CNOT_73', CNOT_73),
    ('CNOT_37', CNOT_37)
]

states = [
    DiamondState('0000 0001'),
    DiamondState('0000 0010'),
    DiamondState('0000 0011'),
    DiamondState('0001 0000'),
    DiamondState('0000 0001'),
    DiamondState('0001 0010'),
    DiamondState('0001 0011'),
    DiamondState('0010 0000'),
    DiamondState('0010 0001'),
    DiamondState('0010 0010'),
    DiamondState('0010 0011'),
    DiamondState('0011 0000'),
    DiamondState('0011 0001'),
    DiamondState('0011 0010'),
    DiamondState('0011 0011')
]

for i,oper in enumerate(opers):
    print('Operation:',oper[0])
    for state in states:
        state_new = oper[1] * state
        if state != state_new and state != -state_new:
            print(state, '->', bcolors.BOLD, state_new, bcolors.ENDC)
        else:
            print(state, '-> ', state_new)
    print()

