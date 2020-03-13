from qutip.qip.operations import swap

from diamond import *
from qutip import *
from math import *

# Define gates for 2 diamond chip
# -- 1: z = π/2 -------- 2: z = π/3 ---
#
#       0:C1a                4:C1b
# 2:T1a  |    3:T2a –– 6:T1b  |    7:T2b
#       1:C2a                5:C2b
#
N = 2*4
U2 = lambda t: UN(t,t)
SWAP = swap(N,[3,6])

init = ket('0'*8)

entangling_opers = [
    {
        'oper': U2(π/2) * X(N,0) * U2(π/2) * X(N,1),
        'state': (ket('11000000') + zero8)/sqrt(2),
        'goal state': '1100 0000'
    }, {
        'oper': U2(π/2) * X(N,4) * U2(π/2) * X(N,5),
        'state': (ket('00001100') + zero8)/sqrt(2),
        'goal state': '0000 1100'
    }, {
        'oper': U2(π/2) * X(N,2) * U2(π/2) * X(N,3),
        'state': (ket('00110000') + zero8)/sqrt(2),
        'goal state': '0011 0000'
    }, {
        'oper': U2(π/2) * X(N,6) * U2(π/2) * X(N,7),
        'state': (ket('00000011') + zero8)/sqrt(2),
        'goal state': '0000 0011'
    }, {
        'oper': SWAP * U2(π/2) * X(N,2) * U2(π/2) * X(N,3),
        'state': (ket('00100010') + zero8)/sqrt(2),
        'goal state': '0010 0010'
    }, {
        'oper': psi_p(N,0,1).dag() * U2(π) * psi_p(N,0,1) * SWAP * U2(π/2) * X(N,2) * U2(π/2) * X(N,3),
        'state': (ket('00100001') + zero8)/sqrt(2),
        'goal state': '0010 0001'
    }, {
        'oper': U2(π) * SWAP * U2(π/2) * X(N,2) * U2(π/2) * X(N,3),
        'state': (ket('00010001') + zero8)/sqrt(2),
        'goal state': '0001 0001'
    }, {
        'oper': psi_p(N,0,1).dag() * U2(π) * psi_p(N,0,1) * U2(π) * SWAP * U2(π/2) * X(N,2) * U2(π/2) * X(N,3),
        'state': (ket('00010010') + zero8)/sqrt(2),
        'goal state': '0001 0010'
    }
]

# entangling_opers = sorted(entangling_opers, key=lambda ent: ent['goal state'])
for entangling_oper in entangling_opers:
    oper = entangling_oper['oper']
    state = entangling_oper['state']
    goal_state = entangling_oper['goal state']
    final_state = oper*init
    print('State:', state2string(final_state))
    print('F({}) = {}'.format(goal_state, fidelity(final_state, state)))
    # log_negs = [log2(2 * negativity(final_state * final_state.dag(), i) + 1) for i in range(N)]
    # #log_negs = [0]*final_state.shape[0]
    # log_neg = sum(list(map(lambda x: x[1] if x[0]==1 else 0,zip(map(int,goal_state.replace(' ','')),log_negs))))
    # neg = (2**(log_neg) - 1)/2
    # print('N({}) = {}'.format(goal_state, neg))
    print()


def containsSolutionFor(pattern):
    for entangling_oper in entangling_opers:
        if entangling_oper['goal state'].replace(' ','') == str(pattern):
            return True
    return False


print('Missing patterns:')
for i in range(N):
    for j in range(i):
        index = 2**i + 2**j
        pattern = index2label(index, N)
        if not containsSolutionFor(pattern):
            print(pattern)


for i in range(2**N):
    init_state = ket(index2label(i,N))
    final_state = entangling_opers[2]['oper'] * init_state
    print(state2string(init_state,4), '->', state2string(final_state,4))
