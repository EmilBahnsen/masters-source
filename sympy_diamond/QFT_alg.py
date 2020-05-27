from sympy import *
from sympy_diamond import *

# THIS is based on: https://www.nature.com/articles/s41598-018-23764-x

print('iSWAP(pi/2)')
pprint(iSWAP_N_pi_2(2, [0,1]))
print('iSWAP(3pi/2)')
pprint(iSWAP_N_3pi_2(2, [0,1]))

def Rn(n):
    return U3(0,0,pi/2**(n-1))

t0, t1 = pi, pi
U_pi = U(t0)

# Can choose either C1/C2 for state of 2, and either T1/T2 for state of 1
C1_state, vs, phis = cp_state(1, ['v', 'phi'])  # State of 1
C2_state, ws, rhos = cp_state(1, ['w', 'rho'])  # State of 2
T1_state = s0
T2_state = s0

state0 = kronecker_product(C1_state, C2_state, T1_state, T2_state).simplify()

print('Init. state:')
pprint(state0)

print('First H on 1 (C1)')
H_C1 = kronecker_product(H, eye(2), eye(2), eye(2))
state = H_C1 @ state0
del state0  # Don't use this again
pprint(state)

print('iSWAP of 1 from C1 to T1')
iswap_C1T1_1 = iSWAP_N_pi_2(4, [0, 2])
state = iswap_C1T1_1 @ state
pprint(state)

print('Do the controlled Rn')
n = Symbol('n', real=True)
Rn = Rn(n)
Rn_T1 = kronecker_product(eye(2), eye(2), Rn, eye(2))
state = (U_pi @ Rn_T1 @ U_pi @ state).simplify()
pprint(state)

print('iSWAP of 1 from C1 to T1 (back)... NO phase by first doing iSWAP(pi/2) and then iSWAP(3pi/2) going back')
iswap_C1T1_2 = iSWAP_N_3pi_2(4, [0, 2])
state = iswap_C1T1_2 @ state
pprint(state)

print('H on 2 (C2)')
H_C2 = kronecker_product(eye(2), H, eye(2), eye(2))
state = H_C2 @ state
pprint(state)

print('Target state')


# print('NOW: Can we swap 1 and 2 with U?')
# X_T1 = kronecker_product(eye(2), eye(2), X, eye(2))
# X_T2 = kronecker_product(eye(2), eye(2), eye(2), X)
# # state = X_T1 @ state
# state = U_pi @ state
# pprint(state)
# target_state = kronecker_product(C2_state, C1_state, T1_state, T2_state)\
#     .simplify().subs(phis[0] + rhos[0], phis[0] + rhos[0] + 2**(n-1))
# print('target_state')
# pprint(target_state)
exit()
# --- STOP STOP STOP ---

print('U(t0) on state:')
state = Ut1 @ state0
pprint(state)

n = Symbol('n', real=True)
Rn = Rn(n)
print('Rn(n)')
pprint(Rn)

Rn_T2 = kronecker_product(eye(2), eye(2), eye(2), Rn)

print('Rn_T2 @ U(t0) @ state:')
state = Rn_T2 @ state
pprint(state)

print('U(t1) @ Rn_T2 @ U(t0) @ state:')
state = Ut1 @ state
pprint(state)

print('t0 = t1 = π')
state = state.simplify()
pprint(state)

print('Target of C_C2Rn_T1 operaton')
CRn = Matrix([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, Rn_T2[0,0], Rn_T2[0,1]],
    [0, 0, Rn_T2[1,0], Rn_T2[1,1]]
])
C_C2Rn_T1 = kronecker_product(eye(2), CRn, eye(2))  # control is C2 and target is T1
target_state = (C_C2Rn_T1 @ state0).simplify()
pprint(target_state)

equal = state == target_state
print(f'Algorithm state equal target state: {equal}')
assert equal

# Let's look at how swapping states down the chain affects the phases!

