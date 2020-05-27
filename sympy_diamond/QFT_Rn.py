from sympy import *
from sympy_diamond import *

# THIS is based on: https://www.nature.com/articles/s41598-018-23764-x

def Rn(n):
    return U3(0,0,pi/2**(n-1))

t0, t1 = pi, pi
Ut0, Ut1 = U(t0), U(t1)

# Can choose either C1/C2 for state of 2, and either T1/T2 for state of 1
C1_state = s1
C2_state, ws, rhos  = cp_state(1, ['w', 'rho'])  # State of 2
T1_state, vs, phis  = cp_state(1, ['v', 'phi'])  # State of 1
T2_state = s0

state0 = kronecker_product(C1_state, C2_state, T1_state, T2_state)

print('Init. state:')
pprint(state0)

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