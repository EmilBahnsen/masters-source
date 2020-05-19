from sympy import *
from sympy_diamond import cp_state, U, partial_trace, density_matrix

# THIS is akin to diamond_nn/x2y2/analytic_xTTyCC.py

C1C2, v, phi = cp_state(2, ['c', r'c^'])
T1T2, w, rho = cp_state(2, ['t', r't^'])
state = kronecker_product(C1C2, T1T2)
print('state')
pprint(state)

t = Symbol('t', real=True, positive=True)
Ut = U(t)

print('Ut traced down to act on ')

print('State after U')
state = Ut @ state
pprint(state)

print('dm of T1T2')
dmT1T2 = partial_trace(density_matrix(state), 4, [0, 1])
pprint(dmT1T2)

print('P(00)')
pprint(dmT1T2[0,0].simplify())
print('P(01)')
pprint(dmT1T2[1,1].simplify())
print('P(10)')
pprint(dmT1T2[2,2].simplify())
print('P(11)')
pprint(dmT1T2[3,3].simplify())

print('dm of T1T2 simplified')
pprint(dmT1T2.simplify())
