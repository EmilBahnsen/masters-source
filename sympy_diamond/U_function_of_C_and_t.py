from sympy import *
from sympy_diamond import U, U00, U11, Up, Um, s00, s01, s10, s11, _sp, _sm, cp_state, hadamard_division, \
    hadamard_product, SWAP, iSWAP, density_matrix, CNOT, H, U3, CZ, X, Y, Z

x = Wild('x')
y = Wild('y')

a00, a01, a10, a11, ap, am = symbols('a_00 a_01 a_10 a_11 a_p a_m', complex=True)
b00, b01, b10, b11, bp, bm = symbols('b_00 b_01 b_10 b_11 b_p b_m', complex=True)
# Basis coffs.
a = [a00, a11, ap, am]
b = [b00, b11, bp, bm]
bA = [b00, b01, b10, b11]
# Time
t = Symbol('t', real=True, positive=True)
a_abs_square_num = sum([Abs(a[n])**2 for n in range(4)])
Ut = U(t)

A = Matrix.hstack(s00, s01, s10, s11)
B = Matrix.hstack(s00, s11, _sp, _sm)
C = Matrix.hstack(s00, _sp, _sm, s11)
D = Matrix.hstack(s00, (s11 + _sp)/sqrt(2), (s11 - _sp)/sqrt(2), _sm)
T_BA = A @ B.T  # Coordinate transformation matrix from A to B basis
T_AB = T_BA.T   # B @ A.T also, but since they are unitary, this il the case
T_CA = A @ C.T
T_AC = T_CA.T
assert T_BA @ T_AB == eye(4)
assert T_CA @ T_AC == eye(4)

print('T_BA')
pprint(T_BA)
print('T_AB')
pprint(T_AB)

T_I_BA = kronecker_product(eye(4), T_BA)
T_I_AB = kronecker_product(eye(4), T_AB)
T_BA_I = kronecker_product(T_BA, eye(4))
T_AB_I = kronecker_product(T_AB, eye(4))
T_AB_BA = kronecker_product(T_AB, T_BA)
T_BA_AB = kronecker_product(T_BA, T_AB)
T_AB_AB = kronecker_product(T_AB, T_AB)
T_BA_BA = kronecker_product(T_BA, T_BA)

def basis_change(expr, init_basis: Matrix, final_basis: Matrix):
    T_IF = simplify(final_basis @ init_basis.T)
    T_FI = simplify(T_IF.T)
    eye_check = simplify(T_FI @ T_IF)
    assert eye_check == eye(init_basis.cols), f'Transformations not good!\n {pprint(eye_check)}'
    return simplify(T_FI @ expr @ T_IF)

# The U-gate is in the basis AA (as we have rewritten the C-qubits's basis vectors in terms
# of the regular computationel basis (A basis), in the definition of the U-gate),
# now we try to diagonalize it by writing it in the BB basis!
AA = kronecker_product(A, A)
BA = kronecker_product(B, A)
AB = kronecker_product(A, B)
BB = kronecker_product(B, B)
CB = kronecker_product(C, B)
CC = kronecker_product(C, C)
DB = kronecker_product(D, B)
U_BB = basis_change(Ut, AA, BB)
U_CB = basis_change(Ut, AA, CB)
U_AB = basis_change(Ut, AA, AB)
U_CC = basis_change(Ut, AA, CC)
U_DB = basis_change(Ut, AA, DB)

print('U (U_AA)')
pprint(Ut)

print('U_CC')
pprint(U_CC)

# P, D = U(t).diagonalize()
# pprint(P)
# pprint(D)
# U_BB = (T_I_BA @ U(t) @ T_I_AB).simplify()
print('U_BB')
pprint(U_BB)  # Transform the basis of T to B. It's not so much a mess
assert (U_BB @ U_BB.T.conjugate()).simplify() == eye(2**4)  # Just to sanity check
# Now we look at a collective state in the BxB-basis
c = symbols('c_0000 c_0011 c_00p c_00m c_1100 c_1111 c_11p c_11m c_p00 c_p11 c_pp c_pm c_m00 c_m11 c_mp c_mm')
state_TC0 = Matrix([*[[_c] for _c in c]])
# Apply the U_BB
state_TC = U_BB @ state_TC0
print('state_TC')
pprint(state_TC)

print('TC probabs')
dm_TC = density_matrix(state_TC)
for i,_c in enumerate(c):
    pprint(_c)
    pprint(dm_TC[i,i].simplify())

print('iSWAP in A basis')
pprint(iSWAP(t))
print('iSWAP in B basis')
pprint((T_BA @ iSWAP(t) @ T_AB).simplify())

print('SWAP in A basis')
pprint(SWAP(t))
print('SWAP in B basis')
pprint((T_BA @ SWAP(t) @ T_AB).simplify())

print('CNOT in B basis')
pprint((T_BA @ CNOT @ T_AB).simplify())

# pprint((T_AB_I @ U(t) @ T_BA_I).simplify())  # Transform the basis of C to A. It's a mess
# pprint((T_AB_BA @ U(t) @ T_BA_AB).simplify())  # Transform the basis of C to A and T to B. It's a mess

print('Basis matrices in B-basis')
Ui_A = [U00(t), U11(t), Up(t), Um(t)]
Ui_B = [simplify(T_BA @ Ui @ T_AB) for Ui in Ui_A]
print('U00_B')
pprint(Ui_B[0])
print('U11_B')
pprint(Ui_B[1])
print('Up_B')
pprint(Ui_B[2])
print('Um_B')
pprint(Ui_B[3])
exit()

stateT_B = Matrix([
    [b[0]],
    [b[1]],
    [b[2]],
    [b[3]]
])

stateT_A = Matrix([
    [bA[0]],
    [bA[1]],
    [bA[2]],
    [bA[3]]
])

# The init dm of the T-state
rho__T_B0 = stateT_B @ stateT_B.transpose().conjugate()
rho__T_A0 = stateT_A @ stateT_A.transpose().conjugate()
for i in range(4):
    rho__T_B0[i, i] = rho__T_B0[i, i].replace(y * x * x.conjugate(), y * Abs(x) ** 2)
    rho__T_A0[i, i] = rho__T_A0[i, i].replace(y * x * x.conjugate(), y * Abs(x) ** 2)

# The T-state after application of the U-gate
rho__T_B = Matrix.zeros(4)
rho__T_A = Matrix.zeros(4)
for i in range(4):
    rho__T_B += Abs(a[i]) ** 2 * Ui_B[i] @ rho__T_B0 @ Ui_B[i].replace(t, -t)
    rho__T_A += Abs(a[i]) ** 2 * Ui_A[i] @ rho__T_A0 @ Ui_A[i].replace(t, -t)

def simplify_rho_B(rho):
    for i in range(4):
        rho[i, i] = rho[i, i].collect(Abs(b[i])**2)
        rho[i, i] = rho[i, i].replace(a_abs_square_num, 1)
    for i in range(4):
        for j in range(4):
            rho[i, j] = rho[i, j].collect(b[i] * b[j].conjugate())
    return rho

def simplify_rho_A(rho):
    for i in range(4):
        rho[i, i] = rho[i, i].collect(Abs(bA[i])**2)
        rho[i, i] = rho[i, i].replace(a_abs_square_num, 1)
    for i in range(4):
        for j in range(4):
            rho[i, j] = rho[i, j].collect(bA[i] * bA[j].conjugate())
    return rho

rho__T_B = MutableMatrix(rho__T_B)
rho__T_B = simplify_rho_B(rho__T_B)

rho__T_A = MutableMatrix(rho__T_A)
rho__T_A = simplify_rho_A(rho__T_A)

print('rho__T_B')
pprint(rho__T_B)
# print('rho__T_A')
# pprint(rho__T_A)

a00_, a11_, ap_, am_ = Abs(a[0])**2, Abs(a[1])**2, Abs(a[2])**2, Abs(a[3])**2
a01_, a10_ = Abs(a01)**2, Abs(a10)**2

print('rho__T_A from rho__T_B')
# rho__T_A_fromB = T_AB * rho__T_B * T_BA
rho__T_A_fromB = T_AB @ rho__T_B @ T_BA
rho__T_A_fromB = rho__T_A_fromB.replace(bp, (b01 + b10)/sqrt(2)).replace(bm, (b01 - b10)/sqrt(2)).expand()
for i in range(4):
    for j in range(4):
        entry_AfromB = rho__T_A_fromB[i, j]
        entry_AfromB = entry_AfromB.replace(Abs(x + y)**2, Abs(x)**2 + Abs(y)**2 + x*y.conjugate() + y*x.conjugate())
        for sign in [1, -1]:
            for _a in a:
                entry_AfromB = entry_AfromB.subs(b01 * exp(sign*I*t) * conjugate(b01) * Abs(_a)**2 / 4,
                                                 Abs(b01)**2 * exp(sign*I*t) * Abs(_a)**2 / 4)
                entry_AfromB = entry_AfromB.subs(b10 * exp(sign*I*t) * conjugate(b10) * Abs(_a)**2 / 4,
                                                 Abs(b10)**2 * exp(sign*I*t) * Abs(_a)**2 / 4)
        for _a in a:
            entry_AfromB = entry_AfromB.subs(b01 * conjugate(b01) * Abs(_a)**2 / 2,
                                                 Abs(b01)**2 * Abs(_a)**2 / 2)
            entry_AfromB = entry_AfromB.subs(b10 * conjugate(b10) * Abs(_a)**2 / 2,
                                                 Abs(b10)**2 * Abs(_a)**2 / 2)
        entry_A = rho__T_A[i, j].expand()
        diff = entry_AfromB - entry_A
        equal = diff == 0
        if not equal:
            print(i,j)
            print('diff')
            diff = diff.collect(a00_)
            diff = diff.subs(a00_, 1 - a11_ - ap_ - am_).expand()
            diff = diff.expand()
            diff = diff.collect(exp(I*t))
            pprint(diff)
            assert diff == 0
            # exit()

exit()

# print('rho__T_A')
# pprint(rho__T_A)  # This is very cluttered

print('rho__T_B0')
pprint(rho__T_B0)

print('rho__T_B / rho__T_B0')
rho__T_B_div_rho__T_B0 = hadamard_division(rho__T_B, rho__T_B0)
pprint(rho__T_B_div_rho__T_B0)
def rewrite_rho__T_B(rho):
    rho[0,1] = rho[0,1].replace(am_, 1 - a00_ - a11_ - ap_).collect(a00_).collect(a11_).collect(ap_).collect(exp(I*t) - 1)
    rho[1,0] = rho[1,0].replace(am_, 1 - a00_ - a11_ - ap_).collect(a00_).collect(a11_).collect(ap_).collect(exp(-I*t) - 1)

    rho[0,2] = rho[0,2].replace(a11_, 1 - a00_ - ap_ - am_).collect(a00_).collect(ap_).collect(exp(I*t) - 1)
    rho[2,0] = rho[2,0].replace(a11_, 1 - a00_ - ap_ - am_).collect(a00_).collect(ap_).collect(exp(-I*t) - 1)

    rho[0,3] = rho[0,3].replace(a00_, 1 - a11_ - ap_ - am_).collect(a11_).collect(ap_).collect(exp(I*t) - 1)
    rho[3,0] = rho[3,0].replace(a00_, 1 - a11_ - ap_ - am_).collect(a11_).collect(ap_).collect(exp(-I*t) - 1)

    rho[1,2] = rho[1,2].replace(a00_, 1 - a11_ - ap_ - am_).collect(a11_).collect(ap_).collect(exp(-I*t) - 1)
    rho[2,1] = rho[2,1].replace(a00_, 1 - a11_ - ap_ - am_).collect(a11_).collect(ap_).collect(exp(I*t) - 1)

    rho[1,3] = rho[1,3].replace(a11_, 1 - a00_ - ap_ - am_).collect(a00_).collect(ap_).collect(exp(-I*t) - 1)
    rho[3,1] = rho[3,1].replace(a11_, 1 - a00_ - ap_ - am_).collect(a00_).collect(ap_).collect(exp(I*t) - 1)

    rho[2,3] = rho[2,3].replace(ap_, 1 - a00_ - a11_ - am_).collect(a00_).collect(a11_).collect(exp(-I*t) - 1)
    rho[3,2] = rho[3,2].replace(ap_, 1 - a00_ - a11_ - am_).collect(a00_).collect(a11_).collect(exp(-I*t) - 1)
    # rho[0,2] = rho[0,2].replace(x*y - y, )
    return rho
rho__T_B_div_rho__T_B0 = rewrite_rho__T_B(rho__T_B_div_rho__T_B0)
print('... simplified')
pprint(rho__T_B_div_rho__T_B0)
pprint(rho__T_B_div_rho__T_B0.free_symbols)

print('rho__T_B / rho__T_B0 - Matrix.ones(4)')
pprint(rho__T_B_div_rho__T_B0 - Matrix.ones(4))

z_t = Symbol('z(t)')  # z(t)
pprint(z_t)
exit()

print('rho__T_B / rho__T_B0 for a00 = 1')
pprint(rho__T_B_div_rho__T_B0.replace(a[0], 1).replace(a[1], 0).replace(a[2], 0).replace(a[3], 0))
print('rho__T_B / rho__T_B0 for a11 = 1')
pprint(rho__T_B_div_rho__T_B0.replace(a[0], 0).replace(a[1], 1).replace(a[2], 0).replace(a[3], 0))
print('rho__T_B / rho__T_B0 for ap = 1')
pprint(rho__T_B_div_rho__T_B0.replace(a[0], 0).replace(a[1], 0).replace(a[2], 1).replace(a[3], 0))
print('rho__T_B / rho__T_B0 for am = 1')
pprint(rho__T_B_div_rho__T_B0.replace(a[0], 0).replace(a[1], 0).replace(a[2], 0).replace(a[3], 1))

a_U_sum = Matrix.zeros(4)
for i in range(4):
    a_U_sum += a[i] * Ui_B[i]
pprint(a_U_sum @ a_U_sum.transpose().conjugate())

##### STOP #####
exit()

state_T, vs, phis = cp_state(2, ['v', 'phi'])
pprint(state_T @ state_T.T)

U_T_t = a[0]*U00(t) + a[1]*U11(t) + a[2]*Up(t) + a[3]*Um(t)

print('U_T_t')
pprint(U_T_t)

Ut = U(t)
stateC = Matrix([
    [a[0]],
    [(a[2] + a[3])/sqrt(2)],
    [(a[2] - a[3])/sqrt(2)],
    [a[1]]
])

state = kronecker_product(stateC, stateT_B)

print('Ut @ state')
pprint(Ut @ state)

print('kron eye x U_T_t')
U_CT_t = kronecker_product(eye(2**2), U_T_t)
pprint((U_CT_t @ state).simplify())
