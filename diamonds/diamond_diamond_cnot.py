from diamond import *
import sys

# -- 1: z = π/2 -------- 2: z = π/3 ---
#                  8:C1c
#        10:T1c    |     11:T2c
#         /        9:C2c   \
#        /                  \
#       0:C1a              4:C1b
# 2:T1a  |   3:T2a –– 6:T1b |    7:T2b
#       1:C2a               5:C2b
#         \                 /
#          \      12:C1d   /
#         14:T1d   |    15:T2d
#                 13:C2d
#        8
#     10   11
#     /  9  \
#   0        4
#  2 3––––––6 7
#   1        5
#    \  12  /
#     14  15
#       13

n = 4
N = n*4
U = lambda z: UN(*([z]*n))
SWAP = lambda n,m: iswap(N,[n,m])
SAVE = SWAP(0,10) * SWAP(4,11) * SWAP(1,14) * SWAP(5,15) * psi_m(N,8,9) * psi_m(N,12,13)
REST = SAVE.dag()

basic_entanglement = lambda n: X(N,n+1) * X(N,n+0) * H(N,n+2) * U(π) * H(N,n+3) * X(N,n+1) * X(N,n+0)
basic_entanglement_not = lambda n: H(N,n+2) * U(π) * H(N,n+3)

CNOT_23 = U(π) * basic_entanglement(0)
CNOT_32 = basic_entanglement(0) * U(π)
CNOT_67 = U(π) * basic_entanglement(4)
CNOT_76 = basic_entanglement(4) * U(π)
CNOT_62 = SWAP36(N) * CNOT_32 * SWAP36(N)
CNOT_26 = SWAP36(N) * CNOT_23 * SWAP36(N)
CNOT_73 = U(π) * CNOT_62 * U(π)
CNOT_37 = U(π) * CNOT_26 * U(π)


def pre_post_swap(oper, n, m):
    return SWAP(n,m) * REST * oper * SAVE * SWAP(n,m)


# C also CNOT
CNOT_03 = pre_post_swap(CNOT_23,2,0)
CNOT_13 = pre_post_swap(CNOT_23,2,1)
CNOT_20 = pre_post_swap(CNOT_23,3,0)
CNOT_21 = pre_post_swap(CNOT_23,3,1)


# TODO: MORE OPER HERE

opers = [
    ('CNOT_23', REST * CNOT_23 * SAVE),
    ('CNOT_32', REST * CNOT_32 * SAVE),
    ('CNOT_67', REST * CNOT_67 * SAVE),
    ('CNOT_76', REST * CNOT_76 * SAVE),
    ('CNOT_62', REST * CNOT_62 * SAVE),
    ('CNOT_26', REST * CNOT_26 * SAVE),
    ('CNOT_73', REST * CNOT_73 * SAVE),
    ('CNOT_37', REST * CNOT_37 * SAVE),
    ('CNOT_03', CNOT_03),
    ('CNOT_13', CNOT_13),
    ('CNOT_20', CNOT_20),
    ('CNOT_21', CNOT_21)
]


for i, oper in enumerate(opers):
    print('Operation:',oper[0])
    apply_to_basis(oper[1], four_diamond_basis, print_non_changed=True, print_states=False)
    print()
