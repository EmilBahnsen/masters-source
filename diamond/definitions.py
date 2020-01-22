import qutip.settings
from qutip import *
from math import *
import numpy as np

π = pi

# Basis states
I1 = qeye([2])
I2 = qeye([2,2])
I4 = tensor(I2,I2)
I8 = tensor(I4,I4)
s0 = basis(2,0)         # |0>
s1 = basis(2,1)         # |1>
s00 = tensor(s0,s0)     # |00>
s01 = tensor(s0,s1)     # |01>
s10 = tensor(s1,s0)     # |10>
s11 = tensor(s1,s1)     # |11>
zero = lambda N: ket('0'*N)
zero4 = zero(4)
zero8 = zero(8)

# Bell states
sp = (s01 + s10)/sqrt(2)      # (|01> + |10>)/√2
sm = (s01 - s10)/sqrt(2)      # (|01> - |10>)/√2

# Basic opers
X = lambda N,n: gate_expand_1toN(sigmax(),N,n)
Y = lambda N,n: gate_expand_1toN(sigmay(),N,n)
Z = lambda N,n: gate_expand_1toN(sigmaz(),N,n)
H = lambda N,n: gate_expand_1toN(hadamard_transform(1),N,n)
__phi_p = lambda N,n,m: cnot(N,n,m) * H(N,n)
__phi_m = lambda N,n,m: Z(N,n) * __phi_p(N,n,m)
psi_p = lambda N,n,m: X(N,m) * __phi_p(N,n,m)
psi_m = lambda N,n,m: Z(N,n) * psi_p(N,n,m)
# Tests
# print(X(1,0) == sigmax())
# print(tensor(sigmax(),I1) == X(2,0))
# print(tensor(I1,sigmax()) == X(2,1))
# print(tensor(I1,sigmax(),I1) == X(3,1))


def u3(tx, ty, tz, N = None, target = 0):
    """
    U3 gate
    :param tx:
    :param ty:
    :param tz:
    :param N:
    :param target:
    :return:
    """
    if N is not None:
        return gate_expand_1toN(u3(tx, ty, tz), N, target)
    else:
        return Qobj([[np.cos(tx / 2), -np.exp(1j * tz) * np.sin(tx / 2)],
                     [np.exp(1j * ty) * np.sin(tx / 2), np.exp(1j * (tz + ty)) * np.cos(tx / 2)]])


# --- Construct U ---
# z = t_g · J_C
def U(z: float) -> Qobj:
    U00 = Qobj([[1,0,0,0],
                [0, (np.exp(-1j*z)+1)/2, (np.expm1(-1j*z))/2, 0],
                [0, np.expm1(-1j*z)/2, (np.exp(-1j*z)+1)/2, 0],
                [0,0,0,np.exp(-1j*z)]],dims=[[2,2],[2,2]])
    U11 = Qobj([[np.exp(1j*z),0,0,0],
                [0, (np.exp(1j*z)+1)/2, np.expm1(1j*z)/2, 0],
                [0, np.expm1(1j*z)/2, (np.exp(1j*z)+1)/2, 0],
                [0,0,0,1]],dims=[[2,2],[2,2]])
    Up = Qobj([[np.exp(1j*z),0,0,0],
               [0,1,0,0],
               [0,0,1,0],
               [0,0,0,np.exp(-1j*z)]],dims=[[2,2],[2,2]])
    Um = Qobj([[1,0,0,0],
               [0,1,0,0],
               [0,0,1,0],
               [0,0,0,1]],dims=[[2,2],[2,2]])

    return tensor(s00*s00.dag(), U00) +\
           tensor(s11*s11.dag(), U11) +\
           tensor(sp*sp.dag(), Up) +\
           tensor(sm*sm.dag(), Um)


def UN(*angles: float):
    return tensor(*[U(angle) for angle in angles])


U2 = lambda angle: UN(angle,angle)


SWAP36 = lambda N: swap(N,[3,6])
iSWAP36 = lambda N: swap(N,[3,6])

def _diamond_cnot_gate_list(control:int, target:int):
    return None


def diamond_cnot(control:int, target:int) -> Qobj:
    return None


def index2label(index:int,digits:[int],spacing=None):
    label = ''
    for n in np.arange(digits-1,-1,-1):
        if spacing is not None and spacing is not 0 and n != digits-1 and (n+1)%spacing == 0:
            label += ' '
        if index - 2 ** n >= 0:
            label += '1'
            index -= 2 ** n
        else:
            label += '0'
    return label


def format_coef(amp: complex):
    return round(amp.real,5) + round(amp.imag,5)*1j


def state2string(state: Qobj, spacing=None) -> str:
    occ_idx = state.data.tocoo().row
    amp = state.data.tocoo().data

    states = []
    N = int(log2(state.shape[0]))
    # if it's only a coeff. that's left, then just return that
    if N == 0:
        return str(format_coef(amp[0]))
    for i, coef in zip(occ_idx, amp):
        coef = format_coef(coef)
        if coef.imag == 0:
            coef = coef.real
            if coef == 1:
                coef = ''
            elif coef == -1:
                coef = '-'
        states.append(str(coef) + '|' + index2label(i, N, spacing) + '⟩')
    return ' + '.join(states).replace('+ -', '- ')


def string2state(N:int,string:str):
    state_string = string.replace(' ','')
    return ket(state_string, N)


def diamond2numericBasis(state:Qobj):
    if len(state.dims[0]) % 4 is not 0:
        print('not multiple of 4: not whole nuwmber of diamonds')
        exit(-1)
    n_diamonds = len(state.dims[0]) // 4
    basis_change_matrix = Qobj([[i for i in range(2 ** 4)]], dims=[[16], [2, 2, 2, 2]])
    matrix = tensor(*[basis_change_matrix for _ in range(n_diamonds)])
    print(basis_change_matrix)
    print(matrix)
    return matrix*state


class DiamondState(Qobj):
    def __init__(self, inpt):
        if isinstance(inpt, str):
            Qobj.__init__(self, ket(inpt.replace(' ', '')))
        else:
            Qobj.__init__(self, inpt)

    def __str__(self):
        return state2string(self, spacing=4)

    def __mul__(self, other):
        return DiamondState(super().__mul__(other))

    def __rmul__(self, other):
        if isinstance(other, Qobj):
            return DiamondState(other.__mul__(self))
        else:
            return DiamondState(super().__rmul__(other))

    def __truediv__(self, other):
        return DiamondState(super().__truediv__(other))

    def __neg__(self):
        return DiamondState(super().__neg__())

    # def __eq__(self, other):
    #     # TODO: Equals SKAL IMPLEMENTERS SÅ DEN KAN BRUGES TIL AT SAMMENLIGNED mine cnot's med qutips
    #     pass


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


diamond_basis = lambda n: [DiamondState(index2label(i, n)) for i in range(2 ** n)]
one_diamond_basis = [DiamondState('00' + index2label(i, 2)) for i in range(2 ** 2)]
# Keep the four last qubits (diamond c and d) zero
four_diamond_basis = [DiamondState(index2label(i, 8) + '0' * 8) for i in range(2 ** 8)]


def apply_to_basis(oper: Qobj, basis_states: [Qobj], print_non_changed=True, print_states=True):
    changed_count = 0
    for state in basis_states:
        state_new = oper * state
        # Print if it is unchanged (assuming orthornormal basis)
        inner_prod = state_new.dag()*state
        if abs(inner_prod) < 0.99:
            if print_states:
                print(state, '->', bcolors.BOLD, state_new, bcolors.ENDC)
            changed_count += 1
        elif print_non_changed:
            if print_states:
                print(state, '-> ', state_new)
    print('{} of {} changed'.format(changed_count, len(basis_states)))



#print(state2string(4, ket('0010') + ket('0011')/sqrt(2)))

# print(ket([15,12], 16))
# print(diamond2numericBasis(ket('1001')))
# print(diamond2numericBasis(ket('11111111')))

# U2 = lambda t: UN(t,t)
# state = ket('00100000')
# for n in range(1,100):
#     state = swap(8,[3,6]) * U2(π/n) * state
# print(state2string(8,state))
# fig,ax  = plot_fock_distribution(state * state.dag())
# ax.set_xlim(0,10)
# fig.show()

# print(state2string(psi_p(8,0,1).dag()*UN(π,π)*psi_p(8,0,1) * (ket('00100010') + zero8)/sqrt(2)))