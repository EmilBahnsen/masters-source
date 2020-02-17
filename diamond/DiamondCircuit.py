from diamond.definitions import *
from qutip.qip.circuit_latex import _latex_compile
from functools import reduce
import math
from collections import defaultdict

def intersection(list1, list2):
    if list1 is None or list2 is None:
        return []
    return list(set(list1).intersection(list2))


class ReducibleCircuit(QubitCircuit):
    def __init__(self, N, input_states=None, output_states=None, reverse_states=True, user_gates=None):
        super().__init__(N, input_states, output_states, reverse_states, user_gates)
        self.reduction_ids = {'SWAP': 'SWAP'}

    def reduce(self, full=False):
        # Make copy of map and invert it also, to accmedate reverse orders of gates
        reduction_ids = self.reduction_ids.copy()
        reduction_ids.update({v: k for k, v in reduction_ids.items()})
        len_gates = len(self.gates)
        changed = False
        i = 0
        while i != len_gates:
            gate1: Gate = self.gates[i]
            # Stop if we don't have a reduction rule for this gate
            if gate1.name not in reduction_ids:
                i += 1
                continue
            j = i+1
            while j != len_gates:
                gate2: Gate = self.gates[j]
                # if gate2 is the one that cancels gate1, then rm them both
                if (reduction_ids[gate1.name] == gate2.name) and \
                        (len(gate1.targets or []) == len(intersection(gate1.targets, gate2.targets))) and \
                        (len(gate1.controls or []) == len(intersection(gate1.controls, gate2.controls))):
                    self.gates.pop(j)
                    self.gates.pop(i)
                    len_gates -= 2
                    i -= 1
                    changed = True
                    break
                # make sure that we always are at a point that allows us to rm a pair wo. destroying any ting
                if (len(gate2.targets or []) == 0 and len(gate2.controls or []) == 0) or \
                        len(intersection(gate1.targets, gate2.targets)) != 0 or \
                        len(intersection(gate1.controls, gate2.controls)) != 0 or \
                        len(intersection(gate1.targets, gate2.controls)) != 0 or \
                        len(intersection(gate1.controls, gate2.targets)) != 0:
                    break
                j += 1
            i += 1
        # If it must be fully reduced then we do it again if we made progress
        if full and changed:
            return self.reduce(True)
        else:
            return self

    def gate_count_dict(self):
        result = defaultdict(int)
        for g in self.gates:
            result[g.name] += 1
        return result



# --- Circuit ---
class DiamondCircuit(ReducibleCircuit):
    def __init__(self, N: int):
        if N % 4 != 0:
            raise Exception('N must be multiple of 4, got ' + str(N) + '.')
        gate_dict = {
            'X': lambda: sigmax(),
            'Y': lambda: sigmay(),
            'Z': lambda: sigmaz(),
            'H': lambda: hadamard_transform(1),
            r'U(\pi)': lambda: UN(*(N // 4 * [π])),
            r'U(\pi/2)': lambda: UN(*(N // 4 * [π / 2])),
            r'\psi^{-}': lambda: psi_m(2, 0, 1),
            r'(\psi^{-})^{\dag}': lambda: psi_m(2, 0, 1).dag()
        }
        super().__init__(N, user_gates=gate_dict, reverse_states=False)
        self.reduction_ids.update({
            r'\psi^{-}': r'(\psi^{-})^{\dag}'
        })


def add_gates(qc, gate_list):
    for x in gate_list:
        qc.add_gate(*x)


def save_circuit(qc, filename):
    _latex_compile(qc.latex_code(), filename=filename)


def qc2matrix(qc: QubitCircuit):
    return reduce(lambda x,y: y*x, qc.propagators() or [identity([2]*qc.N)])  # Important to apply in reverse order


# --- 4 Diamond Circuit ---
# -- a: z = π/2 -------- b: z = π/3 ---
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
N = 4*4

# Save circuit
save_gate_list = [[r'\psi^{-}',[12,13]], [r'\psi^{-}',[8,9]], ['SWAP',[5,15]], ['SWAP',[1,14]], ['SWAP',[4,11]], ['SWAP',[0,10]]]

# Reset circuit
rest_gate_list = [['SWAP',[0,10]], ['SWAP',[4,11]], ['SWAP',[1,14]], ['SWAP',[5,15]], [r'(\psi^{-})^{\dag}',[8,9]], [r'(\psi^{-})^{\dag}',[12,13]]]

basic_entanglement = lambda n: [['X',n], ['X',n+1], ['H',n+3], [r'U(\pi)'], ['H',n+2], ['X',n+0], ['X',n+1]]
basic_entanglement2 = lambda n: [['X',n], ['X',n+1], ['H',n+2], [r'U(\pi)'], ['H',n+3], ['X',n+0], ['X',n+1]]

cnot_gate_list = {
    'cnot_23': [['SAVE'], *basic_entanglement(0), [r'U(\pi)'], ['REST']],
    'cnot_67': [['SAVE'], *basic_entanglement(4), [r'U(\pi)'], ['REST']],
    'cnot_32': [['SAVE'], [r'U(\pi)'], *basic_entanglement(0), ['REST']],
    'cnot_76': [['SAVE'], [r'U(\pi)'], *basic_entanglement(4), ['REST']],
}
cnot_gate_list['cnot_62'] = [['SAVE'], ['SWAP',[3,6]], *cnot_gate_list['cnot_32'][1:-2], ['SWAP', [3,6]], ['REST']]
cnot_gate_list['cnot_26'] = [['SAVE'], ['SWAP',[3,6]], *cnot_gate_list['cnot_23'][1:-2], ['SWAP', [3,6]], ['REST']]
cnot_gate_list['cnot_73'] = [['SAVE'], [r'U(\pi)'], *cnot_gate_list['cnot_62'][1:-2], [r'U(\pi)'], ['REST']]
cnot_gate_list['cnot_37'] = [['SAVE'], [r'U(\pi)'], *cnot_gate_list['cnot_26'][1:-2], [r'U(\pi)'], ['REST']]


def pre_post_swap(gate_list, n, m):
    return [['SWAP',[n,m]], *gate_list, ['SWAP',[n,m]]]

# swap over 23
cnot_gate_list['cnot_03'] = pre_post_swap(cnot_gate_list['cnot_23'],2,0)
cnot_gate_list['cnot_13'] = pre_post_swap(cnot_gate_list['cnot_23'],2,1)
cnot_gate_list['cnot_20'] = pre_post_swap(cnot_gate_list['cnot_23'],3,0)
cnot_gate_list['cnot_21'] = pre_post_swap(cnot_gate_list['cnot_23'],3,1)
cnot_gate_list['cnot_01'] = pre_post_swap(cnot_gate_list['cnot_03'],3,1)  # double swap
cnot_gate_list['cnot_10'] = pre_post_swap(cnot_gate_list['cnot_13'],3,0)

# swap over 32
cnot_gate_list['cnot_02'] = pre_post_swap(cnot_gate_list['cnot_32'],3,0)
cnot_gate_list['cnot_12'] = pre_post_swap(cnot_gate_list['cnot_32'],3,1)
cnot_gate_list['cnot_30'] = pre_post_swap(cnot_gate_list['cnot_32'],2,0)
cnot_gate_list['cnot_31'] = pre_post_swap(cnot_gate_list['cnot_32'],2,1)

# swap over 67
cnot_gate_list['cnot_47'] = pre_post_swap(cnot_gate_list['cnot_67'],6,4)
cnot_gate_list['cnot_57'] = pre_post_swap(cnot_gate_list['cnot_67'],6,5)
cnot_gate_list['cnot_64'] = pre_post_swap(cnot_gate_list['cnot_67'],7,4)
cnot_gate_list['cnot_65'] = pre_post_swap(cnot_gate_list['cnot_67'],7,5)
cnot_gate_list['cnot_45'] = pre_post_swap(cnot_gate_list['cnot_47'],7,5)  # double swap
cnot_gate_list['cnot_54'] = pre_post_swap(cnot_gate_list['cnot_57'],7,4)

# swap over 76
cnot_gate_list['cnot_46'] = pre_post_swap(cnot_gate_list['cnot_76'],7,4)
cnot_gate_list['cnot_56'] = pre_post_swap(cnot_gate_list['cnot_76'],7,5)
cnot_gate_list['cnot_74'] = pre_post_swap(cnot_gate_list['cnot_76'],6,4)
cnot_gate_list['cnot_75'] = pre_post_swap(cnot_gate_list['cnot_76'],6,5)

# swap over 62
cnot_gate_list['cnot_42'] = pre_post_swap(cnot_gate_list['cnot_62'],6,4)
cnot_gate_list['cnot_52'] = pre_post_swap(cnot_gate_list['cnot_62'],6,5)
cnot_gate_list['cnot_60'] = pre_post_swap(cnot_gate_list['cnot_62'],2,0)
cnot_gate_list['cnot_61'] = pre_post_swap(cnot_gate_list['cnot_62'],2,1)
cnot_gate_list['cnot_40'] = pre_post_swap(cnot_gate_list['cnot_42'],2,0)  # double swap
cnot_gate_list['cnot_41'] = pre_post_swap(cnot_gate_list['cnot_42'],2,1)
cnot_gate_list['cnot_50'] = pre_post_swap(cnot_gate_list['cnot_52'],2,0)
cnot_gate_list['cnot_51'] = pre_post_swap(cnot_gate_list['cnot_52'],2,1)

# swap over 06
cnot_gate_list['cnot_06'] = pre_post_swap(cnot_gate_list['cnot_26'],2,0)
cnot_gate_list['cnot_16'] = pre_post_swap(cnot_gate_list['cnot_26'],2,1)
cnot_gate_list['cnot_24'] = pre_post_swap(cnot_gate_list['cnot_26'],6,4)
cnot_gate_list['cnot_25'] = pre_post_swap(cnot_gate_list['cnot_26'],6,5)
cnot_gate_list['cnot_04'] = pre_post_swap(cnot_gate_list['cnot_06'],6,4)  # double swap
cnot_gate_list['cnot_05'] = pre_post_swap(cnot_gate_list['cnot_06'],6,5)
cnot_gate_list['cnot_14'] = pre_post_swap(cnot_gate_list['cnot_16'],6,4)
cnot_gate_list['cnot_15'] = pre_post_swap(cnot_gate_list['cnot_16'],6,5)

# swap over 73
cnot_gate_list['cnot_43'] = pre_post_swap(cnot_gate_list['cnot_73'],7,4)
cnot_gate_list['cnot_53'] = pre_post_swap(cnot_gate_list['cnot_73'],7,5)
cnot_gate_list['cnot_70'] = pre_post_swap(cnot_gate_list['cnot_73'],3,0)
cnot_gate_list['cnot_71'] = pre_post_swap(cnot_gate_list['cnot_73'],3,1)
cnot_gate_list['cnot_63'] = pre_post_swap(cnot_gate_list['cnot_43'],4,6)  # double swap
cnot_gate_list['cnot_72'] = pre_post_swap(cnot_gate_list['cnot_70'],0,2)

# swap over 37
cnot_gate_list['cnot_07'] = pre_post_swap(cnot_gate_list['cnot_37'],3,0)
cnot_gate_list['cnot_17'] = pre_post_swap(cnot_gate_list['cnot_37'],3,1)
cnot_gate_list['cnot_34'] = pre_post_swap(cnot_gate_list['cnot_37'],7,4)
cnot_gate_list['cnot_35'] = pre_post_swap(cnot_gate_list['cnot_37'],7,5)
cnot_gate_list['cnot_27'] = pre_post_swap(cnot_gate_list['cnot_07'],0,2)  # double swap
cnot_gate_list['cnot_36'] = pre_post_swap(cnot_gate_list['cnot_34'],4,6)


class FourDiamondCircuit(DiamondCircuit):
    def __init__(self):
        super().__init__(4*4)

    def add_gate(self, gate, targets=None, controls=None, arg_value=None,
                 arg_label=None, index=None):
        if gate == 'CNOT':
            name = 'cnot_' + str(controls) + str(targets)
            add_gates(self, cnot_gate_list[name])
        elif gate == 'SAVE':
            add_gates(self, save_gate_list)
        elif gate == 'REST':
            add_gates(self, rest_gate_list)
        # Controlled R_k for QFT
        elif gate == 'Rk':
            k = arg_value
            theta = math.pi/2**k
            super().add_gate('RZ', controls, arg_value=theta, arg_label=r'\pi/2^{}'.format(k))
            self.add_gate('CNOT',targets,controls)
            super().add_gate('RZ', targets, arg_value=-theta, arg_label=r'\pi/2^{}'.format(k))
            self.add_gate('CNOT', targets, controls)
            super().add_gate('RZ', targets, arg_value=theta, arg_label=r'\pi/2^{}'.format(k))
        elif gate == 'FFT':
            self.add_gate('H', 0)  # First qubit
            for i in targets[:-1]:
                for j in targets[i+1:]:
                    self.add_gate('Rk', i, j, arg_value=j - (i - 1))
            self.add_gate('H', targets[-1])  # Last qubit
        else:
            super().add_gate(gate, targets, controls, arg_value, arg_label, index)
