from diamond import *
from qutip.qip.circuit_latex import _latex_compile
import glob
import shutil
import os
import sys

def add_gates(qc, gate_list):
    for x in gate_list:
        qc.add_gate(*x)

n = 4
N = n*4

def save_circuit(qc, filename):
    _latex_compile(qc.latex_code(), filename=filename)

save = DiamondCircuit(N)
save.add_gate(r'\psi^{-}',[12,13])
save.add_gate(r'\psi^{-}',[8,9])
save.add_gate('SWAP',[5,15])
save.add_gate('SWAP',[1,14])
save.add_gate('SWAP',[4,11])
save.add_gate('SWAP',[0,10])
#save_circuit(save, 'save')

rest = DiamondCircuit(N)
rest.add_gate('SWAP',[0,10])
rest.add_gate('SWAP',[4,11])
rest.add_gate('SWAP',[1,14])
rest.add_gate('SWAP',[5,15])
rest.add_gate(r'(\psi^{-})^{\dag}',[8,9])
rest.add_gate(r'(\psi^{-})^{\dag}',[12,13])
#save_circuit(rest, 'rest')

basic_entanglement = lambda n: [['X',n], ['X',n+1], ['H',n+3], [r'U(\pi)'], ['H',n+2], ['X',n+0], ['X',n+1]]

cnot_gate_list = {
    'cnot_23': [[r'\textrm{SAVE}'], *basic_entanglement(0), [r'U(\pi)'], [r'\textrm{SAVE}^{\dag}']],
    'cnot_67': [[r'\textrm{SAVE}'], *basic_entanglement(4), [r'U(\pi)'], [r'\textrm{SAVE}^{\dag}']],
    'cnot_32': [[r'\textrm{SAVE}'], [r'U(\pi)'], *basic_entanglement(0), [r'\textrm{SAVE}^{\dag}']],
    'cnot_76': [[r'\textrm{SAVE}'], [r'U(\pi)'], *basic_entanglement(4), [r'\textrm{SAVE}^{\dag}']],
}
cnot_gate_list['cnot_62'] = [[r'\textrm{SAVE}'], ['SWAP',[3,6]], *cnot_gate_list['cnot_32'][1:-2], ['SWAP', [3,6]], [r'\textrm{SAVE}^{\dag}']]
cnot_gate_list['cnot_26'] = [[r'\textrm{SAVE}'], ['SWAP',[3,6]], *cnot_gate_list['cnot_23'][1:-2], ['SWAP', [3,6]], [r'\textrm{SAVE}^{\dag}']]
cnot_gate_list['cnot_73'] = [[r'\textrm{SAVE}'], [r'U(\pi)'], *cnot_gate_list['cnot_62'][1:-2], [r'U(\pi)'], [r'\textrm{SAVE}^{\dag}']]
cnot_gate_list['cnot_37'] = [[r'\textrm{SAVE}'], [r'U(\pi)'], *cnot_gate_list['cnot_26'][1:-2], [r'U(\pi)'], [r'\textrm{SAVE}^{\dag}']]


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
cnot_gate_list['cnot_27'] = pre_post_swap(cnot_gate_list['cnot_07'],0,2)  # doubel swap
cnot_gate_list['cnot_36'] = pre_post_swap(cnot_gate_list['cnot_34'],4,6)

for name,gate_list in cnot_gate_list.items():
    qc = DiamondCircuit(N)
    add_gates(qc, gate_list)
    save_circuit(qc, name+'_compact')

cwd = os.getcwd()
for f in glob.glob('*.pdf'):
    shutil.copy(f, 'gates_pdf')
    os.remove(f)
for f in glob.glob('*.tex'):
    shutil.copy(f, 'gates_tex')
    os.remove(f)
