from qutip import *
from math import *
import numpy as np
import matplotlib.pyplot as plt
import itertools
import functools
import operator
import random
from joblib import Parallel, delayed
from diamond import *

# Basis states
I1 = qeye([2])
I2 = qeye([2,2])
I4 = tensor(I2,I2)
s0 = basis(2,0)         # |0>
s1 = basis(2,1)         # |1>
s00 = tensor(s0,s0)     # |00>
s01 = tensor(s0,s1)     # |01>
s10 = tensor(s1,s0)     # |10>
s11 = tensor(s1,s1)     # |11>

# Bell states
sp = (s01 + s10)/sqrt(2)      # (|01> + |10>)/√2
sm = (s01 - s10)/sqrt(2)      # (|01> - |10>)/√2

def index2label(index,digits=4):
    label = ''
    for n in np.arange(digits-1,-1,-1):
        if index - 2 ** n >= 0:
            label += '1'
            index -= 2 ** n
        else:
            label += '0'
    return label

def U(z):
    U00 = Qobj([[1,0,0,0],
                [0, np.exp(-1j*z)/2+1/2, np.exp(-1j*z)/2-1/2, 0],
                [0, np.exp(-1j*z)/2-1/2, np.exp(-1j*z)/2+1/2, 0],
                [0,0,0,np.exp(-1j*z)]],dims=[[2,2],[2,2]])
    U11 = Qobj([[np.exp(1j*z),0,0,0],
                [0, np.exp(1j*z)/2+1/2, np.exp(1j*z)/2-1/2, 0],
                [0, np.exp(1j*z)/2-1/2, np.exp(1j*z)/2+1/2, 0],
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

U_pi = U(np.pi)
fig,_ = matrix_histogram(U_pi)
fig.savefig('fig.pdf')
fig,_ = matrix_histogram_complex(U_pi)
fig.savefig('fig_c.pdf')
fig,_ = hinton(U_pi)
fig.savefig('fig_hinton.pdf')
indecies = range(15)
angles = np.arange(0,np.pi,np.pi/100)

fig = plt.figure()
def make_U_gate_figures():
    c_states = [*[ket(index2label(i,2)) for i in range(4)], *[bell_state('10'), bell_state('11')]]
    t_states = [ket(index2label(i, 2)) for i in range(4)]
    for init_state_c in c_states:
        for init_state_t in t_states:
            init_state = tensor(init_state_c, init_state_t)
            states = [U(z) * init_state for z in angles]
            def get_state(m):
                return [states[n][m][0][0] for n in range(len(angles))]

            label_c = state2string(init_state_c)
            label_t = state2string(init_state_t)
            title = 'U' + label_c + '_' + label_t
            fig.clf()
            ax = fig.gca()
            norm_sum = np.zeros(len(angles))
            for index in indecies:
                states_amps = np.abs(get_state(index))**2
                norm_sum += states_amps
                plt.plot(angles, states_amps)
                mitter_x = angles[len(angles) // 3]
                mitter_y = states_amps[len(states_amps) // 3]
                plt.text(mitter_x, mitter_y, index2label(index))
            plt.plot(angles, norm_sum, '--')
            plt.title(title)
            ax.set_xticks([0,np.pi/2,np.pi])
            ax.set_xticklabels(['0', 'π/2', 'π'])
            plt.xlabel(r'$tζ$')
            plt.ylabel(r'$|a|^2$')
            fig.savefig(title + '.pdf')
make_U_gate_figures()

# Fock state plot
def do_transform(oper, target):
    state_new = oper * ket('0000')

    fig = plt.figure(figsize=(10,5))
    indecies_plot = [index2label(index) for index in indecies]
    state_plot = np.abs([state_new[n][0][0] for n in indecies])**2
    plt.bar(indecies_plot, state_plot)
    plt.title(target)
    plt.xlabel(r'state')
    plt.ylabel(r'$|a|^2$')
    fig.savefig('fock_'+target+'.pdf')
    fig.show()

X1 = tensor(sigmax(),I1,I1,I1)
X2 = tensor(I1,sigmax(),I1,I1)
X3 = tensor(I1,I1,sigmax(),I1)
X4 = tensor(I1,I1,I1,sigmax())
T1 = tensor(phasegate(pi/8),I1,I1,I1)
T2 = tensor(I1,phasegate(pi/8),I1,I1)
T3 = tensor(I1,I1,phasegate(pi/8),I1)
T4 = tensor(I1,I1,I1,phasegate(pi/8))
H1 = tensor(hadamard_transform(1),I1,I1,I1)
H2 = tensor(I1,hadamard_transform(1),I1,I1)
H3 = tensor(I1,I1,hadamard_transform(1),I1)
H4 = tensor(I1,I1,I1,hadamard_transform(1))
#do_transform(U(pi/2) * X1 * U(pi/2) * X2, '1100')
#do_transform(U(pi/2) * X3 * U(pi/2) * X4, '0011')
#do_transform(H4 * X1 * U(pi/2) * X1 * H4, '?')

def random_product(*args, repeat=1):
    "Random selection from itertools.product(*args, **kwds)"
    pools = [tuple(pool) for pool in args] * repeat
    return tuple(random.choice(pool) for pool in pools)

# Combinatorics search
basic_opers = [X1,X2,X3,X4,T1,T2,T3,T4,H1,H2,H3,H4]
U_opers = [U(theta) for theta in np.arange(0,pi,pi/50)]
opers = [*basic_opers, *U_opers]
oper_combis = itertools.product(opers, repeat=8)
init = ket('0000')
targets = [(ket('0000') + ket('0110'))/sqrt(2),
           (ket('0000') + ket('1001'))/sqrt(2),
           (ket('0000') + ket('1010'))/sqrt(2),
           (ket('0000') + ket('0101'))/sqrt(2)]

def process(t):
    max_fid = 0
    oper_id = -1
    for i, oper_list in enumerate(oper_combis):
        oper = functools.reduce(operator.mul, oper_list)

        target = targets[t]
        fid = np.abs((target.dag() * oper * init)[0][0][0])**2
        if max_fid < fid:
            max_fid = fid
            oper_id = i
            print('t = {}: {}: [{}]'.format(t, fid, i))
    return {'t': t, 'max_fid': max_fid, 'oper_id': oper_id}


results = Parallel(n_jobs=len(targets))(delayed(process)(t) for t in range(len(targets)))
print(results)

f = open("result.txt","w")
f.write( str(results) )
f.close()

def print_seq(oper_list):
    for oper in oper_list:
        for o, op in enumerate(opers):
            if oper == op:
                print(o)


# for i, oper_list in enumerate(oper_combis):
#     if i == 504:
#         print('0110')
#         print('1001')
#         print('1010')
#         print('0101')
#         print_seq(oper_list)
#     elif i == 8828121:
#         print('0011')
#         print_seq(oper_list)
#     elif i == 8820431:
#         print('1100')
#         print_seq(oper_list)

