import pennylane as qml
import pennylane
from pennylane import numpy as np
from pennylane.variable import Variable

device = qml.device("default.qubit", wires=1)

def rotation(t):
    qml.RX(t / 2, wires=0)
    qml.RY(t / 2, wires=0)

@qml.qnode(device)
def mean_photon_gaussian(t):
    # if type(t) is Variable:
    #     matrix = np.array([
    #         [0, np.exp(1j * t.val)],
    #         [np.exp(1j * t.val), 0]
    #     ])
    # else:
    matrix = np.array([
        [0, (1 - 1j*t)/(t**2 + 1)],
        [(1j*t + 1)/(t**2 + 1), 0]
    ])

    rotation(t)
    qml.QubitUnitary(matrix, wires=0)

    return qml.expval(qml.PauliZ(0))

def cost(t):
    return mean_photon_gaussian(t)


opt = qml.AdagradOptimizer()
steps = 20
params = 0.015

#print('test', cost(params), cost(2*params))

for i in range(steps):
    # update the circuit parameters
    params = opt.step(cost, params)

    print("Cost after step {:5d}: {:8f}, params = {}".format(i + 1, cost(t=params), params))
