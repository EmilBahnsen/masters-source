import pennylane as qml

device = qml.device("default.qubit", wires=1)

@qml.qnode(device)
def circuit(params):
    qml.RX(params, wires=0)
    return qml.expval(qml.PauliZ(0))

# Loss function to end in eg. state -1
def cost(params):
    return circuit(params)

init_params = 0.011
print(cost(init_params))

# initialise the optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.4)

# set the number of steps
steps = 100
# set the initial parameter values
params = init_params

for i in range(steps):
    # update the circuit parameters
    params = opt.step(cost, params)

    if (i + 1) % 5 == 0:
        print("Cost after step {:5d}: {: .7f}".format(i + 1, cost(params)))

print("Optimized rotation angles: {}".format(params))