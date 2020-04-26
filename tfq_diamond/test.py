import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
from cirq.ops import CCX
from cirq import Simulator
import sympy
import numpy as np

# visualization tools
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit


simulator = Simulator()
q0 = cirq.GridQubit(0, 0)
q1 = cirq.GridQubit(1, 0)
q2 = cirq.GridQubit(2, 0)
q3 = cirq.GridQubit(3, 0)
rot_w_gate = cirq.X**sympy.Symbol('x')
circuit = cirq.Circuit()
circuit.append([rot_w_gate(q0), rot_w_gate(q1)])
for y in range(5):
    resolver = cirq.ParamResolver({'x': y / 4.0})
    result = simulator.simulate(circuit, resolver)
    print(np.round(result.final_state, 2))

