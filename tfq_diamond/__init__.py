# DO: conda activate TFQ

import tensorflow_quantum as tfq
import tensorflow as tf
import tensorflow_quantum as tfq
import sympy as sp

import cirq
import numpy as np
from math import sqrt
from diamond_nn.datasets import circle_in_plain

π = np.pi

# Basis states
s0 = np.array([
    [1],
    [0]
])
s1 = np.array([
    [0],
    [1]
])
s00 = np.kron(s0, s0)
s01 = np.kron(s0, s1)
s10 = np.kron(s1, s0)
s11 = np.kron(s1, s1)

# Bell states
_sp = (s01 + s10)/sqrt(2)      # (|01> + |10>)/√2
_sm = (s01 - s10)/sqrt(2)      # (|01> - |10>)/√2


class U(cirq.Gate):
    def __init__(self, t):
        super().__init__()
        self.t = t

    def _num_qubits_(self):
        return 4

    def _unitary_(self):
        t = self.t
        U00 = np.array([
            [1, 0, 0, 0],
            [0, (np.exp(-1j * t) + 1) / 2, np.expm1(-1j * t) / 2, 0],
            [0, np.expm1(-1j * t) / 2, (np.exp(-1j * t) + 1) / 2, 0],
            [0, 0, 0, np.exp(-1j * t)]
        ])
        U11 = np.array([
            [np.exp(1j * t), 0, 0, 0],
            [0, (np.exp(1j * t) + 1) / 2, np.expm1(1j * t) / 2, 0],
            [0, np.expm1(1j * t) / 2, (np.exp(1j * t) + 1) / 2, 0],
            [0, 0, 0, 1]
        ])
        Up = np.array([
            [np.exp(1j * t), 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, np.exp(-1j * t)]
        ])
        Um = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        return np.kron(s00 * s00.transpose().conjugate(), U00) + \
               np.kron(s11 * s11.transpose().conjugate(), U11) + \
               np.kron(_sp * _sp.transpose().conjugate(), Up) + \
               np.kron(_sm * _sm.transpose().conjugate(), Um)

    def _json_dict_(self):
        return {
            'cirq_type': 'U',
            'apply_time': {
                'cirq_type': 'sympy.Symbol',
                'name': 't'
            }
        }

    def _from_json_dict_(self, dict):
        return U(dict['apply_time'])

    def _decompose_(self, qubits):
        c = cirq.Circuit()
        qubits = cirq.GridQubit.rect(2, 2)
        C1, C2, T1, T2 = qubits
        def A_block():
            pass

        def B_block():
            pass

        def C_block():
            pass

        def D_block():
            pass

        c.append(cirq.X(T2).controlled_by(C1, C2, T1))
        return  c


def gamma_encode(circuit, q0, q1, *args):
    circuit.append(cirq.rx(args[0]).on(q0))
    circuit.append(cirq.ry(args[1]).on(q0))
    circuit.append(cirq.rx(args[2]).on(q1))
    circuit.append(cirq.ry(args[3]).on(q1))
    circuit.append(cirq.ISwapPowGate(exponent=args[4]/π).on(q0, q1))
    circuit.append(cirq.rx(args[5]).on(q0))
    circuit.append(cirq.ry(args[6]).on(q0))
    circuit.append(cirq.rz(args[7]).on(q0))
    circuit.append(cirq.rx(args[8]).on(q1))
    circuit.append(cirq.ry(args[9]).on(q1))
    circuit.append(cirq.ry(args[10]).on(q1))


def encode_to_circuit(point):
    qubits = cirq.devices.GridQubit.rect(2, 2)
    C1, C2, T1, T2 = qubits
    circuit = cirq.Circuit()
    x0 = point[0]
    x1 = point[1]
    gamma_encode(circuit, C1, T1, x0, x1, π/2, π/2, π/4, π/2, π/2, π/2, π/2, π/2, π/2)
    return circuit


if __name__ == '__main__':
    plain_x, labels = circle_in_plain()
    # Make train/test-split (80%/20%)
    split_idx = int(len(plain_x)//1.25 - 1)
    train_x, test_x = plain_x[:split_idx], plain_x[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]

    x_train_circ = [encode_to_circuit(x) for x in train_x]
    x_test_circ = [encode_to_circuit(x) for x in test_x]

    # Convert input-data circuits to tensors
    x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
    x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)

    # Make the rest of the circuit that applies the weights and U-gate
    qubits = cirq.GridQubit.rect(2, 2)
    C1, C2, T1, T2 = qubits
    model_circuit = cirq.Circuit()

    # First set of weights: w
    w = sp.symbols('w:11')
    gamma_encode(model_circuit, C2, T2, *w)
    # Apply U-gate
    t = sp.symbols('t')
    u = cirq.decompose(U(t).on(qubits[0], qubits[1]))
    print(cirq.to_json(u))
    model_circuit.append(u)


    # Assign readout qubits
    model_readout = cirq.Z(C2)

    # Make parametrized circuit model
    poc = tfq.layers.PQC(model_circuit, model_readout)

    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        poc
    ])

