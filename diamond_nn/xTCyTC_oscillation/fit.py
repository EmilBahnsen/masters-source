# DO: conda activate TFQ

import tensorflow as tf
from tf_qc import complex_type, float_type
from tf_qc.qc import iSWAP, U3
from tf_qc.layers import ISWAPLayer, _uniform_theta
import cloudpickle
from diamond_nn.datasets import circle_in_plain
from typing import *
import sympy as sp
import sympy_diamond as spd
# from tensorflow.math import conj as conjugate

conjugate = tf.math.conj

# Two 1-qubit tensor state embedding with iSWAP
# a = sp.symbols('a:12', real=True)
# o1 = sp.kronecker_product(spd.U3(a[0], a[1], 0), spd.U3(a[2], a[3], 0))
# o2 = spd.iSWAP(a[4])
# o3 = sp.kronecker_product(spd.U3(a[5], a[6], a[7]), spd.U3(a[8], a[9], a[10]))
# _state_lambda2 = sp.lambdify(a[:11], (o3 @ o2 @ o1 @ spd.s00), modules='tensorflow')
# def embed_two_tensor_qubits_U3_ISWAP_U3(*args):
#     return tf.convert_to_tensor(_state_lambda2(*args), complex_type)

def embed_two_tensor_qubits_U3_ISWAP_U3(*args):
    zero0 = tf.zeros(tf.shape(args[0]), complex_type)
    zero1 = tf.zeros(tf.shape(args[2]), complex_type)
    print(zero0)
    o1 = U3([args[0], args[1], zero0], [args[2], args[3], zero1])
    print(o1)
    o2 = iSWAP(args[4])
    o3 = U3([args[5], args[6], args[7]], [args[8], args[9], args[10]])
    return o3 @ o2 @ o1 @ s00


with open(r"analytic_P.pickle", "rb") as input_file:
    P00, P01, P10, P11 = cloudpickle.load(input_file)

# test_x = tf.ones((30**2, 2**4, 1), dtype=complex_type)
# test_y = embed_two_tensor_qubits_U3_ISWAP_U3(test_x, test_x, test_x, test_x, test_x, test_x, test_x, test_x, test_x, test_x, test_x)
# print(*test_y)
# print(tf.shape(test_y))
# exit()

class NN(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.w = tf.Variable(_uniform_theta((11,)))
        self.t = tf.Variable(_uniform_theta((2,)))

    def call(self, inputs, training=None, mask=None):
        x0 = inputs[..., 0]
        x1 = inputs[..., 1]
        x_state = embed_two_tensor_qubits_U3_ISWAP_U3(x0, x1, x0, x0, x0, x0, x0, x0, x0, x0, x0)
        w_state = embed_two_tensor_qubits_U3_ISWAP_U3(*self.w)
        return P00(*x_state,
                   *w_state,
                   0, 0, 0, 0,
                   *self.t)

# Data
plain_x, plain_labels = circle_in_plain()
plain_x = tf.cast(plain_x, complex_type)

# model = NN()

def model(inputs):
    x0 = inputs[..., 0]
    x1 = inputs[..., 1]
    izero = tf.constant(0., complex_type)
    zero = tf.constant(0., complex_type)
    x_state = embed_two_tensor_qubits_U3_ISWAP_U3(x0, x1, x0, x0, tf.cast(x0, float_type), x0, x0, x0, x0, x0, x0)
    w_state = embed_two_tensor_qubits_U3_ISWAP_U3(x0, x1, x0, x0, tf.cast(x0, float_type), x0, x0, x0, x0, x0, x0)
    return P00(tf.cast(x_state[0], float_type), x_state[1], x_state[2], x_state[3],
               tf.cast(w_state[0], float_type), w_state[1], w_state[2], w_state[3],
               zero, izero, izero, izero,
               zero, zero)

model(plain_x)

# !!! The functionality in cirq+tfq is not for custom gates without decompositon (as well as decomp. of CCCX etc.)
# I.e. only simple gates are implemented for tfq, so it is useless at the moment!