import pickle

from tf_qc.models import QCModel, U3Layer, ISWAPLayer, ULayer
from tf_qc import complex_type, float_type
from tf_qc.qc import tensor, s00, s0000, trace, measure, U3, iSWAP, gates_expand_toN, gate_expand_2toN, H
from tf_qc.layers import _uniform_0_1, _normal_0_1, ISWAPLayer, HLayer
from diamond_nn.datasets import normalize_data, two_circles_in_plain, checkerboard_in_plain
import tensorflow as tf
import numpy as np
from txtutils import ndtotext_print
import matplotlib.pyplot as plt
import datetime
import signal
import sys

s0000_real = tf.cast(s0000, float_type)

# This encodes data into a series of gates
class GammaEncodingModel(QCModel):
    def __init__(self, targets):
        super(GammaEncodingModel, self).__init__()
        # Makes sure that the phase of the init. U3 does something
        self.h1 = HLayer(targets[0])
        self.h2 = HLayer(targets[1])
        self.u3_1 = U3Layer(targets)
        self.iswap = ISWAPLayer(targets, parameterized=True)
        self.u3_2 = U3Layer(targets)
        # self.add(self.h1)
        # self.add(self.h2)
        self.add(self.u3_1)
        self.add(self.iswap)
        self.add(self.u3_2)
        self.targets = targets
        assert len(targets) == 2

    def build(self, input_shape=None):
        for l in self.layers:
            l.build(input_shape)

    def call(self, inputs, training=None, mask=None):
        return self.matrix() @ inputs


class GammaDataEncodingModel(QCModel):
    def __init__(self, targets, input_weights):
        super(GammaDataEncodingModel, self).__init__(layers=[
            tf.keras.Input((2,))
        ])
        self.input_weights = input_weights
        self.qubit_targets = targets
        # !!! THIS VAR IS NOT CHANGED DURING TRAINING !!!

    def call(self, inputs, training=None, mask=None):
        # Encode the variables into a state with Gamma-function
        def encode_x(x):
            # Hadamard makes sure that the phase of the init. U3 does something
            h_1_2 = gates_expand_toN([H, H], 4, self.qubit_targets)
            u3_1a = U3Layer.matrix_static([x[0], x[1], x[2]])
            u3_1b = U3Layer.matrix_static([x[3], x[4], x[5]])
            u1_1 = gates_expand_toN([u3_1a, u3_1b], 4, self.qubit_targets)

            iswap = ISWAPLayer.matrix_static(4, self.qubit_targets, x[6])

            u3_2a = U3Layer.matrix_static([x[7], x[8], x[9]])
            u3_2b = U3Layer.matrix_static([x[10], x[11], x[12]])
            u1_2 = gates_expand_toN([u3_2a, u3_2b], 4, self.qubit_targets)
            return u1_2 @ (iswap @ (u1_1 @ (s0000)))
            # return  u1_2 @ (iswap @ (u1_1 @ (h_1_2 @ s0000)))


        input_poly_elem = tf.convert_to_tensor([
            tf.ones(tf.shape(inputs)[0]),
            inputs[..., 0],
            inputs[..., 1],
            inputs[..., 0]**2,
            inputs[..., 1]**2,
            inputs[..., 0]*inputs[..., 1]
        ])
        input_poly_elem = tf.transpose(input_poly_elem)

        weighted_xs = tf.reduce_sum(tf.expand_dims(self.input_weights, axis=0) * tf.expand_dims(input_poly_elem, axis=1), axis=-1)
        states = tf.map_fn(encode_x, weighted_xs, dtype=complex_type, back_prop=True)
        return states  # tf.squeeze(states, axis=1)



class OnePassModel(QCModel):
    def __init__(self):
        # This encodes the input states to a circuit via the gamma encoder (IS trainable)
        input_weights = tf.Variable(_normal_0_1((13, 6), dtype=float_type),
                                    trainable=True,
                                    dtype=float_type,
                                    name='input_weights')
        input_encoder = GammaDataEncodingModel([0, 2], input_weights)  # C1 and T1
        # In this model C1 and T1 is the input and C2 and T2 is output and where the weights are applied
        w_layer = GammaEncodingModel([1, 3])  # C2 and T2
        # v_layer = GammaEncodingModel([0, 2])  # C1 and T1
        self.P_weights = tf.Variable(_normal_0_1((16,), dtype=float_type),
                                     trainable=True,
                                     dtype=float_type,
                                     name='P_weights')
        self.P_bias = tf.Variable(_normal_0_1((1,), dtype=float_type),
                                  trainable=True,
                                  dtype=float_type,
                                  name='P_bias')

        super(OnePassModel, self).__init__(layers=[
            tf.keras.layers.InputLayer((2,), dtype=float_type),
            input_encoder,
            w_layer,
            ULayer([0, 1, 2, 3]),
            # v_layer,
            # ULayer([0, 1, 2, 3])
            # Measurement of C2 and T2
        ])

    def call(self, inputs, training=None, mask=None):
        outputs = super().call(inputs)
        # print(outputs.shape)
        # Ps = tf.cast(measure(outputs, [1, 3]), float_type)
        Ps = tf.cast(measure(outputs, [0, 1, 2, 3]), float_type)
        # print(result.shape)
        # exit()
        # return Ps[..., 0]
        # Ps = 2*(Ps - 0.5)  # Map to the range [-1, +1]
        # return tf.reduce_sum(self.P_weights * Ps, axis=-1) + self.P_bias
        # print(self.P_weights.shape)
        # print(Ps.shape)
        # exit()
        # We can choose to cum all the probs for all diff. outputs 0000, 0001, ...
        # OR we can just measure one of the qubits, or two, or ... to have a more simple model
        out = tf.nn.sigmoid(
            tf.reduce_sum(self.P_weights * Ps, axis=-1) +
            self.P_bias
        )
        return tf.expand_dims(out, axis=-1)
        # return tf.nn.sigmoid(self.P_weights[0] * tf.expand_dims(Ps[..., 0], axis=-1) + self.P_bias)
        # return tf.expand_dims(Ps[..., 0], axis=-1)


class OnePassModelNoU(OnePassModel):
    def __init__(self):
        super(OnePassModelNoU, self).__init__()
        self.layers.pop(-1)  # Remove the U-layer to test the difference!


# This data goes into C1 and T1
plain_x, labels = checkerboard_in_plain(False)
plain_x = normalize_data(plain_x)
plain_x = tf.convert_to_tensor(plain_x, dtype=float_type)
labels = tf.convert_to_tensor(labels, dtype=float_type)

# --- Model and loss ---
# The model executes the alg. and then
# measure probabilities of 00, 01, 10, 11 for C2 and T2
tag = 'OnePassModelNoU'
model = OnePassModelNoU()

class P00MaximisationLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(P00MaximisationLoss, self).__init__()

    def call(self, y_true, y_pred):
        y_true = tf.squeeze(y_true)  # This is apparently not properly formatted
        y_pred = tf.squeeze(y_pred)
        P_sum = y_pred
        delta = P_sum - y_true
        abs_delta = tf.abs(delta)
        dist_to_true = abs_delta# + tf.nn.relu(abs_delta - 1) # abs loss + loss for being on the wrong side of 0!
        mean = tf.reduce_mean(dist_to_true)
        return mean


optimizer = tf.optimizers.Adam(0.001)
# loss = P00MaximisationLoss()
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
print(*model.trainable_variables, sep='\n')
# checkpoint = tf.keras.callbacks.ModelCheckpoint('fit_qcmodel_model/best.h5',
#                                                 verbose=1,
#                                                 monitor='val_loss',
#                                                 save_best_only=True,
#                                                 mode='auto',
#                                                 save_weights_only=True)
class KeepBestCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(KeepBestCallback, self).__init__()
        self.best_weights = None
        self.best_val_loss = 1

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs['val_loss']
        if self.best_weights is None:
            self.best_weights = self.model.weights
        elif val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_weights = self.model.weights


keep_best_callback = KeepBestCallback()

# signal.signal(signal.SIGINT, last_code)
with tf.device('cpu'):
    model.fit(plain_x, labels, validation_split=0.2, batch_size=50, epochs=1000, callbacks=[keep_best_callback])

    # --- Test result ---
    # model.load_weights('fit_qcmodel_model/best.h5')
    model = OnePassModel()
    for i, v in enumerate(model.variables):
        v.assign(keep_best_callback.best_weights[i])
    out_P00 = model(plain_x)
    x = plain_x.numpy()
    P00 = out_P00.numpy()
    # labels = labels.numpy()
    fig = plt.figure()
    plt.title('fit_qcmodel_pred')
    plt.scatter(x[:, 0], x[:, 1], c=P00.flatten(), cmap=plt.get_cmap('bwr'))
    plt.show()

    t = datetime.datetime.now().isoformat()
    with open(f'fit_qcmodel_fig/fit_qcmodel_figure_x_c_{tag}_{t}.pickle', 'bw') as f:
        pickle.dump([fig, x, P00.flatten()], f)

    # --- True Plot ---
    # plt.figure()
    # plt.title('true')
    # plt.scatter(x[:100, 0], x[:100, 1], c=labels[:100], cmap=plt.get_cmap('bwr'), vmin=-1, vmax=1)
    # plt.show()