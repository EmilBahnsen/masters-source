import tensorflow as tf
# Force to use the CPU
tf.config.experimental.set_visible_devices([], 'GPU')

import os
import pickle
from tf_qc.models import QCModel, U3Layer, ISWAPLayer, ULayer
from tf_qc import complex_type, float_type
from tf_qc.qc import tensor, s00, s0000, trace, measure, U3, RX, RY, RZ, iSWAP, gates_expand_toN, gate_expand_2toN, H, apply_operators2state
from tf_qc import qc
from tf_qc.layers import _uniform_0_1, _normal_0_1, ISWAPLayer, HLayer
import diamond_nn.datasets as my_datasets
from diamond_nn.datasets import normalize_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
from functools import reduce
from logtools import Logger
from sklearn import datasets
from diamond_nn import plot_decision_boundary, plot_decision_boundary2
from mlxtend.plotting import plot_decision_regions
import sys

π = np.pi

# sHHHH = tensor(4*[H]) @ s0000

class U3_U(tf.keras.Sequential):
    def __init__(self):
        input_dense = tf.keras.layers.Dense(2, name='input_dense',)
        # output_dense = tf.keras.layers.Dense(1, activation='sigmoid', name='output_linear_combi')
        # MNIST
        output_dense = tf.keras.layers.Dense(1, name='output_dense', activation='sigmoid')
        # MNIST

        # This defines the quantum part of the model
        class QC(tf.keras.Model):
            def __init__(self):
                init = tf.initializers.glorot_uniform()
                super(QC, self).__init__()
                self.wXY_1 = tf.Variable(init((8,), dtype=float_type),
                                         trainable=True,
                                         dtype=float_type, name='wXY_1')
                self.wXY_2 = tf.Variable(init((8,), dtype=float_type),
                                         trainable=True,
                                         dtype=float_type, name='wXY_2')
                self.wXY_3 = tf.Variable(init((8,), dtype=float_type),
                                         trainable=True,
                                         dtype=float_type, name='wXY_3')
                self.wXY_4 = tf.Variable(init((8,), dtype=float_type),
                                         trainable=True,
                                         dtype=float_type, name='wXY_4')
                self.wU3_1 = tf.Variable(init((12,), dtype=float_type),
                                         trainable=True,
                                         dtype=float_type, name='wU3_1')
                self.wU3_2 = tf.Variable(init((12,), dtype=float_type),
                                         trainable=True,
                                         dtype=float_type, name='wU3_2')
                self.wU3_3 = tf.Variable(init((12,), dtype=float_type),
                                         trainable=True,
                                         dtype=float_type, name='wU3_3')
                self.w_U_iswap = tf.Variable(init((4,), dtype=float_type),
                                      trainable=True,
                                      dtype=float_type, name='w_U_iswap')


            def call(self, inputs, training=None, mask=None):
                uXY_1 = tensor([
                    RX(self.wXY_1[0]) @ RZ(self.wXY_1[1]),
                    RX(self.wXY_1[2]) @ RZ(self.wXY_1[3]),
                    RX(self.wXY_1[4]) @ RZ(self.wXY_1[5]),
                    RX(self.wXY_1[6]) @ RZ(self.wXY_1[7])
                ])
                uXY_2 = tensor([
                    RX(self.wXY_2[0]) @ RZ(self.wXY_2[1]),
                    RX(self.wXY_2[2]) @ RZ(self.wXY_2[3]),
                    RX(self.wXY_2[4]) @ RZ(self.wXY_2[5]),
                    RX(self.wXY_2[6]) @ RZ(self.wXY_2[7])
                ])
                uXY_3 = tensor([
                    RX(self.wXY_3[0]) @ RZ(self.wXY_3[1]),
                    RX(self.wXY_3[2]) @ RZ(self.wXY_3[3]),
                    RX(self.wXY_3[4]) @ RZ(self.wXY_3[5]),
                    RX(self.wXY_3[6]) @ RZ(self.wXY_3[7])
                ])
                uXY_4 = tensor([
                    RX(self.wXY_4[0]) @ RZ(self.wXY_4[1]),
                    RX(self.wXY_4[2]) @ RZ(self.wXY_4[3]),
                    RX(self.wXY_4[4]) @ RZ(self.wXY_4[5]),
                    RX(self.wXY_4[6]) @ RZ(self.wXY_4[7])
                ])
                u3_1 = tensor([
                    U3(self.wU3_1[0:3]),
                    U3(self.wU3_1[3:6]),
                    U3(self.wU3_1[6:9]),
                    U3(self.wU3_1[9:12])
                ])
                u3_2 = tensor([
                    U3(self.wU3_2[0:3]),
                    U3(self.wU3_2[3:6]),
                    U3(self.wU3_2[6:9]),
                    U3(self.wU3_2[9:12])
                ])
                u3_3 = tensor([
                    U3(self.wU3_3[0:3]),
                    U3(self.wU3_3[3:6]),
                    U3(self.wU3_3[6:9]),
                    U3(self.wU3_3[9:12])
                ])
                u_1 = qc.U(self.w_U_iswap[0])
                u_2 = qc.U(self.w_U_iswap[1])
                u_3 = qc.U(self.w_U_iswap[2])
                u_4 = qc.U(self.w_U_iswap[3])
                # iswap_C1T1 = ISWAPLayer.matrix_static(4, [0, 2], self.w_U_iswap[0])
                # iswap_C1T2 = ISWAPLayer.matrix_static(4, [0, 3], self.w_U_iswap[1])
                # iswap_C2T1 = ISWAPLayer.matrix_static(4, [1, 2], self.w_U_iswap[2])
                # iswap_C2T2 = ISWAPLayer.matrix_static(4, [1, 3], self.w_U_iswap[3])
                # u_2 = qc.U(self.wU[1])
                # u_3 = qc.U(self.wU[2])
                def apply_qc_to_data(data):
                    opers = []
                    opers.append(tensor([
                        RX(data[0]) @ RY(π/4) @ RZ(π/4),
                        RX(data[1]) @ RY(π/4) @ RZ(π/4),
                        RX(data[0]) @ RY(π/4) @ RZ(π/4),
                        RX(data[1]) @ RY(π/4) @ RZ(π/4)
                    ]))
                    opers.append(u3_1)
                    opers.append(u_1)
                    opers.append(u3_2)
                    opers.append(u_2)
                    opers.append(u3_3)
                    opers.append(u_3)
                    # opers.append(uXY_4)
                    # opers.append(u_4)
                    return apply_operators2state(opers, s0000)

                outputs = tf.map_fn(apply_qc_to_data, inputs, dtype=complex_type, parallel_iterations=12)

                # Make a linearcombi. of the output probabilities
                # we measure all of them as combine them in linear combi. + bias and then apply sigmoid
                # print(outputs)
                Ps = tf.cast(measure(outputs, [0]), float_type)
                # print(Ps)
                return Ps

        super(U3_U, self).__init__(layers=[
            # input_dense,
            QC(),  # Thils works as an activation function
            output_dense
        ])


class KeepBestCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(KeepBestCallback, self).__init__()
        self.best_weights = None
        self.best_val_loss = 1
        self.epochs = 0
        self.best_epoch = -1

    def on_epoch_end(self, epoch, logs=None):
        self.epochs += 1
        val_loss = logs['val_loss']
        if self.best_weights is None:
            self.best_weights = self.model.get_weights()
        elif val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_weights = self.model.get_weights()
            self.best_epoch = self.epochs

    def on_train_end(self, logs=None):
        tf.print(f'Best val_loss was on epoch #{self.best_epoch}')


def main(*args):
    file_name = os.path.basename(os.path.splitext(__file__)[0])
    time = datetime.datetime.now().isoformat()
    data_idx = int(args[0])
    # data_idx = 6
    description = f'encoding_U3_U_U3_U_U3_U_measure0_data{data_idx}'
    # description = 'TEST'
    logger = Logger(log_dir=os.path.join(file_name, description, time))
    # logger.log_file(__file__)
    print('log_dir', logger.log_dir)

    with open(__file__) as this_file:
        this_file = this_file.read()
    # This data goes into C1 and T1
    plain_x, labels = my_datasets.article_2003_09887_data(data_idx)
    colors = ['blue' if label == 1 else 'red' for label in labels]
    plain_x = normalize_data(plain_x)
    plain_x = tf.convert_to_tensor(plain_x, dtype=float_type)
    labels = tf.convert_to_tensor(labels, dtype=float_type)

    # --- Feature extraction ---
    features = tf.convert_to_tensor([
        plain_x[..., 0],
        plain_x[..., 1],
    ])
    features = tf.transpose(features)

    # --- Model and loss ---
    model = U3_U()

    optimizer = tf.optimizers.Adam(0.01)
    # loss = P00MaximisationLoss()
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    # model.build((1, 2))
    # print(model.summary())
    print(*model.trainable_variables, sep='\n')

    keep_best_callback = KeepBestCallback()
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0.0005, patience=500, verbose=0, mode='auto',
        baseline=None, restore_best_weights=False)

    history = model.fit(features, labels,
                          validation_split=0.2,
                          epochs=5000,
                          callbacks=[keep_best_callback, early_stopping_callback],
                          use_multiprocessing=True)

    # --- Test result ---
    model_fit = model
    model_fit(features[:2,...])  # Phony init of model
    model_fit.set_weights(keep_best_callback.best_weights)
    for i in range(len(model_fit.weights)):
        model_fit.weights[i].assign(keep_best_callback.best_weights[i])
    # out = model_fit(features)
    # x = plain_x.numpy()
    # c_out = out.numpy().flatten()
    # # labels = labels.numpy()
    # fig = plt.figure()
    # plt.title(f'{description}')
    # plt.scatter(x[:, 0], x[:, 1], c=c_out, cmap=plt.get_cmap('bwr'))
    # plt.show()

    X = plain_x.numpy()
    lab = labels.numpy()
    int_labels = np.array(lab>.5, int)
    fig = plt.figure(figsize=(5, 5))
    plot_decision_regions(X=X, y=int_labels, clf=model_fit, legend=2)
    train_labels = model_fit.predict(X, use_multiprocessing=True)
    plt.title(description)
    plt.show()

    logger.log_variables(fig, 'fig',
                         X, 'X',
                         int_labels, 'y',
                         train_labels, 'train_labels',
                         keep_best_callback.best_weights, 'weights',
                         history.history, 'history')
    logger.log_figure(fig, 'fig.pdf')

if __name__ == '__main__':
    main(sys.argv[1:])