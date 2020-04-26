import tensorflow as tf
# Force to use the CPU
tf.config.experimental.set_visible_devices([], 'GPU')

import os
import pickle
from tf_qc.models import QCModel, U3Layer, ISWAPLayer, ULayer
from tf_qc import complex_type, float_type
from tf_qc.qc import tensor, s00, s0000, trace, measure, iSWAP, gates_expand_toN, gate_expand_2toN, H, apply_operators2state
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
from diamond_nn import plot_decision_boundary, plot_decision_boundary2, KeepBestCallback
from mlxtend.plotting import plot_decision_regions
import sys

π = np.pi

class Model(tf.keras.Model):
    def __init__(self, samples=1000):
        super(Model, self).__init__()
        self.samples = samples
        self.w_U = tf.Variable(2*π*_uniform_0_1((6,), float_type),
                               trainable=True,
                               dtype=float_type,
                               name='w_U')
        self.w_RXZX = tf.Variable(2*π*_uniform_0_1((6,4,3), float_type),
                                  trainable=True,
                                  dtype=float_type,
                                  name='w_RXZX')
        self.out_scale = tf.Variable(1, trainable=True, dtype=float_type, name='output_scale')
        self.out_bias = tf.Variable(1, trainable=True, dtype=float_type, name='output_bias')

    def call(self, inputs, training=None, mask=None):
        u_1 = ISWAPLayer.matrix_static(4, [1,2], self.w_U[0])
        u_2 = ISWAPLayer.matrix_static(4, [1,3], self.w_U[1])
        u_3 = ISWAPLayer.matrix_static(4, [1,2], self.w_U[2])
        u_4 = ISWAPLayer.matrix_static(4, [1,3], self.w_U[3])
        u_5 = ISWAPLayer.matrix_static(4, [0,3], self.w_U[4])
        u_6 = ISWAPLayer.matrix_static(4, [0,2], self.w_U[5])
        # u_1 = qc.U(self.w_U[0])
        # u_2 = qc.U(self.w_U[1])
        # u_3 = qc.U(self.w_U[2])
        # u_4 = qc.U(self.w_U[3])
        # u_5 = qc.U(self.w_U[4])
        # u_6 = qc.U(self.w_U[5])
        rxzx_1 = tensor([
            qc.RXZX(self.w_RXZX[0,0,0], self.w_RXZX[0,0,1], self.w_RXZX[0,0,2]),
            qc.RXZX(self.w_RXZX[0,1,0], self.w_RXZX[0,1,1], self.w_RXZX[0,1,2]),
            qc.RXZX(self.w_RXZX[0,2,0], self.w_RXZX[0,2,1], self.w_RXZX[0,2,2]),
            qc.RXZX(self.w_RXZX[0,3,0], self.w_RXZX[0,3,1], self.w_RXZX[0,3,2])
        ])
        rxzx_2 = tensor([
            qc.RXZX(self.w_RXZX[1, 0, 0], self.w_RXZX[1, 0, 1], self.w_RXZX[1, 0, 2]),
            qc.RXZX(self.w_RXZX[1, 1, 0], self.w_RXZX[1, 1, 1], self.w_RXZX[1, 1, 2]),
            qc.RXZX(self.w_RXZX[1, 2, 0], self.w_RXZX[1, 2, 1], self.w_RXZX[1, 2, 2]),
            qc.RXZX(self.w_RXZX[1, 3, 0], self.w_RXZX[1, 3, 1], self.w_RXZX[1, 3, 2])
        ])
        rxzx_3 = tensor([
            qc.RXZX(self.w_RXZX[2, 0, 0], self.w_RXZX[2, 0, 1], self.w_RXZX[2, 0, 2]),
            qc.RXZX(self.w_RXZX[2, 1, 0], self.w_RXZX[2, 1, 1], self.w_RXZX[2, 1, 2]),
            qc.RXZX(self.w_RXZX[2, 2, 0], self.w_RXZX[2, 2, 1], self.w_RXZX[2, 2, 2]),
            qc.RXZX(self.w_RXZX[2, 3, 0], self.w_RXZX[2, 3, 1], self.w_RXZX[2, 3, 2])
        ])
        rxzx_4 = tensor([
            qc.RXZX(self.w_RXZX[3, 0, 0], self.w_RXZX[3, 0, 1], self.w_RXZX[3, 0, 2]),
            qc.RXZX(self.w_RXZX[3, 1, 0], self.w_RXZX[3, 1, 1], self.w_RXZX[3, 1, 2]),
            qc.RXZX(self.w_RXZX[3, 2, 0], self.w_RXZX[3, 2, 1], self.w_RXZX[3, 2, 2]),
            qc.RXZX(self.w_RXZX[3, 3, 0], self.w_RXZX[3, 3, 1], self.w_RXZX[3, 3, 2])
        ])
        rxzx_5 = tensor([
            qc.RXZX(self.w_RXZX[4, 0, 0], self.w_RXZX[4, 0, 1], self.w_RXZX[4, 0, 2]),
            qc.RXZX(self.w_RXZX[4, 1, 0], self.w_RXZX[4, 1, 1], self.w_RXZX[4, 1, 2]),
            qc.RXZX(self.w_RXZX[4, 2, 0], self.w_RXZX[4, 2, 1], self.w_RXZX[4, 2, 2]),
            qc.RXZX(self.w_RXZX[4, 3, 0], self.w_RXZX[4, 3, 1], self.w_RXZX[4, 3, 2])
        ])
        rxzx_6 = tensor([
            qc.RXZX(self.w_RXZX[5, 0, 0], self.w_RXZX[5, 0, 1], self.w_RXZX[5, 0, 2]),
            qc.RXZX(self.w_RXZX[5, 1, 0], self.w_RXZX[5, 1, 1], self.w_RXZX[5, 1, 2]),
            qc.RXZX(self.w_RXZX[5, 2, 0], self.w_RXZX[5, 2, 1], self.w_RXZX[5, 2, 2]),
            qc.RXZX(self.w_RXZX[5, 3, 0], self.w_RXZX[5, 3, 1], self.w_RXZX[5, 3, 2])
        ])

        def apply_qc_to_data(fea):
            opers = []
            # Encoding is Z(acos(x**2)) * Y(asin(x)) on all qubits
            encoding_oper = qc.RZ(fea[0]) @ qc.RY(fea[1])
            opers.append(tensor([
                encoding_oper,
                encoding_oper,
                encoding_oper,
                encoding_oper
            ]))
            return apply_operators2state(opers, s0000)

        encoding = tf.map_fn(apply_qc_to_data, inputs, dtype=complex_type, parallel_iterations=12)
        W_circuit = [  # This is the parametrized circuit
            u_1, rxzx_1,
            u_2, rxzx_2,
            u_3, rxzx_3,
            u_4, rxzx_4,
            u_5, rxzx_5,
            u_6, rxzx_6
        ]
        outputs = apply_operators2state(W_circuit, encoding)

        # Measure Z on 1st qubit and add a bit of noise
        # https://en.wikipedia.org/wiki/Density_matrix#Measurement
        # The expectation value <Z> = tr(density_matrix @ Z_oper)
        # Z_oper = [[1,0], [0,-1]] so <Z> = density_matrix[0,0] - density_matrix[1,1]
        # But P0 - P1 = P0 - (1-P0) = 2*P0 - 1, so <Z> = 2*density_matrix[0,0] - 1
        # OK to take partial trace first: https://en.wikipedia.org/wiki/Partial_trace#Partial_trace_as_a_quantum_operation
        Z01 = measure(outputs, [0])
        Z_exp = 2*tf.cast(Z01[..., 0], float_type) - 1
        noise = tf.sqrt(2/self.samples) * (Z_exp**2 - 1)/4
        measurement = self.out_scale * (Z_exp + noise) + self.out_bias
        return measurement


def main(data_idx):
    # Data from -1 to 1
    data_generator = my_datasets.complex_1D_functions
    data_index = data_idx
    x, y = data_generator(data_index)

    file_name = os.path.basename(os.path.splitext(__file__)[0])
    time = datetime.datetime.now().isoformat()
    description = f'{data_generator.__name__}({data_index})_smart_iSWAP_rxzx_6'
    # description = 'TEST_U_bias'
    logger = Logger(log_dir=os.path.join(file_name, description, time))
    logger.log_file(__file__)
    print('log_dir', logger.log_dir)


    # --- Feature extraction ---
    def calc_features(x):
        features = tf.convert_to_tensor([  # Features as descried in the article
            np.arccos(x**2),
            np.arcsin(x),
        ])
        return tf.transpose(features)  # (batch, feature)

    features = calc_features(x)

    # --- Model and loss ---
    model = Model()

    optimizer = tf.optimizers.Adam(0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    print(*model.trainable_variables, sep='\n')

    keep_best_callback = KeepBestCallback()
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(min_delta=0.00001, patience=100)

    history = model.fit(features, y,
                        validation_split=0.2,
                        epochs=10000,
                        callbacks=[keep_best_callback],
                        use_multiprocessing=True,
                        verbose=2)

    # --- Test result ---
    model_fit = model
    model_fit(features[:10,...])  # Phony init of model
    model_fit.set_weights(keep_best_callback.best_weights)
    for i in range(len(model_fit.weights)):
        model_fit.weights[i].assign(keep_best_callback.best_weights[i])

    fig = plt.figure()
    x_fit = np.linspace(-1, 1, 1000)
    features_fit = calc_features(x_fit)
    y_fit = model_fit.predict(features_fit)
    plt.plot(x, y, '.b', label='Data')
    plt.plot(x_fit, y_fit, '-r', label='Fit')
    plt.title(description)
    plt.show()

    # X = f.numpy()
    # lab = labels.numpy()
    # int_labels = np.array(lab>.5, int)
    # fig = plt.figure(figsize=(5, 5))
    # plot_decision_regions(X=X, y=int_labels, clf=model_fit, legend=2)
    # train_labels = model_fit.predict(X, use_multiprocessing=True)
    # plt.title(description)
    # plt.show()

    logger.log_variables(fig, 'fig',
                         x, 'X',
                         y, 'y',
                         keep_best_callback.best_weights, 'weights',
                         history.history, 'history')
    logger.log_text(f'best_val_loss = {keep_best_callback.best_val_loss}', 'best_val_loss.txt')
    logger.log_text(f'{keep_best_callback.best_weights}', 'weights.txt')
    logger.log_figure(fig, 'fig.pdf')

if __name__ == '__main__':
    # for i in range(5):
    #     main(i)
    main(4)