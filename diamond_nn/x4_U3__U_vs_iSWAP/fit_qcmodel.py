import tensorflow as tf
# Force to use the CPU
tf.config.experimental.set_visible_devices([], 'GPU')

import os
import pickle
from tf_qc.models import QCModel, U3Layer, ISWAPLayer, ULayer
from tf_qc import complex_type, float_type
from tf_qc.qc import tensor, s00, s0000, trace, measure, U3, iSWAP, gates_expand_toN, gate_expand_2toN, H, apply_operators2state
from tf_qc import qc
from tf_qc.layers import _uniform_0_1, _normal_0_1, ISWAPLayer, HLayer
from diamond_nn.datasets import normalize_data, two_circles_in_plain, checkerboard_in_plain
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
from functools import reduce
from logtools import Logger
from sklearn import datasets
from diamond_nn import plot_decision_boundary

π = np.pi

file_name = os.path.basename(os.path.splitext(__file__)[0])
time = datetime.datetime.now().isoformat()
description = 'U3_U_2003.09887_ish_MNIST'
# description = 'TEST'
logger = Logger(log_dir=os.path.join(file_name, description, time))
logger.log_file(__file__)
print('log_dir', logger.log_dir)

# sHHHH = tensor(4*[H]) @ s0000

class U3_U(tf.keras.Sequential):
    def __init__(self):
        input_dense = tf.keras.layers.Dense(4, name='input_linear_combi')
        # output_dense = tf.keras.layers.Dense(1, activation='sigmoid', name='output_linear_combi')
        # MNIST
        output_dense = tf.keras.layers.Dense(10, name='mnist')
        # MNIST

        # This defines the quantum part of the model
        class QC(tf.keras.Model):
            def __init__(self):
                init = tf.initializers.glorot_uniform()
                super(QC, self).__init__()
                # self.w1 = tf.Variable(init((8,), dtype=float_type),
                #                       trainable=True,
                #                       dtype=float_type)
                # self.w2 = tf.Variable(init((8,), dtype=float_type),
                #                       trainable=True,
                #                       dtype=float_type)
                # self.w3 = tf.Variable(init((11,), dtype=float_type),
                #                       trainable=True,
                #                       dtype=float_type)
                # self.w4 = tf.Variable(init((11,), dtype=float_type),
                #                       trainable=True,
                #                       dtype=float_type)
                # self.w3 = tf.Variable(init((13,), dtype=float_type),
                #                       trainable=True,
                #                       dtype=float_type)
                # self.w4 = tf.Variable(init((13,), dtype=float_type),
                #                       trainable=True,
                #                       dtype=float_type)
                # self.w5 = tf.Variable(init((13,), dtype=float_type),
                #                       trainable=True,
                #                       dtype=float_type)
                self.wU = tf.Variable(init((2,), dtype=float_type),
                                      trainable=True,
                                      dtype=float_type)


            def call(self, inputs, training=None, mask=None):
                # u3_1 = tensor([
                #     U3([self.w1[0], 0, self.w1[1]]),
                #     U3([self.w1[2], 0, self.w1[3]]),
                #     U3([self.w1[4], 0, self.w1[5]]),
                #     U3([self.w1[6], 0, self.w1[7]])
                # ])
                # u3_2 = tensor([
                #     U3([self.w2[0], 0, self.w2[1]]),
                #     U3([self.w2[2], 0, self.w2[3]]),
                #     U3([self.w2[4], 0, self.w2[5]]),
                #     U3([self.w2[6], 0, self.w2[7]])
                # ])
                # u3_3 = tensor([
                #     U3(self.w3[0:3]),
                #     U3(self.w3[3:6]),
                #     U3(self.w3[6:9]),
                #     U3(self.w3[9:12])
                # ])
                # u3_4 = tensor([
                #     U3(self.w4[0:3]),
                #     U3(self.w4[3:6]),
                #     U3(self.w4[6:9]),
                #     U3(self.w4[9:12])
                # ])
                # u3_5 = tensor([
                #     U3(self.w5[0:3]),
                #     U3(self.w5[3:6]),
                #     U3(self.w5[6:9]),
                #     U3(self.w5[9:12])
                # ])
                u_1 = qc.U(self.wU[0])
                # u_2 = qc.U(self.wU[1])
                # u_3 = qc.U(self.wU[2])
                def apply_qc_to_data(data):
                    opers = []
                    opers.append(tensor([
                        U3([data[0], π/2, π/2]),
                        U3([data[1], π/2, π/2]),
                        U3([data[2], π/2, π/2]),
                        U3([data[3], π/2, π/2])
                    ]))
                    opers.append(u_1)
                    # opers.append(u3_1)
                    # opers.append(u_2)
                    # opers.append(u3_2)
                    return apply_operators2state(opers, s0000)

                outputs = tf.map_fn(apply_qc_to_data, inputs, dtype=complex_type, parallel_iterations=12)

                # Make a linearcombi. of the output probabilities
                # we measure all of them as combine them in linear combi. + bias and then apply sigmoid
                Ps = tf.cast(measure(outputs, [0, 1, 2, 3]), float_type)
                return Ps

        super(U3_U, self).__init__(layers=[
            # MNIST
            tf.keras.layers.Flatten(),
            # MNIST
            input_dense,
            QC(),  # This works as an activation function
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


if __name__ == '__main__':
    with open(__file__) as this_file:
        this_file = this_file.read()
    # This data goes into C1 and T1
    # plain_x, labels = datasets.make_moons(n_samples=1000, noise=0.1, random_state=0)
    # colors = ['blue' if label == 1 else 'red' for label in labels]
    # plain_x = normalize_data(plain_x)
    # plain_x = tf.convert_to_tensor(plain_x, dtype=float_type)
    # labels = tf.convert_to_tensor(labels, dtype=float_type)
    #
    # # --- Feature extraction ---
    # features = tf.convert_to_tensor([
    #     plain_x[..., 0],
    #     plain_x[..., 1],
    #     # plain_x[..., 0] ** 2,
    #     # plain_x[..., 1] ** 2,
    #     # plain_x[..., 0] * plain_x[..., 1]
    # ])
    # features = tf.transpose(features)

    # MNIST
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_images = np.expand_dims(train_images, axis=-1)
    test_images = tf.expand_dims(test_images, axis=-1)
    # MNIST

    # --- Model and loss ---
    model = U3_U()

    if False:
        optimizer = tf.optimizers.Adam(0.01)
        # loss = P00MaximisationLoss()
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        # model.build((1, 2))
        # print(model.summary())
        print(*model.trainable_variables, sep='\n')
    # MNIST
    optimizer = tf.optimizers.Adam(0.001)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    # MNIST

    keep_best_callback = KeepBestCallback()

    # history = model.fit(features, labels,
    #                       validation_split=0.2,
    #                       batch_size=200,
    #                       epochs=1000,
    #                       callbacks=[keep_best_callback],
    #                       use_multiprocessing=True)

    # MNIST
    history = model.fit(train_images, train_labels, epochs=100, callbacks=[keep_best_callback],
                        validation_data=(test_images, test_labels))
    # MNIST

    # --- Test result ---
    # model.load_weights('fit_qcmodel_model/best.h5')
    # model_fit = model
    # model_fit(features[:2,...])  # Phony init of model
    # model_fit.set_weights(keep_best_callback.best_weights)
    # for i in range(len(model_fit.weights)):
    #     model_fit.weights[i].assign(keep_best_callback.best_weights[i])
    # out_P00 = model_fit(features)
    # x = plain_x.numpy()
    # P00 = out_P00.numpy().flatten()
    # # labels = labels.numpy()
    # fig = plt.figure()
    # plt.title(f'{description}')
    # plt.scatter(x[:, 0], x[:, 1], c=P00, cmap=plt.get_cmap('bwr'))
    # plt.show()

    # X = plain_x.numpy()
    # lab = labels.numpy()
    # fig, ax, train_labels = plot_decision_boundary(X, lab, model)
    # plt.title(description)
    # plt.show()

    # logger.log_variables(fig, 'fig',
    #                      X, 'x',
    #                      train_labels, 'train_labels',
    #                      keep_best_callback.best_weights, 'weights',
    #                      history, 'history')
    # logger.log_figure(fig, 'fig.eps')
    # MNIST
    logger.log_variables(keep_best_callback.best_weights, 'weights',
                        history.history, 'history')
    # MNIST
