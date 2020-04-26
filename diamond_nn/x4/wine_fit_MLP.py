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
import pandas as pd

π = np.pi

# sHHHH = tensor(4*[H]) @ s0000

class Model(tf.keras.Sequential):
    def __init__(self):
        super(Model, self).__init__(layers=[
            tf.keras.layers.Dense(15, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(6, activation='sigmoid')
        ])

def main(*args):
    file_name = os.path.basename(os.path.splitext(__file__)[0])
    time = datetime.datetime.now().isoformat()
    description = f'redwine'
    # description = 'TEST'
    logger = Logger(log_dir=os.path.join(file_name, description, time))
    logger.log_file(__file__)
    print('log_dir', logger.log_dir)

    # Red wine data
    data = pd.read_csv('../data/winequality-red.csv')
    X = np.asarray(data.drop(['quality'], axis=1))  # Drop the labels for the X-data
    # Adjust to quality scale between 0 and 6
    y = np.asarray(data['quality'] - min(data['quality']), int)

    # --- Model and loss ---
    model = Model()

    optimizer = tf.optimizers.Adam(0.001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    # model.build((1, 2))
    # print(model.summary())
    print(*model.trainable_variables, sep='\n')

    keep_best_callback = KeepBestCallback()
    history: tf.keras.callbacks.History = model.fit(X, y,
                                                    validation_split=0.2,
                                                    epochs=1000,
                                                    callbacks=[keep_best_callback],
                                                    verbose=2)
    print('Best', 'val_accuracy', history.history['val_accuracy'][keep_best_callback.best_epoch])
    print('Best', 'val_loss', history.history['val_loss'][keep_best_callback.best_epoch])

    # --- Test result ---
    # model_fit = model
    # model_fit(features[:2,...])  # Phony init of model
    # model_fit.set_weights(keep_best_callback.best_weights)
    # for i in range(len(model_fit.weights)):
    #     model_fit.weights[i].assign(keep_best_callback.best_weights[i])

    # logger.log_variables(fig, 'fig',
    #                      X, 'X',
    #                      int_labels, 'y',
    #                      train_labels, 'train_labels',
    #                      keep_best_callback.best_weights, 'weights',
    #                      history.history, 'history')
    # logger.log_figure(fig, 'fig.pdf')

if __name__ == '__main__':
    main(sys.argv[1:])