import tensorflow as tf

import os
import tensorflow as tf
import numpy as np
import datetime
from logtools import Logger

π = np.pi

file_name = os.path.basename(os.path.splitext(__file__)[0])
time = datetime.datetime.now().isoformat()
description = 'MNIST_intermetdiate_layer'
# description = 'TEST'
logger = Logger(log_dir=os.path.join(file_name, description, time))
logger.log_file(__file__)
print('log_dir', logger.log_dir)

# sHHHH = tensor(4*[H]) @ s0000

class NN(tf.keras.Sequential):
    def __init__(self):
        input_dense = tf.keras.layers.Dense(10, name='input_linear_combi')
        # output_dense = tf.keras.layers.Dense(10, name='mnist')

        super(NN, self).__init__(layers=[
            tf.keras.layers.Flatten(),
            input_dense,
            # QC(),  # This works as an activation function
            # output_dense
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
    model = NN()

    # MNIST
    optimizer = tf.optimizers.SGD(0.01)
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

    # MNIST
    logger.log_variables(keep_best_callback.best_weights, 'weights',
                        history.history, 'history')
    # MNIST
