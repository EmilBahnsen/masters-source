import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from colorblind import colorblind

def plot_decision_boundary(X, y, model: tf.keras.models.Model, steps=200, cmap='RdBu'):
    cmap = plt.get_cmap(cmap)

    # Define region of interest by data limits
    xmin, xmax = X[:,0].min() - 1, X[:,0].max() + 1
    ymin, ymax = X[:,1].min() - 1, X[:,1].max() + 1
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    # Make predictions across region of interest
    labels = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Plot decision boundary in region of interest
    z = labels.reshape(xx.shape)

    fig, ax = plt.subplots()
    CS = ax.contourf(xx, yy, z, cmap=cmap, vmin=0, vmax=1, alpha=.5)
    ax.contour(CS, levels=[0], colors='k', linestyles='--')

    # Get predicted labels on training data and plot
    train_labels = model.predict(X, use_multiprocessing=True)
    ax.scatter(X[:,0], X[:,1], c=y, cmap=cmap, lw=0, vmin=0, vmax=1)

    return fig, ax, train_labels


def plot_decision_boundary2(X, Y, model: tf.keras.models.Model, steps = 200):
    # Define region of interest by data limits
    xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
    ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    # Make predictions across region of interest
    labels = model.predict(np.c_[xx.ravel(), yy.ravel()]) > .5

    # Plot decision boundary in region of interest
    z = labels.reshape(xx.shape)

    fig, ax = plt.subplots()
    plt.contourf(xx, yy, z, cmap=plt.cm.Paired)
    plt.axis('off')

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

    train_labels = model.predict(X, use_multiprocessing=True)
    return fig, ax, train_labels


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