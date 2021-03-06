import os

import datetime
import tensorflow as tf
from tf_qc.losses import Mean1mFidelity
from tf_qc.models import OneDiamondQFT, OneDiamondISWAP
from tf_qc.utils import random_pure_states
from tf_qc import complex_type

N = 4

# Random normalized vectors
n_datapoints = 100000

# vectors = random_state_vectors(n_datapoints, N, 0)
vectors = random_pure_states((n_datapoints, 2**N, 1))

input = tf.cast(vectors, complex_type)
output = input

# Optimizer and loss
lr = 0.01
print('Learning rate:', lr)
optimizer = tf.optimizers.SGD(lr)
loss = Mean1mFidelity()

for _ in range(100):
    # Model
    model = OneDiamondISWAP()

    # Fitting
    filename = os.path.basename(__file__).rstrip('.py')
    log_path = ['./logs', filename]

    from tf_qc.training import train  # Reimport so that we don't have to reset the run
    train(model, input, output, optimizer, loss, log_path, epochs=10000)
