import os

import datetime
import tensorflow as tf
from tf_qc.losses import Mean1mFidelity, Mean1mUhlmannFidelity
from tf_qc.models import OneMemoryDiamondQFT, OneDiamondQFT
from tf_qc.utils import random_pure_states
from tf_qc import complex_type

# Random normalized vectors
n_datapoints = 100000

N = 6
vectors = random_pure_states((n_datapoints, 2**N, 1), post_zeros=2)

input_states = tf.cast(vectors, complex_type)
output_states = input_states

# Optimizer and loss
lr = 0.001
print('Learning rate:', lr)
optimizer = tf.optimizers.Adam(lr)
# loss = Mean1mUhlmannFidelity([0, 1, 2, 3], N)
loss = Mean1mFidelity()

for _ in range(100):
    # Model
    model = OneMemoryDiamondQFT()

    # Fitting
    filename = os.path.basename(__file__).rstrip('.py')
    log_path = ['./logs', filename]

    from tf_qc.training import train  # Reimport so that we don't have to reset the run
    train(model, input_states, output_states, optimizer, loss, log_path, epochs=10000, batch_size=32)
