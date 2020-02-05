import os

import datetime
import tensorflow as tf
from tf_qc.losses import Mean1mFidelity
from tf_qc.models import OneDiamondQFT
from tf_qc.utils import random_state_vectors
from tf_qc import complex_type

N = 4

# Random normalized vectors
n_datapoints = 100000


vectors = random_state_vectors(n_datapoints, N, 0)

input = tf.cast(vectors, complex_type)
output = input

# Optimizer and loss
lr = 0.0001
print('Learning rate:', lr)
optimizer = tf.optimizers.Adam(lr)
loss = Mean1mFidelity()

for _ in range(100):
    # Model
    model = OneDiamondQFT()
    model.compile(optimizer, loss=loss)

    # Fitting
    current_time = datetime.datetime.now().replace(microsecond=0).isoformat()
    filename = os.path.basename(__file__).rstrip('.py')
    log_path = ['./logs', filename, 'model_b', current_time]

    from tf_qc.training import train  # Reimport so that we don't have to reset the run
    train(model, input, output, optimizer, loss, log_path, epochs=1000)
