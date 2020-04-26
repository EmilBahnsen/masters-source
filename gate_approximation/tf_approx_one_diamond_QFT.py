import os

import datetime
import tensorflow as tf
# Force to use the CPU
# tf.config.experimental.set_visible_devices([], 'GPU')
from tf_qc.losses import Mean1mFidelity, MeanTraceDistance
from tf_qc.models import OneDiamondQFT
from tf_qc.utils import random_pure_states
from tf_qc import complex_type
from tf_qc.training import train


N = 4

# Random normalized vectors
n_datapoints = 100000

# vectors = random_state_vectors(n_datapoints, N, 0)
vectors = random_pure_states((n_datapoints, 2**N, 1))

input = tf.cast(vectors, complex_type)
output = input

# Optimizer and loss
lr = 0.02
print('Learning rate:', lr)
optimizer = tf.optimizers.Adam(lr)
# loss = lambda y_true, y_pred: Mean1mFidelity()(y_true, y_pred) + MeanTraceDistance()(y_true, y_pred)
loss = Mean1mFidelity()

for _ in range(100):
    # Model
    model = OneDiamondQFT(model_name='model_e10')
    model.compile(optimizer, loss=loss)

    # Fitting
    current_time = datetime.datetime.now().replace(microsecond=0).isoformat()
    filename = os.path.basename(__file__).rstrip('.py')
    log_path = ['./logs', filename]

    # from tf_qc.training import train  # Reimport so that we don't have to reset the run
    train(model, input, output, optimizer, loss, log_path, epochs=10000, batch_size=64)
