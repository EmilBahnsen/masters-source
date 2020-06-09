import os

import tensorflow as tf
from tf_qc.metrics import OperatorFidelity, FidelityMetric, StdFidelityMetric

from tf_qc.losses import Mean1mFidelity, Mean1mUhlmannFidelity, MeanTraceDistance
from tf_qc.models import OneMemoryDiamondQFT, OneDiamondQFT
from tf_qc.utils import random_pure_states
from tf_qc import complex_type

device = 'cpu'
# Random normalized vectors
n_datapoints = 100000

print('Using:', device)

N = 6
# Init. states that are random in the input but |00> on the two ancilla qubits
with tf.device(device):
    vectors = random_pure_states((n_datapoints, 2**N, 1), post_zeros=2)

input_states = tf.cast(vectors, complex_type)
output_states = input_states

# Optimizer and loss
lr = 0.1
print('Learning rate:', lr)
optimizer = tf.optimizers.Adam(lr)
# loss = Mean1mUhlmannFidelity([0, 1, 2, 3], N)
subsystem = [0, 1, 2, 3]
# loss = Mean1mFidelity(subsystem, true_is_pure_on_sub=True)
loss = Mean1mFidelity(subsystem, true_is_pure_on_sub=True)

for _ in range(100):
    # Model
    model = OneMemoryDiamondQFT('model_e4')

    # Fitting
    filename = os.path.basename(__file__).rstrip('.py')
    log_path = ['./logs', filename]

    # Metrics
    oper_fid_metric = OperatorFidelity(model)
    state_fid_metric = FidelityMetric(subsystem=subsystem, a_subsys_is_pure=True)
    state_std_fid_metric = StdFidelityMetric(subsystem=subsystem, a_subsys_is_pure=True)
    metrics = [state_fid_metric, state_std_fid_metric]

    from tf_qc.training import train  # Reimport so that we don't have to reset the run
    with tf.device(device):
        train(model, input_states, output_states, optimizer, loss, log_path, epochs=100, batch_size=10000, metrics=metrics)
