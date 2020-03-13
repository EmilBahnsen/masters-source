import os

import tensorflow as tf
from tf_qc.metrics import OperatorFidelity, FidelityMetric, StdFidelityMetric

from tf_qc.losses import Mean1mFidelity, Mean1mUhlmannFidelity, MeanTraceDistance
from tf_qc.models import TwoMemoryDiamondQFT
from tf_qc.utils import random_pure_states
from tf_qc import complex_type

device = 'gpu'
# Random normalized vectors
n_datapoints = 10000

print('Using:', device)

N = 12
# Init. states that are random in the input but |0000> on the 4 ancilla qubits
with tf.device(device):
    vectors = random_pure_states((n_datapoints, 2**N, 1), post_zeros=4)

input_states = tf.cast(vectors, complex_type)
output_states = input_states

# Optimizer and loss
lr = 0.01
print('Learning rate:', lr)
optimizer = tf.optimizers.Adam(lr)
# loss = Mean1mUhlmannFidelity([0, 1, 2, 3], N)
subsystem = [0, 1, 2, 3]
# loss = Mean1mFidelity(subsystem, true_is_pure_on_sub=True)
loss = Mean1mFidelity(subsystem, true_is_pure_on_sub=True)

with tf.device(device):
    for _ in range(100):
        # Model
        model = TwoMemoryDiamondQFT('model_1_a_ref')

        # Fitting
        filename = os.path.basename(__file__).rstrip('.py')
        log_path = ['./logs', filename]

        # Metrics
        oper_fid_metric = OperatorFidelity(model)
        state_fid_metric = FidelityMetric(subsystem=subsystem, a_subsys_is_pure=True)
        state_std_fid_metric = StdFidelityMetric(subsystem=subsystem, a_subsys_is_pure=True)
        metrics = [state_fid_metric, state_std_fid_metric]

        from tf_qc.training import train  # Reimport so that we don't have to reset the run
        train(model, input_states, output_states, optimizer, loss, log_path, epochs=100, batch_size=12, metrics=metrics)
