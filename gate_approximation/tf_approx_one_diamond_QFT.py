import os

import datetime
import tensorflow as tf
# Force to use the CPU
from tf_qc.metrics import OperatorFidelity, FidelityMetric, StdFidelityMetric

# tf.config.experimental.set_visible_devices([], 'GPU')
from tf_qc.losses import Mean1mFidelity, MeanTraceDistance
from tf_qc.models import OneDiamondQFT, QCModel
from tf_qc.layers import QFTLayer, ILayer
from tf_qc.utils import random_pure_states
from tf_qc import complex_type
from tf_qc.training import train
from tf_qc.RAdam import RAdamOptimizer


class OneDiamondQFT_no_inverse(OneDiamondQFT):
    def __init__(self, model_name):
        super().__init__(model_name)
        print(self.layers)
        self.layers.pop(-1)
        self.target_inv = QCModel([ILayer()])
        self.add(self.target_inv)


N = 4

# Random normalized vectors
n_datapoints = 1000

# vectors = random_state_vectors(n_datapoints, N, 0)
vectors = random_pure_states((n_datapoints, 2**N, 1))

input = tf.cast(vectors, complex_type)

# Convert teh output so that is is teh true QFT'ed input
true_QFT_gate = QFTLayer()
output = true_QFT_gate(input)

# Optimizer and loss
lr = .1
# beta1 = 0.999
print('Learning rate:', lr)
# print('First momentum decay rate:', beta1)
# optimizer = RAdamOptimizer(lr, beta1)
optimizer = tf.optimizers.Adam(lr)
# loss = lambda y_true, y_pred: Mean1mFidelity()(y_true, y_pred) + MeanTraceDistance()(y_true, y_pred)
loss = Mean1mFidelity()

# for _ in range(100):
# Model
model = OneDiamondQFT_no_inverse(model_name='model_a_swapU')
model.compile(optimizer, loss=loss)

# Fitting
current_time = datetime.datetime.now().replace(microsecond=0).isoformat()
filename = os.path.basename(__file__).rstrip('.py')
log_path = ['./logs', filename]

# Metrics
# oper_fid_metric = OperatorFidelity(model)
state_fid_metric = FidelityMetric()
state_std_fid_metric = StdFidelityMetric()
metrics = [state_fid_metric, state_std_fid_metric]

# from tf_qc.training import train  # Reimport so that we don't have to reset the run
train(model, input, output, optimizer, loss, log_path, epochs=500, batch_size=200, metrics = metrics)
