import os

import datetime
import tensorflow as tf
# Force to use the CPU
tf.config.experimental.set_visible_devices([], 'GPU')
from tf_qc.losses import Mean1mFidelity, MeanTraceDistance
from tf_qc.models import OneDiamondQFT, QCModel, ApproxUsingTarget
from tf_qc.layers import QFTLayer, ILayer, U3Layer, ULayer, CNOTGateLayer, ToffoliGateLayer, _normal_theta, FredkinGateLayer, ISWAPLayer
from tf_qc.utils import random_pure_states
from tf_qc import complex_type, float_type
from tf_qc.training import train_ApproxUsingTarget
from tf_qc.RAdam import RAdamOptimizer
import numpy as np

π = np.pi

class OneDiamondSimpleGate(ApproxUsingTarget):

    @property
    def model_layers(self):
        return {
            'model_a': [
                U3Layer(),
                ULayer(),
                U3Layer()
            ],
            'model_a2': [
                U3Layer(),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer()
            ],
            'model_a3': [
                U3Layer(),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer()
            ],
            'model_b': [
                U3Layer(),
                ISWAPLayer([0, 2], parameterized=True),
                ISWAPLayer([0, 3], parameterized=True),
                ISWAPLayer([1, 2], parameterized=True),
                ISWAPLayer([1, 3], parameterized=True),
                ULayer(),
                ISWAPLayer([0, 2], parameterized=True),
                ISWAPLayer([0, 3], parameterized=True),
                ISWAPLayer([1, 2], parameterized=True),
                ISWAPLayer([1, 3], parameterized=True),
                U3Layer()
            ],
        }

    @property
    def target_model_layers(self):
        return {
            'CNOT01': [
                CNOTGateLayer(0, 1)
            ], # model_a: .470(38) loss
            'toffoli012': [
                ToffoliGateLayer([0, 1], 2)
            ], # model_a: .41(18), model_a2: .41 loss
            'toffoli013': [
                ToffoliGateLayer([0, 1], 3)
            ],
            'toffoli012_013': [
                ToffoliGateLayer([0, 1], 2),
                ToffoliGateLayer([0, 1], 3)
            ], # model_a: .41, model_a2: .41(21) loss
            'fredkin023': [
                FredkinGateLayer(0, [2, 3])
            ],
            'U_pi_2': [  # Just to test for bugs, but this seems to converge
                ULayer(initializer=lambda shape,dtype: tf.constant(π/2, dtype=dtype, shape=shape))
            ]
        }

    def __init__(self, model_name, target_name):
        model = QCModel(self.model_layers[model_name], name=model_name)
        target_model = QCModel(self.target_model_layers[target_name], name=target_name)
        super().__init__(model, target_model)

    @property
    def name(self):
        return f'{super().name}__{self.model.name}__{self.target_model.name}'


N = 4

# Random normalized vectors
n_datapoints = 100000

# vectors = random_state_vectors(n_datapoints, N, 0)
vectors = random_pure_states((n_datapoints, 2**N, 1), post_zeros=1)
input = tf.cast(vectors, complex_type)

# Model
model = OneDiamondSimpleGate(model_name='model_b', target_name='toffoli012')
print(model.name)

# Convert input to generate output
output = model.apply_target_to_states(input)

# Optimizer and loss
lr = .001
beta1 = 0.9
print('Learning rate:', lr)
print('First momentum decay rate:', beta1)
# optimizer = RAdamOptimizer(lr, beta1)
optimizer = tf.optimizers.Adam(lr, beta1)
# loss = lambda y_true, y_pred: Mean1mFidelity()(y_true, y_pred) + MeanTraceDistance()(y_true, y_pred)
loss = Mean1mFidelity([0, 1, 2])

# for _ in range(100):
# Model

model.compile(optimizer, loss=loss)

# Fitting
current_time = datetime.datetime.now().replace(microsecond=0).isoformat()
filename = os.path.basename(__file__).rstrip('.py')
info = '_use012_only_'  # random_pure_states((n_datapoints, 2**N, 1), post_zeros=1) AND Mean1mFidelity([0, 1, 2])
log_path = ['./logs', filename, info]

# from tf_qc.training import train  # Reimport so that we don't have to reset the run
train_ApproxUsingTarget(model, input, output, optimizer, loss, log_path, epochs=100, batch_size=250)
