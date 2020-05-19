import os

import datetime
import tensorflow as tf
# Force to use the CPU
tf.config.experimental.set_visible_devices([], 'GPU')
from tf_qc.losses import Mean1mFidelity, MeanTraceDistance
from tf_qc.models import OneDiamondQFT, ApproxUsingInverse, QCModel
from tf_qc.layers import IQFTLayer, QFTCrossSwapLayer, U3Layer, ISWAPLayer, XGateLayer, YGateLayer, ZGateLayer, ILayer, \
    QFTLayer
from tf_qc.utils import random_pure_states
from tf_qc import complex_type
from tf_qc.training import train
from tf_qc import qc


class TwoQubitQFT(ApproxUsingInverse):
    def __init__(self, model_name):
        model = QCModel(layers=[
            U3Layer(),
            ISWAPLayer([0,1], parameterized=True),
            XGateLayer(1),
            ISWAPLayer([0,1], parameterized=True),
            ZGateLayer(0),
            ISWAPLayer([0, 1], parameterized=True),
            ZGateLayer(0),
            XGateLayer(1),
            U3Layer(),
            # QFTCrossSwapLayer()
        ])
        target = QCModel(layers=[
            QFTLayer()
        ])
        super(TwoQubitQFT, self).__init__(model, target, model_name)


class TwoQubitQFT_TEST(TwoQubitQFT):
    def __init__(self, model_name):
        super().__init__(model_name)
        print(self.layers)
        self.layers.pop(-1)
        self.target_inv = QCModel([ILayer()])
        self.add(self.target_inv)


N = 2

# Random normalized vectors
n_datapoints = 1000000

# vectors = random_state_vectors(n_datapoints, N, 0)
vectors = random_pure_states((n_datapoints, 2**N, 1))

input = tf.cast(vectors, complex_type)
true_QFT_gate = QFTLayer()
output = true_QFT_gate(input)
# output = input
# Optimizer and loss
lr = .1
print('Learning rate:', lr)
optimizer = tf.optimizers.Adam(lr)
# loss = lambda y_true, y_pred: Mean1mFidelity()(y_true, y_pred) + MeanTraceDistance()(y_true, y_pred)
loss = Mean1mFidelity()

# Model
model = TwoQubitQFT_TEST(model_name='model_0410001')
model.compile(optimizer, loss=loss)

# Fitting
current_time = datetime.datetime.now().replace(microsecond=0).isoformat()
filename = os.path.basename(__file__).rstrip('.py')
log_path = ['./logs', filename]

# from tf_qc.training import train  # Reimport so that we don't have to reset the run
train(model, input, output, optimizer, loss, log_path, epochs=10, batch_size=2500)
