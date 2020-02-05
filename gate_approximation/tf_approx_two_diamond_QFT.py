import os

from tf_qc.layers import QFTLayer
from txtutils import *
from tf_qc import *
import datetime

N = 8

# Random vectors
n_datapoints = 2000
vectors = random_state_vectors(n_datapoints, N, 0)

input = tf.cast(vectors, complex_type)
output = input

# model, optimizer and loss
model = TwoDiamondQFT()
lr = 0.001
print('Learning rate:', lr)
optimizer = tf.optimizers.Adam(lr)
loss = Mean1mFidelity()
model.compile(optimizer, loss=loss)

# Fitting
current_time = datetime.datetime.now().replace(microsecond=0).isoformat()
filename = os.path.basename(__file__).rstrip('.py')
log_path = './logs/' + filename + '/' + current_time
print('logs:', log_path)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1, profile_batch=0)
model.fit(input,
          output,
          validation_split=0.2,
          batch_size=32,
          epochs=100,
          callbacks=[tensorboard_callback])
print(*model.variables, sep='\n')
model.summary()

result = model.model_matrix()
print(ndtotext(result.numpy()))

# Sanity check: test the QFT_U against QFT on all data
qft_layer = QFTLayer()
real_output = qft_layer(input)
model_output = result @ input
print('Sanity check loss:', loss(real_output, model_output).numpy())
std_loss = StdFidelity()
print('Sanity check loss std:', std_loss(real_output, model_output).numpy())