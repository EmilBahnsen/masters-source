import os

from txtutils import *
from tf_qc import *
import datetime

N = 4

# Random vectors
n_datapoints = 1000000
vectors = random_unifrom_complex((n_datapoints, 2**N, 1))
# normalize
vectors = normalize_state_vectors(vectors)

input = tf.cast(vectors, complex_type)
output = input

# model, optimizer and loss
pre_post_model = PrePostQFTUIQFT()
optimizer = tf.optimizers.Adam()
loss = Mean1mFidelity()
pre_post_model.compile(optimizer, loss=loss)

# Fitting
current_time = datetime.datetime.now().replace(microsecond=0).isoformat()
filename = os.path.basename(__file__).rstrip('.py')
log_path = './logs/' + filename + '/' + current_time
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1, profile_batch=0)
pre_post_model.fit(input, output, validation_split=0.2, batch_size=512, epochs=10, callbacks=[tensorboard_callback])
print(*pre_post_model.variables, sep='\n')
print(pre_post_model.summary())

result = pre_post_model.model_matrix()
print(ndtotext(result.numpy()))

# Sanity check: test the QFT_U against QFT on all data
qft_layer = QFTLayer()
real_output = qft_layer(input)
model_output = result @ input
print('Sanity check loss:', loss(real_output, model_output).numpy())
std_loss = StdFidelity()
print('Sanity check loss std:', std_loss(real_output, model_output).numpy())
