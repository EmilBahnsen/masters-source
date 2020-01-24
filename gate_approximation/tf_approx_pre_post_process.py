from utils import *
from tf_qc import *
import datetime
import itertools

# TEST
# x = tf.constant([1,2,3,4], shape=(2,2,1), dtype=complex_type)
# assert round(MeanNorm()(x, 2*x).numpy(), 5) == 3.61803
# TEST END


N = 4
# Data is just all the kinds of vectors that we might get |...0110...>, |...0111...>, |...1000...>, ... all normalized
# vectors = list(map(list, list(itertools.product([0, 1], repeat=2**N))))[1:]  # Skip the zero-vector as it creates problems
# vectors = tf.constant(vectors, dtype=complex_type, shape=(len(vectors), 2**N, 1))

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
current_time = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
log_path = './logs/tf_approx_pre_post_process_rnd_input/' + current_time
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1, profile_batch=0)
pre_post_model.fit(input, output, batch_size=512, epochs=1000, callbacks=[tensorboard_callback])
print(*pre_post_model.variables, sep='\n')
print(pre_post_model.summary())

result = pre_post_model.U3_out.matrix() @ pre_post_model.QFT_U.matrix() @ pre_post_model.U3_in.matrix()
print(ndtotext(result.numpy()))
