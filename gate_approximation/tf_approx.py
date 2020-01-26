import time
from typing import *
from txtutils import *
from tf_qc import *

N = 4
qft_tensor = qft(N, I[N])
print(ndtotext(qft_tensor.numpy()))
qft_U_tensor = qft_U(N, I[N], tf.convert_to_tensor((N-1) * [π/2]))
print(ndtotext(qft_U_tensor.numpy()))

N = 4

n_t = sum(range(N))
init_t = tf.random.uniform([n_t], 0, 2*π, tf.float64)
t = tf.Variable(init_t, trainable=True, dtype=tf.float64, constraint=lambda t: tf.clip_by_value(t, 0, 2*π))


@tf.function
def loss():
    qft_tensor = qft(N, I4)
    qft_U_tensor = qft_U(N, I4, t)
    diff = qft_tensor - qft_U_tensor
    return tf.cast(tf.reduce_sum(tf.square(tf.norm(diff))), tf.float64)


print('loss:', loss())

opt = tf.keras.optimizers.Adam()
start_time = time.time()
for step in range(10000):
    with tf.GradientTape() as tape:
        grads = tape.gradient(loss(), t)
        opt.apply_gradients([(grads, t)])
    if step % 5000 == 0:
        print('loss:', loss())
        print('t', t)

print('time:', time.time() - start_time, "seconds")
print('loss:', loss())
print('t', t)
qft_U_tensor = qft_U(N, I4, t)
print(ndtotext(qft_U_tensor.numpy()))
