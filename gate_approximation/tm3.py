import tensorflow as tf
import tf_qc.utils
import tf_qc.models

N = 7
# data = tf_qc.utils.random_pure_states((50, 2**N, 1))
# data_batch = data[:32]


@tf.function
def f():
    # result = tf_qc.utils.partial_trace_v2(data, list(range(N-1)))
    # tf.print(tf.math.real(result), 'j', tf.math.imag(result), sep='\n')
    # logm = tf.linalg.logm(m.matrix())
    m = tf.cast(tf.fill((32, 2**N, 2**N), 1), tf.complex64)
    for _ in range(10):
        m @= m
    m = tf.linalg.logm(m)
    tf.print(m)


import time
time_now = time.time()
with tf.device('cpu'):
    f()
print('cpu time:', time.time() - time_now)

time_now = time.time()
with tf.device('gpu'):
    f()
print('gpu time:', time.time() - time_now)

