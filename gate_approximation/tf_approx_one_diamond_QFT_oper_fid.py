from tf_qc.layers import QFTLayer

from tf_qc import *
import txtutils

# tf.compat.v1.enable_eager_execution()

qftu_layer = QFTULayer()
iqft_layer = IQFTLayer()

# dummy_data = random_state_vectors(1,4)
shape = (0, 2**4, 0)
qftu_layer.build(shape)
iqft_layer.build(shape)

@tf.function
def loss():
    return 1 - (tf.abs(tf.linalg.trace(iqft_layer.matrix() @ qftu_layer.matrix())) ** 2) / (2 ** 4)

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
for i in range(10000):
    opt.minimize(loss, var_list=qftu_layer.trainable_variables)
    print(i, loss().numpy())
print(qftu_layer.trainable_variables, sep='\n')

print('qftu_layer.matrix()')
print(txtutils.ndtotext(qftu_layer.matrix().numpy()))

print('iqft_layer.matrix()')
print(txtutils.ndtotext(iqft_layer.matrix().numpy()))

print('eye:')
print(txtutils.ndtotext((iqft_layer.matrix() @ qftu_layer.matrix()).numpy()))

data = random_state_vectors(10000, 4)

print('loss2')
loss2 = Mean1mFidelity()
print(loss2(data, iqft_layer.matrix() @ qftu_layer.matrix() @ data))
