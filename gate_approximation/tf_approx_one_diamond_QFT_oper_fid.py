from tf_qc.layers import QFTLayer, IQFTLayer

from tf_qc import *
import txtutils
from scipy.optimize import basinhopping

# tf.compat.v1.enable_eager_execution()
from tf_qc.losses import Mean1mFidelity, Mean1mTraceDistance
from tf_qc.models import OneDiamondQFT
from tf_qc.utils import random_state_vectors

iqft_layer = IQFTLayer()

# dummy_data = random_state_vectors(1,4)
shape = (0, 2**4, 0)
iqft_layer.build(shape)
phony_data = random_state_vectors(1, 4)


# Based on: https://arxiv.org/pdf/0803.2940.pdf
@tf.function
def loss(variable_layer):
    d = 2**4
    return 1 - tf.abs(tf.linalg.trace(iqft_layer.matrix() @ variable_layer.matrix) / d) ** 2


# def eval_func(*args):
#     v: tf.Variable
#     for i, v in enumerate(qftu_layer.trainable_variables):
#         v.assign(args[i])
#     return loss()


# exit()
#
# basinhopping(eval_func, qftu_layer.trainable_variables)


# loss = Mean1mTraceDistance()
opt = tf.keras.optimizers.Adam(0.005)
best_model = None
best_loss = 1
for itter in range(10000):
    last_loss200 = 1
    model = OneDiamondQFT()
    model(phony_data)
    new_loss = lambda: loss(model)
    for i in range(50000):
        opt.minimize(new_loss, var_list=model.trainable_variables)
        if i % 200 == 0:
            loss_val = loss(model).numpy()
            print(i, loss_val)
            if loss_val < best_loss:
                best_loss = loss_val
                best_model = model
            if last_loss200 - loss_val < 1e-10:
                print('retry!')
                print('best_loss', best_loss)
                print(*best_model.trainable_variables, sep='\n')
                break
            last_loss200 = loss_val
print(*best_model.trainable_variables, sep='\n')
print('loss1:', best_loss)

print('best_model.matrix()')
print(txtutils.ndtotext(best_model.matrix().numpy()))

print('iqft_layer.matrix()')
print(txtutils.ndtotext(iqft_layer.matrix().numpy()))

print('eye:')
print(txtutils.ndtotext((iqft_layer.matrix() @ best_model.matrix()).numpy()))

data = random_state_vectors(10000, 4)

print('loss2')
loss2 = Mean1mFidelity()
print(loss2(data, iqft_layer.matrix() @ best_model.matrix() @ data))
