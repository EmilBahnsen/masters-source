from tf_qc import float_type
from diamond_nn.datasets import normalize_data, two_circles_in_plain, checkerboard_in_plain
import tensorflow as tf
import matplotlib.pyplot as plt


class P00MaximisationLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(P00MaximisationLoss, self).__init__()

    def call(self, y_true, y_pred):
        # y_true = tf.squeeze(y_true)  # This is apparently not properly formatted
        P_sum = y_pred
        delta = P_sum - y_true
        abs_delta = tf.abs(delta)
        dist_to_true = abs_delta# + tf.nn.relu(abs_delta - 1) # abs loss + loss for being on the wrong side of 0!
        mean = tf.reduce_mean(dist_to_true)
        return mean

class Model(tf.keras.models.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer((5,)),
            tf.keras.layers.Dense(7, activation='relu'),
            tf.keras.layers.Dense(6, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs, training=None, mask=None):
        input_poly_elem = tf.convert_to_tensor([
            inputs[..., 0],
            inputs[..., 1],
            inputs[..., 0] ** 2,
            inputs[..., 1] ** 2,
            inputs[..., 0] * inputs[..., 1]
        ])
        input_poly_elem = tf.transpose(input_poly_elem)
        return self.model(input_poly_elem)


# This data goes into C1 and T1
plain_x, labels = checkerboard_in_plain(False)
plain_x = normalize_data(plain_x)
plain_x = tf.convert_to_tensor(plain_x, dtype=float_type)
labels = tf.convert_to_tensor(labels, dtype=float_type)

model = Model()

optimizer = tf.optimizers.Adam(0.005)
# loss = P00MaximisationLoss()
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
print(*model.trainable_variables, sep='\n')
checkpoint = tf.keras.callbacks.ModelCheckpoint('fit_nn_model.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
model.fit(plain_x, labels, validation_split=0.2, batch_size=50, epochs=1000, callbacks=[checkpoint])
print(model.summary())

# --- Test result ---
model = Model()
model.load_weights('fit_nn_model.h5')
out_P00 = model(plain_x)
x = plain_x.numpy()
P00 = out_P00.numpy()
labels = labels.numpy()
plt.figure()
plt.title('fit_nn')
plt.scatter(x[:, 0], x[:, 1], c=P00.flatten(), cmap=plt.get_cmap('bwr'))
plt.show()
