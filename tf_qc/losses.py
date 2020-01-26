import tensorflow as tf
import math

from tf_qc import float_type, complex_type


class MeanNorm(tf.losses.Loss):
    def call(self, y_true, y_pred):
        # y_pred = tf.convert_to_tensor(y_pred)
        diff = y_true - y_pred
        norms = tf.cast(tf.norm(diff, axis=[-2, -1]), dtype=float_type)
        mean_norm = tf.reduce_mean(norms)
        return mean_norm


class Mean1mFidelity(tf.losses.Loss):
    def call(self, y_true, y_pred):
        norm_squares = tf.transpose(y_true, perm=[0, 2, 1]) @ tf.math.conj(y_pred)
        fidelities = tf.square(tf.abs(norm_squares))
        meanFilelity = tf.reduce_mean(fidelities)
        return 1 - meanFilelity


class StdFidelity(tf.losses.Loss):
    def call(self, y_true, y_pred):
        norm_squares = tf.transpose(y_true, perm=[0, 2, 1]) @ tf.math.conj(y_pred)
        fidelities = tf.square(tf.abs(norm_squares))
        stdFilelity = tf.keras.backend.std(fidelities)
        return stdFilelity

# TEST
x = tf.constant([1/math.sqrt(3), 1/math.sqrt(3), 1/math.sqrt(3), 1/math.sqrt(2), 1/math.sqrt(2), 0], shape=(2,3,1), dtype=complex_type)
y = tf.constant([1, 0, 0, 1, 0, 0], shape=(2,3,1), dtype=complex_type)
assert round(Mean1mFidelity()(x, y).numpy(), 5) == round(1 - (1/3 + 1/2)/2, 5)
# TEST END

# TEST
# x = tf.constant([1,2,3,4], shape=(2,2,1), dtype=complex_type)
# assert round(MeanNorm()(x, 2*x).numpy(), 5) == 3.61803
# TEST END