import tensorflow as tf
from .layers import *
from typing import *

class PrePostQFTUIQFT(tf.keras.Model):
    def __init__(self):
        super(PrePostQFTUIQFT, self).__init__()
        self.U3_in = U3Layer()
        self.QFT_U = QFTULayer()
        self.U3_out = U3Layer()
        self.IQFT = IQFTLayer()

    def call(self, inputs, training=None, mask=None):
        x = self.U3_in(inputs)
        x = self.QFT_U(x)
        x = self.U3_out(x)
        x = self.IQFT(x)
        return x

    def matrix(self):
        return self.IQFT.matrix() @ self.U3_out.matrix() @ self.QFT_U.matrix() @ self.U3_in.matrix()


class ApproxUsingInverse(tf.keras.Model):
    def __init__(self, model_class: Callable[..., tf.keras.Model], target_class: Callable[..., tf.keras.Model]):
        super(ApproxUsingInverse, self).__init__()


