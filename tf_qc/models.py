import tensorflow as tf
from .layers import *
from typing import *
from abc import ABCMeta, abstractmethod
from functools import reduce


class QCModel(tf.keras.Sequential):
    def matrix(self):
        # Either a is a layer OR it's the tensor from the previous eval
        def reduction(a: Union[QCLayer, tf.Tensor], b: QCLayer):
            return b.matrix() @ (a.matrix() if isinstance(a, QCLayer) else a)
        return reduce(reduction, self.layers)  # Note order of matmul


class ApproxUsingInverse(QCModel):
    def __init__(self, model_class: QCModel, target_inv_class: QCModel):
        super(ApproxUsingInverse, self).__init__()
        self.model = model_class
        self.target_inv = target_inv_class
        self.add(self.model)
        self.add(self.target_inv)

    def target_inv_matirx(self):
        return self.target_inv.matrix()

    def model_matrix(self):
        return self.model.matrix()


class PrePostQFTUIQFT(ApproxUsingInverse):
    def __init__(self):
        model = QCModel(layers=[
            U3Layer(),
            QFTULayer(),
            U3Layer()
        ])
        target = QCModel(layers=[
            IQFTLayer()
        ])
        super(PrePostQFTUIQFT, self).__init__(model, target)