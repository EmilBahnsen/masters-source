import tensorflow as tf
from typing import *
from functools import reduce

from tf_qc.layers import QCLayer, U3Layer, QFTULayer, IQFTLayer, HLayer, ULayer, ISWAPLayer, ILayer, QFTCrossSwapLayer


class QCModel(tf.keras.Sequential):
    def matrix(self):
        # Either 'a' is a layer OR it's the tensor from the previous eval
        def reduction(a: Union[QCLayer, QCModel, tf.Tensor], b: Union[QCLayer, QCModel]):
            if isinstance(a, tf.Tensor):
                return b.matrix() @ a
            else:
                return b.matrix() @ a.matrix()
        if len(self.layers) == 0:
            raise Exception('No layers to "matricify".')
        elif len(self.layers) == 1:
            return self.layers[0].matrix()
        else:
            return reduce(reduction, self.layers)  # Note order of matmul

    def __matmul__(self, other):
        if isinstance(other, tf.Tensor):
            return self.matrix() @ other
        else:
            return self.matrix() @ other.matrix()


class ApproxUsingInverse(QCModel):
    def __init__(self, model: QCModel, target_inv_model: QCModel):
        super(ApproxUsingInverse, self).__init__()
        self.model = model
        self.target_inv = target_inv_model
        self.add(self.model)
        self.add(self.target_inv)

    def target_inv_matrix(self):
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


class OneDiamondQFT(ApproxUsingInverse):
    def __init__(self):
        model = QCModel(layers=[
            U3Layer(),
            HLayer(0),

            # U3Layer(),
            ULayer(),
            # U3Layer(),

            HLayer(1),

            # U3Layer(),
            ULayer(),
            # U3Layer(),

            HLayer(2),

            # U3Layer(),
            ULayer(),
            # U3Layer(),

            HLayer(3),
            U3Layer(),
            QFTCrossSwapLayer()
        ])
        target = QCModel(layers=[
            IQFTLayer()
        ])
        super(OneDiamondQFT, self).__init__(model, target)


class TwoDiamondQFT(ApproxUsingInverse):
    def __init__(self):
        def u_iswap34_u_h(h_target):
            return [ULayer(), ISWAPLayer([3, 4]), ULayer(), HLayer(h_target)]
        model = QCModel(layers=[
            U3Layer(),
            HLayer(0),
            *u_iswap34_u_h(1),
            *u_iswap34_u_h(2),
            *u_iswap34_u_h(3),
            *u_iswap34_u_h(4),
            *u_iswap34_u_h(5),
            *u_iswap34_u_h(6),
            *u_iswap34_u_h(7),
            U3Layer()
        ])
        target = QCModel(layers=[
            IQFTLayer()
        ])
        super(TwoDiamondQFT, self).__init__(model, target)
