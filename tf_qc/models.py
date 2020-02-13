import tensorflow as tf
from typing import *
from functools import reduce

from .layers import QCLayer, U3Layer, QFTULayer, IQFTLayer, HLayer, ULayer, ISWAPLayer, ILayer, QFTCrossSwapLayer
from abc import abstractmethod


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
    def __init__(self, model: QCModel, target_inv_model: QCModel, name=None):
        super(ApproxUsingInverse, self).__init__(name=name)
        self.model = model
        self.target_inv = target_inv_model
        self.add(self.model)
        self.add(self.target_inv)

    def target_inv_matrix(self):
        return self.target_inv.matrix()

    def model_matrix(self):
        return self.model.matrix()

    def summary(self, line_length=None, positions=None, print_fn=None):
        if print_fn is None:
            print_fn = print
        print('--- ApproxUsingInverse ---')
        print_fn('Model:')
        self.model.summary(line_length, positions, print_fn)
        print_fn('Target inverse:')
        self.target_inv.summary(line_length, positions, print_fn)


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
        super(PrePostQFTUIQFT, self).__init__(model, target, 'PrePostQFTUIQFT')


class OneDiamondQFT(ApproxUsingInverse):
    def __init__(self):
        def all_swaps():
            return [
                ISWAPLayer([0, 2], parameterized=True),
                ISWAPLayer([0, 3], parameterized=True),
                ISWAPLayer([1, 2], parameterized=True),
                ISWAPLayer([1, 3], parameterized=True)
            ]
        model = QCModel(layers=[
            U3Layer(),
            *all_swaps(),
            #ULayer(),
            U3Layer(),
            *all_swaps(),
            #ULayer(),
            U3Layer(),
            *all_swaps(),
            #ULayer(),
            U3Layer(),
            QFTCrossSwapLayer()
        ])
        target = QCModel(layers=[
            IQFTLayer()
        ])
        super(OneDiamondQFT, self).__init__(model, target, 'model_f_ux')


class TwoDiamondQFT(ApproxUsingInverse):
    def __init__(self):
        def u_iswap34_u_h(h_target):
            return [ULayer(), ISWAPLayer([3, 4]), ULayer(), ISWAPLayer([3, 4]), HLayer(h_target)]
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
        super(TwoDiamondQFT, self).__init__(model, target, 'model_a')


class OneDiamondISWAP(ApproxUsingInverse):
    def __init__(self):
        model = QCModel(layers=[
            U3Layer(),
            ULayer(),
            U3Layer()
        ])
        target = QCModel(layers=[
            ISWAPLayer([0, 1])  # It's its own inverse
        ])
        super(OneDiamondISWAP, self).__init__(model, target, 'model_01_a')
