import tensorflow as tf
from typing import *
from functools import reduce

from .layers import QCLayer, U3Layer, QFTULayer, IQFTLayer, HLayer, ULayer, ISWAPLayer, ILayer, QFTCrossSwapLayer, \
    qft_cross_swap_layers, qft_cross_iswap_layers
from tensorflow.keras.layers import InputLayer
from abc import abstractmethod
from tf_qc import complex_type


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
            return self.matrix() @ other.matrix


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
    def all_swaps(self):
        return [
            ISWAPLayer([0, 2], parameterized=True),
            ISWAPLayer([0, 3], parameterized=True),
            ISWAPLayer([1, 2], parameterized=True),
            ISWAPLayer([1, 3], parameterized=True)
        ]

    @property
    def model_layers(self):
        return {
            'model_a': [
                U3Layer(),
                HLayer(0),
                ULayer(),
                HLayer(1),
                ULayer(),
                HLayer(2),
                ULayer(),
                HLayer(3),
                U3Layer(),
                QFTCrossSwapLayer()
            ],  # fid = 0.6069
            'model_a_swap': [
                U3Layer(),
                HLayer(0),
                *self.all_swaps(),
                HLayer(1),
                *self.all_swaps(),
                HLayer(2),
                *self.all_swaps(),
                HLayer(3),
                U3Layer(),
                QFTCrossSwapLayer()
            ],  # fid = 0.6125
            'model_a0': [
                HLayer(0),
                ULayer(),
                HLayer(1),
                ULayer(),
                HLayer(2),
                ULayer(),
                HLayer(3),
                QFTCrossSwapLayer()
            ],
            'model_a2': [
                U3Layer(),
                HLayer(0),
                ULayer(),
                HLayer(0),
                ULayer(),
                HLayer(1),
                ULayer(),
                HLayer(1),
                ULayer(),
                HLayer(2),
                ULayer(),
                HLayer(2),
                ULayer(),
                HLayer(3),
                ULayer(),
                HLayer(3),
                U3Layer(),
                QFTCrossSwapLayer()
            ],
            'model_b': [
                U3Layer(),
                HLayer(0),
                ISWAPLayer([0, 2], parameterized=True),
                ISWAPLayer([0, 3], parameterized=True),
                ULayer(),  # This U must couple 0 with {1,2,3}
                ISWAPLayer([0, 3], parameterized=True),
                ISWAPLayer([0, 2], parameterized=True),
                HLayer(1),
                ISWAPLayer([1, 2], parameterized=True),
                ISWAPLayer([1, 3], parameterized=True),
                ULayer(),  # This U must couple 1 with {2,3}
                ISWAPLayer([1, 3], parameterized=True),
                ISWAPLayer([1, 2], parameterized=True),
                HLayer(2),
                ULayer(),  # This U must couple 2 with 3
                HLayer(3),
                U3Layer(),
                QFTCrossSwapLayer()
            ],
            'model_c': [
                U3Layer(),
                HLayer(0),
                ISWAPLayer([0, 2], parameterized=True),
                ISWAPLayer([0, 3], parameterized=True),
                ISWAPLayer([0, 2], parameterized=True),
                ISWAPLayer([0, 3], parameterized=True),
                ULayer(),  # This U must couple 0 with {1,2,3}
                ISWAPLayer([0, 3], parameterized=True),
                ISWAPLayer([0, 2], parameterized=True),
                ISWAPLayer([0, 3], parameterized=True),
                ISWAPLayer([0, 2], parameterized=True),
                HLayer(1),
                ISWAPLayer([1, 2], parameterized=True),
                ISWAPLayer([1, 3], parameterized=True),
                ISWAPLayer([1, 2], parameterized=True),
                ISWAPLayer([1, 3], parameterized=True),
                ULayer(),  # This U must couple 1 with {2,3}
                ISWAPLayer([1, 3], parameterized=True),
                ISWAPLayer([1, 2], parameterized=True),
                ISWAPLayer([1, 3], parameterized=True),
                ISWAPLayer([1, 2], parameterized=True),
                HLayer(2),
                ULayer(),  # This U must couple 2 with 3
                HLayer(3),
                U3Layer(),
                QFTCrossSwapLayer()
            ],
            'model_d': [
                U3Layer(),

                HLayer(0),
                ULayer(),  # This U must couple 0 with {1,2,3}
                ISWAPLayer([0, 2], parameterized=True),
                ULayer(),
                ISWAPLayer([0, 2], parameterized=True),
                ISWAPLayer([0, 3], parameterized=True),
                ULayer(),
                ISWAPLayer([0, 3], parameterized=True),

                HLayer(1),
                ISWAPLayer([1, 2], parameterized=True),
                ULayer(),  # This U must couple 1 with {2,3}
                ISWAPLayer([1, 2], parameterized=True),
                ISWAPLayer([1, 3], parameterized=True),
                ULayer(),
                ISWAPLayer([1, 3], parameterized=True),

                HLayer(2),
                ULayer(),  # This U must couple 2 with 3

                HLayer(3),
                U3Layer(),
                QFTCrossSwapLayer()
            ],
            'model_e': [
                U3Layer(),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),
                QFTCrossSwapLayer()
            ],
            'model_e4': [
                U3Layer(),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),
                QFTCrossSwapLayer()
            ],
            'model_e10': [
                U3Layer(),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),
                QFTCrossSwapLayer()
            ],
            'model_f': [
                U3Layer(),
                *self.all_swaps(),
                ULayer(),
                U3Layer(),
                *self.all_swaps(),
                ULayer(),
                U3Layer(),
                *self.all_swaps(),
                ULayer(),
                U3Layer(),
                QFTCrossSwapLayer()
            ],
            'model_g': [
                U3Layer(),

                HLayer(0),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),

                HLayer(1),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),

                HLayer(2),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),

                HLayer(3),
                U3Layer(),
                QFTCrossSwapLayer()
            ],
            'model_g2': [
                U3Layer(),

                HLayer(0),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),

                HLayer(1),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),

                HLayer(2),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),
                ULayer(),
                U3Layer(),

                HLayer(3),
                U3Layer(),
                QFTCrossSwapLayer()
            ]  # Results: fid = 0.6042
        }

    def __init__(self, model_name):
        model = QCModel(layers=self.model_layers[model_name])
        target = QCModel(layers=[
            IQFTLayer()
        ])
        super(OneDiamondQFT, self).__init__(model, target, model_name)


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


class OneMemoryDiamondQFT(ApproxUsingInverse):
    @property
    def model_layers(self):
        targets = self.targets  # These are the qubits of the diamond
        ancilla_swap0 = lambda: ISWAPLayer([0, 4], parameterized=True)
        ancilla_swap1 = lambda: ISWAPLayer([1, 5], parameterized=True)
        ancilla_swap_both = lambda: [ancilla_swap0(), ancilla_swap1()]
        CT_swap_02_13 = lambda: [ISWAPLayer([0, 2], parameterized=True), ISWAPLayer([1, 3], parameterized=True)]
        ancilla_U3 = lambda: U3Layer([4, 5])
        return {
            'model_1_a': [
                InputLayer((2**6, 1), dtype=complex_type, name='input_state'),
                U3Layer(targets),
                HLayer(0),
                ULayer(targets),
                HLayer(1),
                ULayer(targets),
                HLayer(2),
                ULayer(targets),
                HLayer(3),
                U3Layer(targets),
                *qft_cross_swap_layers(targets)
            ],  # fid ~= 0.607(86) (lr = 0.05, bs = 120)
            'model_b': [
                InputLayer((2**6, 1), dtype=complex_type, name='input_state'),
                U3Layer(),
                HLayer(0),
                ancilla_swap0(),
                ancilla_swap1(),
                ULayer(targets),
                ancilla_swap1(),
                ancilla_swap0(),
                HLayer(1),
                ancilla_swap0(),
                ancilla_swap1(),
                ULayer(targets),
                ancilla_swap1(),
                ancilla_swap0(),
                HLayer(2),
                ancilla_swap0(),
                ancilla_swap1(),
                ULayer(targets),
                ancilla_swap1(),
                ancilla_swap0(),
                HLayer(3),
                U3Layer(),
                QFTCrossSwapLayer()
            ],
            'model_c': [
                InputLayer((2 ** 6, 1), dtype=complex_type, name='input_state'),
                U3Layer(),
                HLayer(0),
                ancilla_swap0(),
                ancilla_swap1(),
                ULayer(targets),
                ancilla_U3(),
                ancilla_swap1(),
                ancilla_swap0(),
                HLayer(1),
                ancilla_swap0(),
                ancilla_swap1(),
                ULayer(targets),
                ancilla_U3(),
                ancilla_swap1(),
                ancilla_swap0(),
                HLayer(2),
                ancilla_swap0(),
                ancilla_swap1(),
                ULayer(targets),
                ancilla_U3(),
                ancilla_swap1(),
                ancilla_swap0(),
                HLayer(3),
                U3Layer(),
                QFTCrossSwapLayer()
            ],
            # Adjust the controls and swap them in before the U-gate
            'model_d': [
                U3Layer(targets),
                HLayer(0),

                ancilla_U3(),
                *ancilla_swap_both(),
                ULayer(targets),
                *ancilla_swap_both(),
                ancilla_U3(),

                HLayer(1),

                ancilla_U3(),
                *ancilla_swap_both(),
                ULayer(targets),
                *ancilla_swap_both(),
                ancilla_U3(),

                HLayer(2),

                ancilla_U3(),
                *ancilla_swap_both(),
                ULayer(targets),
                *ancilla_swap_both(),
                ancilla_U3(),

                HLayer(3),
                U3Layer(targets),
                *qft_cross_swap_layers(targets)
            ],  # fid = 0.657(63) (lr = 0.05, bs = 120)
            'model_d3': [
                U3Layer(targets),
                HLayer(0),

                ancilla_U3(),
                *ancilla_swap_both(),
                ULayer(targets),
                *ancilla_swap_both(),
                ancilla_U3(),

                ancilla_U3(),
                *ancilla_swap_both(),
                ULayer(targets),
                *ancilla_swap_both(),
                ancilla_U3(),

                ancilla_U3(),
                *ancilla_swap_both(),
                ULayer(targets),
                *ancilla_swap_both(),
                ancilla_U3(),

                HLayer(1),

                ancilla_U3(),
                *ancilla_swap_both(),
                ULayer(targets),
                *ancilla_swap_both(),
                ancilla_U3(),

                ancilla_U3(),
                *ancilla_swap_both(),
                ULayer(targets),
                *ancilla_swap_both(),
                ancilla_U3(),

                ancilla_U3(),
                *ancilla_swap_both(),
                ULayer(targets),
                *ancilla_swap_both(),
                ancilla_U3(),

                HLayer(2),

                ancilla_U3(),
                *ancilla_swap_both(),
                ULayer(targets),
                *ancilla_swap_both(),
                ancilla_U3(),

                ancilla_U3(),
                *ancilla_swap_both(),
                ULayer(targets),
                *ancilla_swap_both(),
                ancilla_U3(),

                ancilla_U3(),
                *ancilla_swap_both(),
                ULayer(targets),
                *ancilla_swap_both(),
                ancilla_U3(),

                HLayer(3),
                U3Layer(targets),
                *qft_cross_swap_layers(targets)
            ],  # fid = 0.667(71) (lr = 0.05, bs = 120)
            'model_e': [
                U3Layer(targets),
                HLayer(0),

                ancilla_U3(),
                *CT_swap_02_13(),
                *ancilla_swap_both(),
                ULayer(targets),
                *ancilla_swap_both(),
                *CT_swap_02_13(),
                ancilla_U3(),

                HLayer(1),

                ancilla_U3(),
                *CT_swap_02_13(),
                *ancilla_swap_both(),
                ULayer(targets),
                *ancilla_swap_both(),
                *CT_swap_02_13(),
                ancilla_U3(),

                HLayer(2),

                ancilla_U3(),
                *CT_swap_02_13(),
                *ancilla_swap_both(),
                ULayer(targets),
                *ancilla_swap_both(),
                *CT_swap_02_13(),
                ancilla_U3(),

                HLayer(3),
                U3Layer(targets),
                *qft_cross_swap_layers(targets)
            ],  # fid = 0.676(57) (lr = 0.05, bs = 120)
            'model_e3': [
                U3Layer(targets),
                HLayer(0),

                ancilla_U3(),
                *CT_swap_02_13(),
                *ancilla_swap_both(),
                ULayer(targets),
                *ancilla_swap_both(),
                *CT_swap_02_13(),
                ancilla_U3(),
                *CT_swap_02_13(),
                *ancilla_swap_both(),
                ULayer(targets),
                *ancilla_swap_both(),
                *CT_swap_02_13(),
                ancilla_U3(),
                *CT_swap_02_13(),
                *ancilla_swap_both(),
                ULayer(targets),
                *ancilla_swap_both(),
                *CT_swap_02_13(),
                ancilla_U3(),

                HLayer(1),

                ancilla_U3(),
                *CT_swap_02_13(),
                *ancilla_swap_both(),
                ULayer(targets),
                *ancilla_swap_both(),
                *CT_swap_02_13(),
                ancilla_U3(),
                *CT_swap_02_13(),
                *ancilla_swap_both(),
                ULayer(targets),
                *ancilla_swap_both(),
                *CT_swap_02_13(),
                ancilla_U3(),
                *CT_swap_02_13(),
                *ancilla_swap_both(),
                ULayer(targets),
                *ancilla_swap_both(),
                *CT_swap_02_13(),
                ancilla_U3(),

                HLayer(2),

                ancilla_U3(),
                *CT_swap_02_13(),
                *ancilla_swap_both(),
                ULayer(targets),
                *ancilla_swap_both(),
                *CT_swap_02_13(),
                ancilla_U3(),
                *CT_swap_02_13(),
                *ancilla_swap_both(),
                ULayer(targets),
                *ancilla_swap_both(),
                *CT_swap_02_13(),
                ancilla_U3(),
                *CT_swap_02_13(),
                *ancilla_swap_both(),
                ULayer(targets),
                *ancilla_swap_both(),
                *CT_swap_02_13(),
                ancilla_U3(),  # Not strictly needed

                HLayer(3),
                U3Layer(targets),
                *qft_cross_swap_layers(targets)
            ]  # fid = 0.9756(46) (lr = 0.1, bs = 120)
        }

    def __init__(self, model_name):
        self.targets = [0, 1, 2, 3]
        model = QCModel(layers=self.model_layers[model_name])
        target = QCModel(layers=[
            IQFTLayer(self.targets)
        ])
        super(OneMemoryDiamondQFT, self).__init__(model, target, model_name)


class TwoMemoryDiamondQFT(ApproxUsingInverse):
    """
     8      10
     |      |
     0      4
    2 3 -- 6 7
     1      5
     |      |
     9      11
    """
    @property
    def model_layers(self):
        targets = self.targets  # These are the qubits of the diamond
        cross_swap = lambda: ISWAPLayer([3, 6], parameterized=True)
        return {
            'model_1_a_ref': [
                InputLayer((2 ** 12, 1), dtype=complex_type, name='input_state'),
                U3Layer([0, 1, 2, 3]),
                HLayer(0),
                ULayer([0, 1, 2, 3]),
                HLayer(1),
                ULayer([0, 1, 2, 3]),
                HLayer(2),
                ULayer([0, 1, 2, 3]),
                HLayer(3),
                U3Layer([0, 1, 2, 3]),
                *qft_cross_swap_layers([0, 1, 2, 3])
            ],
            'model_1_a': [
                InputLayer((2**12, 1), dtype=complex_type, name='input_state'),
                U3Layer(targets),
                HLayer(0),
                ULayer(targets),
                HLayer(1),
                ULayer(targets),
                HLayer(2),
                ULayer(targets),
                HLayer(3),
                ULayer(targets),
                HLayer(4),
                ULayer(targets),
                HLayer(5),
                ULayer(targets),
                HLayer(6),
                ULayer(targets),
                HLayer(7),
                U3Layer(targets),
                *qft_cross_swap_layers(targets)
            ],  # fid = BAD (lr = 0.001, bs = 25)
            'model_b': [
                InputLayer((2**12, 1), dtype=complex_type, name='input_state'),
                U3Layer(targets),
                HLayer(0, cached=False),
                cross_swap(),
                ULayer(targets),
                cross_swap(),
                HLayer(1, cached=False),
                cross_swap(),
                ULayer(targets),
                cross_swap(),
                HLayer(2, cached=False),
                cross_swap(),
                ULayer(targets),
                cross_swap(),
                HLayer(3, cached=False),
                cross_swap(),
                ULayer(targets),
                cross_swap(),
                HLayer(4, cached=False),
                cross_swap(),
                ULayer(targets),
                cross_swap(),
                HLayer(5, cached=False),
                cross_swap(),
                ULayer(targets),
                cross_swap(),
                HLayer(6, cached=False),
                cross_swap(),
                ULayer(targets),
                cross_swap(),
                HLayer(7, cached=False),
                U3Layer(targets),
                *qft_cross_swap_layers(targets)
            ],  # fid =  (lr = 0.05, bs = 25)
        }

    def __init__(self, model_name):
        self.targets = [0, 1, 2, 3, 4, 5, 6, 7]
        model = QCModel(layers=self.model_layers[model_name])
        target = QCModel(layers=[
            IQFTLayer(self.targets)
        ])
        super(TwoMemoryDiamondQFT, self).__init__(model, target, model_name)