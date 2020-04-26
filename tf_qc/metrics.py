import tensorflow as tf

from tf_qc import float_type
from tf_qc.qc import fidelity
from tf_qc.models import ApproxUsingInverse
from tf_qc.losses import Mean1mFidelity


class OperatorFidelity(tf.keras.metrics.Metric):
    def __init__(self, model: ApproxUsingInverse):
        super(OperatorFidelity, self).__init__(name='operator_fidelity', dtype=float_type)
        self.model = model

    def update_state(self, *args, **kwargs):
        pass

    def result(self):
        matrix = self.model.matrix()
        d = matrix.shape[0]
        return tf.abs(tf.linalg.trace(matrix) / d) ** 2


class FidelityMetric(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        self.kwargs_fed_fun = kwargs
        super(FidelityMetric, self).__init__(name='fidelity', dtype=float_type)
        self.fid = tf.Variable(0.0)

    def update_state(self, y_true, y_pred, *args, **kwargs):
        res = tf.reduce_mean(fidelity(y_true, y_pred, **self.kwargs_fed_fun))
        self.fid.assign(res)

    def result(self):
        return self.fid

    def reset_states(self):
        self.fid.assign(0.0)


class StdFidelityMetric(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        self.kwargs_fed_fun = kwargs
        super(StdFidelityMetric, self).__init__(name='std_fidelity', dtype=float_type)
        self.std_fid = tf.Variable(0.0)

    def update_state(self, y_true, y_pred, *args, **kwargs):
        res = tf.math.reduce_std(fidelity(y_true, y_pred, **self.kwargs_fed_fun))
        self.std_fid.assign(res)

    def result(self):
        return self.std_fid

    def reset_states(self):
        self.std_fid.assign(0.0)