import tensorflow as tf
from tf_qc import float_type
from tf_qc.models import ApproxUsingInverse

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
