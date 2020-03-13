import os
from typing import *
from contextlib import redirect_stdout
import tensorflow as tf
from .losses import Mean1mFidelity, StdFidelity
from .layers import QFTLayer, IQFTLayer
from txtutils import ndtotext, ndtotext_print
from tf_qc.models import ApproxUsingInverse
from tf_qc.metrics import OperatorFidelity, FidelityMetric, StdFidelityMetric
import datetime
import sys
import signal
from txtutils import ndtotext_print

def train(model: ApproxUsingInverse,
          input,
          output,
          optimizer: tf.optimizers.Optimizer,
          loss: tf.losses.Loss,
          log_path: Union[str, List[str]],
          epochs: int,
          batch_size=32,
          metrics=None):
    print('optimizer:', optimizer)
    print('batch_size:', batch_size)

    if isinstance(log_path, str):
        log_path = [log_path]
    current_time = datetime.datetime.now().replace(microsecond=0).isoformat()
    log_path.append(model.name)
    log_path.append(current_time)
    log_path = os.path.join(*log_path)
    os.makedirs(log_path, exist_ok=True)

    # Compile model
    model.compile(optimizer, loss=loss, metrics=metrics)
    print('# trainable variables:', len(model.trainable_variables))
    print(model.trainable_variables)

    # Fitting
    print('logs:', log_path)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1, profile_batch=0)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(log_path, 'checkpoint'), save_best_only=True)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(min_delta=0.0001, patience=100)
    early_stopping_high_loss_callback = tf.keras.callbacks.EarlyStopping(patience=20, baseline=0.8)
    early_stopping_high_loss_callback2 = tf.keras.callbacks.EarlyStopping(patience=45, baseline=0.7)
    cvs_logger_callback = tf.keras.callbacks.CSVLogger(os.path.join(log_path, 'log.cvs'), append=True)
    plateau_callback = tf.keras.callbacks.ReduceLROnPlateau()

    def signal_handler(sig, frame, exit=True):
        print('You pressed Ctrl+C! Priting logs...')
        print(*model.variables, sep='\n')
        # Write the summary to the log dir (so that we can reconstruct the model later on with the Variables
        with open(os.path.join(log_path, 'summary.txt'), 'w') as f:
            with redirect_stdout(f):
                model.summary()
        model.save(log_path)

        result = model.model_matrix()
        ndtotext_print(result)

        # Sanity check: test the QFT_U against QFT on all data
        qft_layer = QFTLayer()
        real_output = qft_layer(input)
        model_output = result @ input
        print('Sanity check loss:', loss(real_output, model_output).numpy())
        std_loss = StdFidelity()
        print('Sanity check loss std:', std_loss(real_output, model_output).numpy())
        hs_norm = OperatorFidelity(model)
        print('Hilbert–Schmidt/Frobenius norm:', hs_norm())
        # and check if it's the real inverse!
        targets = None
        if hasattr(model, 'targets'):
            targets = model.targets
        iqft_layer = IQFTLayer(targets)
        iqft_layer(input)
        eye = iqft_layer.matrix() @ model.matrix()
        print('Eye:')
        ndtotext_print(eye)
        if exit:
            sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    model.fit(input,
              output,
              validation_split=0.2,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[tensorboard_callback,
                         model_checkpoint_callback,
                         # early_stopping_callback,
                         # early_stopping_high_loss_callback,
                         # early_stopping_high_loss_callback2,
                         cvs_logger_callback,
                         plateau_callback])
    signal_handler(None, None, False)