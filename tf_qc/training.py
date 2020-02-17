import os
from typing import *
from contextlib import redirect_stdout
import tensorflow as tf
from .losses import Mean1mFidelity, StdFidelity
from .layers import QFTLayer, IQFTLayer
from txtutils import ndtotext
from tf_qc.models import ApproxUsingInverse
from tf_qc.metrics import OperatorFidelity
import datetime


def train(model: ApproxUsingInverse,
          input,
          output,
          optimizer: tf.optimizers.Optimizer,
          loss: tf.losses.Loss,
          log_path: Union[str, List[str]],
          epochs: int,
          batch_size=32):

    if isinstance(log_path, str):
        log_path = [log_path]
    current_time = datetime.datetime.now().replace(microsecond=0).isoformat()
    log_path.append(model.name)
    log_path.append(current_time)
    log_path = os.path.join(*log_path)
    os.makedirs(log_path, exist_ok=True)

    # Compile model
    oper_fid_metric = OperatorFidelity(model)
    model.compile(optimizer, loss=loss)#, metrics=[oper_fid_metric])

    # Fitting
    print('logs:', log_path)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1, profile_batch=0)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(log_path, 'checkpoint'), save_best_only=True)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(min_delta=0.0001, patience=100)
    early_stopping_high_loss_callback = tf.keras.callbacks.EarlyStopping(patience=20, baseline=0.8)
    early_stopping_high_loss_callback2 = tf.keras.callbacks.EarlyStopping(patience=45, baseline=0.7)
    cvs_logger_callback = tf.keras.callbacks.CSVLogger(os.path.join(log_path, 'log.cvs'), append=True)
    plateau_callback = tf.keras.callbacks.ReduceLROnPlateau()
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
    print(*model.variables, sep='\n')
    # Write the summary to the log dir (so that we can reconstruct the model later on with the Variables
    with open(os.path.join(log_path, 'summary.txt'), 'w') as f:
        with redirect_stdout(f):
            model.summary()
    model.save(log_path)

    result = model.model_matrix()
    print(ndtotext(result.numpy()))

    # Sanity check: test the QFT_U against QFT on all data
    qft_layer = QFTLayer()
    real_output = qft_layer(input)
    model_output = result @ input
    print('Sanity check loss:', loss(real_output, model_output).numpy())
    std_loss = StdFidelity()
    print('Sanity check loss std:', std_loss(real_output, model_output).numpy())
    # and check if it's the real inverse!
    iqft_layer = IQFTLayer()
    iqft_layer(input)
    eye = iqft_layer.matrix() @ model.matrix()
    print('Eye:')
    print(ndtotext(eye.numpy()))