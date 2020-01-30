from tf_qc import *
import matplotlib.pyplot as plt
import numpy as np


data = random_unifrom_complex((1000, 2**4, 1))
data = tf.reshape(data, [-1])
x = tf.math.real(data); y = tf.math.imag(data)
plt.figure()
plt.plot(x,y,'.',alpha=0.3)
plt.show()
